import random
import torch
import pyglet
import math
from gymnasium import spaces

from driving_sim.utils.trajectory import *
from driving_sim.envs.t_intersection_pred_front import TIntersectionPredictFront
from driving_sim.car import Car
from driving_sim.drivers.driver import Driver, XYSeperateDriver, YNYDriver, YNYDriverSocial, EgoDriver
from driving_sim.drivers.oned_drivers import IDMDriver, PDDriver
from driving_sim.constants import *
from driving_sim.utils.info import *

# same as TIntersectionPredictFront except that the ob includes each driver's actions in pretext_nodes key
# Usage: used for Morton et. al baseline


class TIntersectionRobustnessSocial(TIntersectionPredictFront):
    def __init__(self):
        super(TIntersectionRobustnessSocial, self).__init__()

        self.beta_delta = 12 / (10e+6)  # num of environment / timestep
        self.beta_base = self.beta_delta
        self.beta_range_min = 1.0
        self.beta_range_max = 1.0

        self.use_idm_social = False
        self.always_rl_social = True
        self.always_idm_social = False

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _reset(self):
        if self.always_rl_social:
            self.use_idm_social = False
        elif self.always_idm_social:
            self.use_idm_social = True
        else:
            self.use_idm_social = [False, True][random.randint(0, 1)]
        super(TIntersectionRobustnessSocial, self)._reset()
        # self._cars[0].set_velocity(np.array([0.0, 0.0]))
        self.ego_terminal = False
        # self.action_with_idx[int(driver._idx - 1)] = self._actions[i].a_x
        self.valid_training = np.zeros(self.max_veh_num)
        for car in self._cars[1:]:
            self.valid_training[int(car._idx - 1)] = 1.0
        self.objective = np.zeros((self.max_veh_num, 2))
        self._drivers[0].safe_control = self.safe_control
        self.collision_vehicle_type = [0.0, None]

    def configure(self, config, nenv=None):
        super(TIntersectionRobustnessSocial, self).configure(config)
        self.nenv = nenv

        self.safe_control = config.car.safe_control
        self.social_beta_only_collision = config.reward.social_beta_only_collision
        self.social_reward_only_collided = config.reward.social_reward_only_collided
        self.social_timestep_penalty = config.reward.stop_penalty_social

    @property
    def observation_space(self):
        d = {}
        # robot node: px, py
        d['robot_node'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # only consider all temporal edges (human_num+1) and spatial edges pointing to robot (human_num)
        d['temporal_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # edge feature will be (px - px_robot, py - py_robot, intent)
        d['spatial_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, self.spatial_ob_size + self.latent_size), dtype=np.float32)

        # observation for pretext latent state inference with S-RNN
        # nodes: the px for all cars except ego car
        d['pretext_nodes'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,), dtype=np.float32)
        # spatial edges: the delta_px, delta_vx from each car i to its front car
        d['pretext_spatial_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 2,), dtype=np.float32)
        # temporal edges: vx of all cars except ego car
        d['pretext_temporal_edges'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,), dtype=np.float32)
        # mask to indicate whether each id has a car present or a dummy car
        d['pretext_masks'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num,), dtype=np.float32)
        # mask to indicate whether each car needs to be inferred (based on its lane and the y position of ego car)
        d['pretext_infer_masks'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num,), dtype=np.int32)
        # the true label of each car (for debugging purpose)
        d['true_labels'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num,), dtype=np.float32)
        d['pretext_actions'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 1,), dtype=np.float32)

        d['front_car_information'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 2,), dtype=np.float32)
        d['objective_weight'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 2,), dtype=np.float32)
        d['social_car_information'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num, 3,), dtype=np.float32)
        d['valid_training'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_veh_num,), dtype=np.float32)

        return spaces.Dict(d)

    def step(self, action):

        self.beta_base += self.beta_delta
        if self.beta_base > 1.0:
            self.beta_base = 1.0

        # apply the ego car's action, compute the apply other cars' action to update all cars' states
        self.update(action)

        # update the current time and current step number
        self.global_time += self.dt
        self.step_counter = self.step_counter + 1

        obs = self.observe()

        reward = self.get_reward()

        done = self.is_terminal()

        info = self.get_info()

        return obs, reward, done, info

    def update(self, action):

        for i, driver in enumerate(self._drivers):

            if i == 0:
                model_action = action[driver._idx]
                rl_action = self.rl_actions[model_action]
                driver.v_des = rl_action[0]
                driver.t_des = rl_action[1]
            else:
                if not self.use_idm_social:
                    model_action = action[driver._idx]
                    rl_action = self.rl_actions[model_action]
                    driver.x_driver.v_des = rl_action[0]

        self._goal = False
        self._collision = False
        self._outroad = False

        for _ in range(self.num_updates):

            for i, driver in enumerate(self._drivers):
                if i == 0:
                    driver.observe(self._cars, self._road)
                else:
                    if self.use_idm_social:
                        driver.observe(self._cars, self._road)
                    else:
                        driver.y_driver.observe(self._cars, self._road)

            self._actions = [driver.get_action() for driver in self._drivers]

            # create an action list that preserves the id of each driver
            self.front_car_information = np.zeros((self.max_veh_num, 2))
            self.action_with_idx = np.zeros(self.max_veh_num)
            for i, driver in enumerate(self._drivers):
                if i == 0:  # skip the ego driver
                    continue

                driver.x_driver.observe(self._cars[1:], self._road)
                self.action_with_idx[int(driver._idx - 1)] = self._actions[i].a_x

            [action.update(car, self.dt) for (car, action) in zip(self._cars, self._actions)]

            ###### collision check (1. ego / 2. social / 3. if collision -> return)
            collision_return = False
            self._collision_social = np.zeros(self.max_veh_num, dtype=bool)
            self._speed_social = np.zeros(self.max_veh_num)
            self._pos_social = np.zeros(self.max_veh_num)
            self._gap_social = np.zeros(self.max_veh_num)
            self.objective = np.zeros((self.max_veh_num, 2))
            for car, driver in zip(self._cars[1:], self._drivers[1:]):
                self._pos_social[int(car._idx - 1)] = car.position[0]
                self._speed_social[int(car._idx - 1)] = car.velocity[0] * driver.x_driver.direction
                self._gap_social[int(car._idx - 1)] = driver.x_driver.get_front_relative_pos(self._cars)[0] * driver.x_driver.direction
                self.objective[int(car._idx - 1)] = driver.objective

                for car_target in self._cars:
                    if car == car_target:
                        continue
                    if car.check_collision(car_target):
                        self._collision_social[int(car._idx - 1)] = True
                        collision_return = True
                        self.collided_veh_id = car._idx - 1

            ego_car = self._cars[0]
            for car in self._cars[1:]:
                if ego_car.check_collision(car):
                    self._collision = True
                    collision_return = True
                    self.collided_veh_id = car._idx - 1
                    self.collision_vehicle_type = self.objective[int(car._idx - 1)]
                    self._collision_social[int(car._idx - 1)] = True

            # deadlock check (deadlock == collision)
            if 0.01 < ego_car.position[0] < 2.0:
                for car in self._cars[1:]:
                    if 0.01 < car.position[0] < 6.0 and 1.5 < car.position[1] < 2.5:
                        self._collision = True
                        collision_return = True
                        self.collided_veh_id = car._idx - 1
                        self.collision_vehicle_type = self.objective[int(car._idx - 1)]
                        self._collision_social[int(car._idx - 1)] = True

            if collision_return:
                return

            if not self._road.is_in(ego_car):
                self._outroad = True
                return

            ###### reached goal check (1. ego / 2. social / 3. if collision -> return)
            if (ego_car.position[0] > 8.) \
                and (ego_car.position[1] > 5.) \
                and (ego_car.position[1] < 7.):
                self._goal = True
                return

            # remove cars that are out-of bound
            self._reached_goal_social = np.zeros(self.max_veh_num, dtype=bool)
            for car, driver in zip(self._cars[1:], self._drivers[1:]):
                if (car.position[1] < 4.) and (car.position[0] < self.left_bound):
                    self._reached_goal_social[int(car._idx - 1)] = True
                    self.remove_car(car, driver)
                    removed_lower = True
                elif (car.position[1] > 4.) and (car.position[0] > self.right_bound):
                    self._reached_goal_social[int(car._idx - 1)] = True
                    self.remove_car(car, driver)
                    removed_upper = True

            # add cars when there is enough space
            # 1. find the right most car in the lower lane and the left most car in the upper lane and their idx
            # i.e. the cars that entered most recently in both lanes
            min_upper_x = np.inf
            max_lower_x = -np.inf
            for car in self._cars[1:]:
                if (car.position[1] < 4.) and (car.position[0] > max_lower_x):
                    max_lower_x = car.position[0]
                if (car.position[1] > 4.) and (car.position[0] < min_upper_x):
                    min_upper_x = car.position[0]

            # add a car to both lanes if there is space

            # lower lane
            # decide the new car's yld = True or False
            new_yld, gap_min, gap_max = self.init_trait()
            # condition for add a new car
            cond = max_lower_x < (self.right_bound - np.random.uniform(gap_min, gap_max) - self.car_length)
            # desired x location for the new car
            x = self.right_bound

            if cond:
                v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
                p_des = 2.
                direction = -1

                car, driver = self.add_car(x, 2., -v_des, 0., v_des, p_des, direction, np.pi, new_yld)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer, awr=True)
                    driver.setup_render(self.viewer)

            # upper lane
            new_yld, gap_min, gap_max = self.init_trait()
            # condition for adding a new car
            cond = min_upper_x > (self.left_bound + np.random.uniform(gap_min, gap_max) + self.car_length)
            # desired x location for the new car
            x = self.left_bound

            if cond:
                v_des = self.desire_speed + np.random.uniform(-1,1)*self.v_noise
                p_des = 6.
                direction = 1

                car, driver = self.add_car(x, 6., v_des, 0., v_des, p_des, direction, 0., new_yld)
                if hasattr(self, 'viewer') and self.viewer:
                    car.setup_render(self.viewer, awr=True)
                    driver.setup_render(self.viewer)

    # given the pose of a car, initialize a Car & a Driver instance and append them to self._cars & self._drivers
    def add_car(self, x, y, vx, vy, v_des, p_des, direction, theta, yld):
        # P(conservative)
        if yld:
            self.con_count = self.con_count + 1
        else:
            self.agg_count = self.agg_count + 1

        if y < 4.:
            idx = self._lower_lane_next_idx
            self._lower_lane_next_idx += 1
            if self._lower_lane_next_idx > int(self.max_veh_num / 2.):
                self._lower_lane_next_idx = 1
        elif y > 4.:
            idx = self._upper_lane_next_idx
            self._upper_lane_next_idx += 1
            if self._upper_lane_next_idx > self.max_veh_num:
                self._upper_lane_next_idx = int(self.max_veh_num / 2.) + 1

        car = Car(idx=idx, length=self.car_length, width=self.car_width, color=random.choice(RED_COLORS),
                  max_accel=self.car_max_accel, max_speed=self.car_max_speed, max_rotation=0.,
                  expose_level=self.car_expose_level)
        if self.use_idm_social:
            driver = YNYDriver(idx=idx, car=car, dt=self.dt,
                               x_driver=IDMDriver(idx=idx, car=car, sigma=self.driver_sigma, s_des=self.s_des,
                                                  s_min=self.s_min, axis=0, min_overlap=self.min_overlap, dt=self.dt),
                               y_driver=PDDriver(idx=idx, car=car, sigma=0., axis=1, dt=self.dt))
        else:
            driver = YNYDriverSocial(idx=idx, car=car, dt=self.dt,
                                     x_driver=IDMDriver(idx=idx, car=car, sigma=self.driver_sigma, s_des=self.s_des,
                                                        s_min=self.s_min, axis=0, min_overlap=self.min_overlap, dt=self.dt),
                                     y_driver=PDDriver(idx=idx, car=car, sigma=0., axis=1, dt=self.dt))
        car.set_position(np.array([x, y]))
        car.set_init_position(np.array([x, y]))
        car.set_velocity(np.array([vx, vy]))
        car.set_rotation(theta)
        driver.x_driver.set_v_des(v_des)
        driver.x_driver.set_direction(direction)
        driver.y_driver.set_p_des(p_des)

        # theta = random.uniform(-3.0, 3.0)
        # theta = random.randint(-1, 3)
        # theta = random.sample([-1., -0.5,  0.,  0.5,  1.,  1.5,  2.,  2.5,  3.], 1)[0]
        theta = -1.0
        reward_object = [1.0, theta]
        driver.set_objective(reward_object)
        driver.set_yld(yld)

        self._cars.append(car)
        self._drivers.append(driver)

        self.car_present[int(idx - 1)] = True
        self.car_lane_info[int(idx - 1)] = direction

        return car, driver

    def init_trait(self, reset=False):

        if self.use_idm_social:
            new_yld, gap_min, gap_max = super().init_trait(reset)
        else:
            new_yld = True
            gap_min = 3.
            gap_max = 6.

        return new_yld, gap_min, gap_max

    def observe(self, normalize=True):
        obs = super().observe(normalize)
        # add normalized actions of other cars
        obs['pretext_actions'] = self.action_with_idx.reshape((self.max_veh_num, 1)) / 9.

        obs['front_car_information'] = np.zeros((self.max_veh_num, 2))
        obs['objective_weight'] = np.zeros((self.max_veh_num, 2))
        obs['social_car_information'] = np.zeros((self.max_veh_num, 3))
        for i, car in enumerate(self._cars):
            if i == 0:
                continue
            else:

                front_pos, front_vel = self._drivers[i].x_driver.get_front_relative_x_pos_vel(self._cars[1:], True)
                if normalize:  # normalize to [-1, 1]
                    front_pos = front_pos / self.right_bound
                    front_vel = front_vel / self.desire_speed
                obs['front_car_information'][int(car._idx - 1)] = \
                    np.array([front_pos, front_vel]) * self._drivers[i].x_driver.direction

                if not self.use_idm_social:
                    obs['objective_weight'][int(car._idx - 1)] = np.array(self._drivers[i].objective)

                pos = self._cars[i].position
                vel = self._cars[i].velocity[:1]
                if normalize:
                    pos = pos / self.right_bound
                    vel = vel / self.desire_speed
                obs['social_car_information'][int(car._idx - 1)] = np.concatenate([pos, vel])

        obs['valid_training'] = np.zeros(self.max_veh_num)
        for car in self._cars[1:]:
            obs['valid_training'][int(car._idx - 1)] = 1.0

        return obs

    def is_terminal(self):
        return (sum(self._collision_social) + self._collision > 0) or self._goal or self.global_time >= self.time_limit

    # get the info
    def get_info(self):
        info = {}

        if self.global_time >= self.time_limit:
            info['info'] = Timeout()
        elif self._collision:
            info['info'] = Collision()
        elif self._outroad:
            info['info'] = OutRoad()
        elif self._goal:
            info['info'] = ReachGoal()
        else:
            info['info'] = Nothing()

        info['social_reward'] = self.get_social_reward()
        info['social_done'] = self.is_social_terminal()
        info['collision_vehicle_type'] = self.collision_vehicle_type
        info['collision'] = self.collided_veh_id
        info['idm_or_rl'] = self.use_idm_social

        return info

    # reward is torch.tensor type
    def get_social_reward(self):

        # personal reward is defined by combination of followings:
        # 1. reached goal : self.reached_goal_social
        # 2. speed reward
        # 3. gap penalty
        # 4. collision penalty

        reward_social = torch.zeros(self.max_veh_num)
        for i in range(self.max_veh_num):
            weight_itself, weight_ego = self.objective[i]

            speed = 0.05 * self._speed_social[i] / 3.0
            goal = 3.0 * self._reached_goal_social[i]
            time_penalty = -0.0
            collision_penalty = -3.0 * self._collision_social[i]
            block_penalty = 0.0
            if self._gap_social[i] >= 18.0:
                if self._speed_social[i] < 1.0:
                    block_penalty = -0.1
            social_reward = speed + goal + time_penalty + collision_penalty + block_penalty

            ego_reward = 0.05 * np.linalg.norm([self._cars[0].velocity]) / 3.0 - 0.0
            if self._collision_social[i]:
                ego_reward -= 3.0

            reward_social[i] = weight_itself * social_reward + weight_ego * ego_reward

        return reward_social

    def is_social_terminal(self):
        is_social_terminal = np.zeros(self.max_veh_num, dtype=bool)
        for i in range(self.max_veh_num):
            is_social_terminal[i] = self._collision_social[i] or self._reached_goal_social[i]
        return is_social_terminal

    def get_reward(self):
        if not self.ego_terminal:
            reward = 0.
            ego_car = self._cars[0]
            v_x, v_y = ego_car.velocity[0], ego_car.velocity[1]

            if self._collision:
                reward -= self.collision_cost
                self.ego_terminal = True
            elif self._outroad:
                reward -= self.outroad_cost
            elif self._goal:
                reward += self.goal_reward
                self.ego_terminal = True
            else:
                # add reward for larger speeds & a small constant penalty to discourage the ego car from staying in place
                reward = reward - 0.0 + 0.05 * np.linalg.norm([v_x, v_y]) / 3
                # reward = reward + 0.01 * np.linalg.norm([v_x, v_y]) / 3

            # print('reward:', reward)
            return reward

        else:
            return 0.0

    def render(self, mode='human', screen_size=800, extra_input=None):
        if (not hasattr(self, 'viewer')) or (self.viewer is None):
            self.setup_viewer()

            self._road.setup_render(self.viewer)

            for driver in self._drivers:
                driver.setup_render(self.viewer)

            for idx, car in enumerate(self._cars):
                if idx == 0:
                    car.setup_render(self.viewer)
                else:
                    car.setup_render(self.viewer, awr=True)

        camera_center = np.array([0.0, 0.0])
        self._road.update_render(camera_center)

        # infer_mask = self.fill_infer_masks()
        for i, driver in enumerate(self._drivers):
            # visualize the cars that satisfy the data collection condition
            driver.update_render(camera_center, collected=False)

            if not self.use_idm_social:
                if i != 0:
                    obj = driver.objective[-1]

                    if driver.objective[-1] > 0.0:  # green
                        driver.car._color = [*(0, 0.6, 0.4), 1.0]
                    else:  # red
                        driver.car._color = [*(0.85, 0.12, 0.09), 0.5]

                    if 3 <= obj:
                        driver.car._color = [*(2 / 255, 200 / 255, 120 / 255), 1.0]
                    elif 2 <= obj:
                        driver.car._color = [*(2 / 255, 140 / 255, 80 / 255), 1.0]
                    elif 1 <= obj:
                        driver.car._color = [*(2 / 255, 140 / 255, 80 / 255), 1.0]
                    elif 0 <= obj:
                        driver.car._color = [*(140 / 255, 2 / 255, 2 / 255), 1.0]
                    elif -1 <= obj:
                        driver.car._color = [*(140 / 255, 2 / 255, 2 / 255), 1.0]
                    elif -2 <= obj:
                        driver.car._color = [*(200 / 255, 2 / 255, 2 / 255), 1.0]
                    else:
                        driver.car._color = [*(200 / 255, 2 / 255, 2 / 255), 1.0]



        # label = pyglet.text.Label('Hello, world', x=0.0, y=0.0, font_size=100)
        # label.draw()

        for cid, car in enumerate(self._cars):
            car.update_render(camera_center)

        self.update_extra_render(extra_input, camera_center)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
