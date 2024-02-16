import numpy as np


# defines the trajectory of the ego car
class EgoTrajectory:
    """ Translates between trajectory coordinates (s, t) and cartesian coordinates (x, y)
            Trajectory is defined to be a right-turning motion with the turn starting at (x, y) = (0, 0)
            turning radius is 6m, from (x, y) = (0, 0) to (x, y) = (6, 6)
            before the turn: car is on a straight trajectory parallel to y-axis
            after the turn: car is on a straight trajectory parallel to x-axis
        - s: distance travelled along trajectory
        - t: transverse distance from trajectory (side-ways deviation)
        - theta: orientation angle from position x-axis
        - curve: curvature of current part of trajectory (1/radius)
    """

    def xy_to_traj(self, pos):
        """ Translates cartesian coordinates (x, y) to trajectory coordinates (s, t)

            params: pos (x, y) - cartesian coordinates   [np.array]
            return: (s, t, theta, curv) - trajectory coordinates, and rotation and curvature
        """
        x, y = pos[0], pos[1]
        r = 6 # 4.5
        if y < 0.:
            # Car is before the turning motion
            s = y
            t = -x
            theta = np.pi/2
            curv = 0.
        elif x > r:
            # Car is beyond the turning motion
            s = r*np.pi/2. + x - r
            t = y - r
            theta = 0.
            curv = 0.
        else:
            # Car is in turning motion
            theta = np.arctan2(r-x, y)
            curv = 1./r
            s = r*(np.pi/2.-theta)
            t = np.sqrt((r-x)**2 + y**2) - r

        return s, t, theta, curv

    def traj_to_xy(self, pos):
        """ Translates trajectory coordinates (s, t) to cartesian coordinates (x, y)

            params: pos (s, t) - coordinates along trajectory   [np.array]
            return: (x, y, theta, curv) - cartesian coordinates, and rotation and curvature
        """
        s, t = pos[0], pos[1]
        r = 6 # 4.5
        if s < 0.:
            # Car is before the turning motion
            x = -t                      # side-ways deviation in x-direction with t [m]
            y = s                       # position along y-direction
            theta = np.pi/2             # orientation in positive y-direction
            curv = 0.                   # driving on straight road
        elif s > r*np.pi/2.:
            # Car is beyond the turning motion
            x = r + s - r*np.pi/2.      # position along x-direction
            y = r + t                   # side-ways deviation in y-direction with t [m]
            theta = 0.                  # orientation in positive x-direction
            curv = 0.                   # driving on straight road
        else:
            # Car is in turning motion
            theta = np.pi/2 - s/r       # orientation along curve
            curv = 1./r                 # curvature of turning motion
            x = r - (r+t)*np.sin(theta)
            y = (r+t)*np.cos(theta)

        return x, y, theta, curv
