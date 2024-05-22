import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


class LowPassFiter:

    def __init__(self, dt, om_c, n):
        self.om_c = om_c
        self.K = 2 / dt
        self.prev_signal = np.zeros((n,))
        self.prev_output = np.zeros((n,))

    def filter(self, curr_signal):
        output = (self.om_c * (curr_signal + self.prev_signal) + (self.K - self.om_c) * self.prev_output) / (self.K + self.om_c)
        self.prev_signal = curr_signal
        self.prev_output = output
        return output


filter_1 = LowPassFiter(0.1, 0.5, 1)
filter_2 = LowPassFiter(0.1, 0.5, 1)
filter_3 = LowPassFiter(0.1, 0.5, 1)

ego_uniform = np.array(pd.read_csv('../data/experiment_1/rl_ego_uniform-13/progress.csv')['eprewmean'])
rl_ego_05_05 = np.array(pd.read_csv('../data/experiment_1/rl_ego_05_05/progress.csv')['eprewmean'])
rl_ego_15_05 = np.array(pd.read_csv('../data/experiment_1/rl_ego_15_05/progress.csv')['eprewmean'])


ego_uniform_filtered = np.zeros(ego_uniform.shape)
rl_ego_05_05_filtered = np.zeros(rl_ego_05_05.shape)
rl_ego_15_05_filtered = np.zeros(rl_ego_15_05.shape)
for i, (x_1, x_2, x_3) in enumerate(zip(ego_uniform, rl_ego_05_05, rl_ego_15_05)):
    ego_uniform_filtered[i] = filter_1.filter(x_1)
    rl_ego_05_05_filtered[i] = filter_2.filter(x_2)
    rl_ego_15_05_filtered[i] = filter_3.filter(x_3)


plt.figure()
plt.plot(ego_uniform_filtered, 'g', label="Smoothed GEP")
plt.plot(ego_uniform, 'g', label="GEP", alpha=0.2)
plt.plot(rl_ego_05_05_filtered, 'r', label="Smoothed OEP")
plt.plot(rl_ego_05_05, 'r', label="OEP", alpha=0.2)
plt.plot(rl_ego_15_05_filtered, 'b', label="Smoothed NEP")
plt.plot(rl_ego_15_05, 'b', label="NEP", alpha=0.2)
plt.xlabel("Training episode")
plt.ylabel("Mean reward")
plt.grid()
plt.xlim([0, len(ego_uniform)])
plt.ylim([-0.5, 5])
plt.legend()


plt.savefig('training_curves.png')
plt.show()

