import pandas as pd
import numpy as np
import help_funcs

arrival_table = pd.read_csv('simulated_from_PC3_small_example.csv')

# Windowed periodogram and frequency estimation
obs = arrival_table.arrival_time_in_days.values.reshape((1, -1))
T = np.ceil(np.max(obs))
freq_grid = (np.arange(0, 10 * 365 + 1) / 365).reshape((1, -1))
a = obs.size / T
# periodogram_window = help_funcs.center_periodogram(T, obs, freq_grid, a)
# np.save("periodogram_window", periodogram_window)
periodogram_window = np.load("periodogram_window.npy")

# tau = help_funcs.tau_simulate(np.max(periodogram_window), T, obs.size / T, freq_grid)
# np.save("tau", tau)
tau = np.load("tau.npy")
tau_constant = 6

fitted_freq, a, c, d = help_funcs.lse_time_cont(obs, periodogram_window, freq_grid, tau_constant, T)
fitted_mag = np.sqrt(c**2 + d**2)
