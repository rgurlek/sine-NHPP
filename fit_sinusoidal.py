import pandas as pd
import numpy as np
import help_funcs
import os
os.chdir(r"poisson_python_code")

# # read data from csv files. mdb -> csv is done by mdbtools and then csv is
# # preprocessed in R
# arrival_table = pd.read_csv('C:/Users/rgurlek/Desktop/poisson_code/data/usbank/calls_business.csv')
# arrival_day = arrival_table.call_start / 24 / 3600  # in days
# arrival_day = arrival_day[arrival_table.entry_service_group != 1]  # VRU if ==1 not VRU if not 1
# arrival_day = arrival_day.sort_values()
#
# # Start from Apr 1 2001
# origin = pd.to_datetime("01-01-1970", format="%d-%m-%Y")
# startday = pd.to_datetime("01-04-2001", format="%d-%m-%Y")
# endday = pd.to_datetime("01-05-2002", format="%d-%m-%Y")
# arrival_day = arrival_day[(arrival_day > (startday - origin).days)
#                           & (arrival_day < (endday - origin).days)]
# arrival_day = arrival_day - np.floor(arrival_day.iloc[0])
# arrival_day = arrival_day.values

arrival_table = pd.read_csv('C:/Users/rgurlek/Desktop/USBank/simulated_from_PC1_with_trend.csv')



# Windowed periodogram and frequency estimation
obs = arrival_table.arrival_time_in_days.values.reshape((1, -1))
T = np.ceil(np.max(obs))
freq_grid = (np.arange(0, 10 * 365 + 1) / 365).reshape((1, -1))
a = obs.size / T
# periodogram_window = help_funcs.center_periodogram(T, obs, freq_grid, a)
# np.save("periodogram_window", periodogram_window)
periodogram_window = np.load("periodogram_window.npy")

# TODO: Debug tau function
# tau = help_funcs.tau_simulate(np.max(periodogram_window), T, obs.size / T, freq_grid)
# np.save("tau", tau)
tau = np.load("tau.npy")
tau_574 = tau + (0.0574 - 0.0181) * np.max(periodogram_window)

# TODO: Run everything and see if the results are similar to Matlab output

fitted_freq, a, c, d = help_funcs.lse_time_cont(obs, periodogram_window, freq_grid, tau_574, T)
fitted_mag = np.sqrt(c**2 + d**2)

import importlib
importlib.reload(help_funcs)