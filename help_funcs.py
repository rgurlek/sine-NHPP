import numpy as np
import pandas as pd
np.random.seed(123)
from scipy.integrate import quad, simps

def center_periodogram(T, obs, freq_grid, a):
    def hann(t):
        return (1 - np.cos(2 * np.pi * t / T)) / 2

    def hann_ft(v):
        return (T / 2 * np.exp(np.array(-1j) * np.pi * T * v) * np.sinc(T * v) / (
                1 - (T * v * (np.abs(v) != (1 / T))) ** 2) *
                (np.abs(v) != (1 / T)) - T / 4 * (np.abs(v) == 1 / T))

    matrix_size = obs.size * freq_grid.size
    if matrix_size < 1000000000:
        periodogram = 1 / T * np.abs((np.sum(np.repeat(hann(obs), repeats = freq_grid.size, axis = 0) *
                                             np.exp(np.array(-1j) * 2 * np.pi * freq_grid.T * obs), axis = 1)).T -
                                     a * hann_ft(freq_grid))
    else:
        n_sub_vectors = round(matrix_size / 1000000000 * 2)
        indices = np.linspace(start = 0, stop = freq_grid.size, num = n_sub_vectors, endpoint = False).round()
        indices = np.delete(indices, 0).astype(int)

        def sub_periodogram(sub_vector):
            return (1 / T * np.abs((np.sum(np.repeat(hann(obs), repeats = sub_vector.size, axis = 0) *
                                           np.exp(np.array(-1j) * 2 * np.pi * sub_vector.T * obs), axis = 1)).T -
                                   a * hann_ft(sub_vector)))

        freq_grid = np.split(freq_grid, indices, axis = 1)
        periodogram = [sub_periodogram(sub_vector) for sub_vector in freq_grid]
        periodogram = np.concatenate(periodogram, axis = 1)

    return periodogram


def tau_simulate(max_h, T, lambd, freq_grid):
    # compute the threshold tau by simulating an HPP
    obs = generate_data(T, np.zeros([1, 1]), np.zeros([1, 1]), np.zeros([1, 1]), lambd) # Homogeneous Poisson Process
    p = center_periodogram(T, obs, freq_grid, lambd)
    threshold = 0.0181 * max_h + 1.06 * np.max(p)
    return threshold


def generate_data(T, freq, phase, mag, const_term):
    ub = const_term + sum(np.abs(mag))  # upper bound of the rate
    cos_coef = np.cos(phase) * mag
    sin_coef = -np.sin(phase) * mag
    tt = 0
    obs = np.array([])
    while tt < T:
        tt = tt + np.random.exponential(1 / ub)
        if np.random.uniform(0, 1) < (rate(tt, freq, const_term, cos_coef, sin_coef) / ub):
            obs = np.append(obs, tt)
    obs = np.delete(obs, -1)
    return obs.reshape((1, -1))


def     rate(t, freq, constant, cos_coef, sin_coef):
    # generate the rate at time t, with frequency, phase, and magnitude generated in the .m file vectorize the t argument
    return (constant +
            np.dot(cos_coef, np.cos(2 * np.pi * freq.T * t)) +
            np.dot(sin_coef, np.sin(2 * np.pi * freq.T * t)))


def lse_time_cont(obs, periodogram, freq_grid, tau, T):
    # This function estimates the frequencies and their magnitudes from the time stamps(obs),
    # their computed periodogram(periodogram), given frequency grid(freq_grid), the threshold(tau), and T
    # The output is an array of frequencies(fitted_freq), a, c, d are in a + c_i cos(2 pi freq_i t)+d_i sin(2 pi freq_i t)

    # Get the indices of frequencies that have bigger density than its immediate neighbours and that are bigger than
    # the two thresholds tau and 2/T.
    ind = (periodogram > np.concatenate((periodogram[:,1:], [[0]]), axis = 1)) * \
          (periodogram > np.concatenate(([[0]], periodogram[:,0:-1]), axis = 1)) * \
           (freq_grid > (2 / T)) * (periodogram > tau)
    ind = np.nonzero(ind)[1]

    I = periodogram[:, ind].argsort(axis = 1).flatten()
    I = ind[I]  # I is the positions of the frequencies that satisfy the criteria. It is sorted according to periodogram  values(descending).
    ind = []
    while I.size > 0:
        ind.append(I[0]) # Append the index with the biggest density to ind.
        I = I[np.abs(freq_grid[:, I] - freq_grid[:, I[0]]).flatten() > (2 / T)] # Remove that index and its neighbours
    fitted_freq = freq_grid[:, ind]
    fitted_freq = np.unique(np.concatenate(
        [fitted_freq.flatten(), np.array([1,2,3,4,5,6]) / (T + T * 0.3)]))
    freq_num = fitted_freq.size

    freq_double = np.concatenate([[0], fitted_freq, -fitted_freq])
    xty = np.zeros((freq_double.size, 1), dtype=np.complex_) # x^T * y vector
    for j in range(freq_double.size):
        xty[j,0] = np.exp(-freq_double[j] * 2 * np.pi * obs * np.array(1j)).sum()

    xtx = np.zeros((freq_double.size, freq_double.size), dtype=np.complex_) # X^T * X
    for i1 in range(freq_double.size):
        for i2 in range(freq_double.size):
            myeta = freq_double[i1] - freq_double[i2]
            xtx[i1, i2] = T * np.exp(np.array(-1j) * np.pi * T * myeta) * np.sinc(T * myeta)

    coef_est = np.linalg.solve(xtx, xty)
    constant = np.abs(coef_est[0])
    c = 2 * np.real(coef_est[1:(1+freq_num)]).T
    d = -2 * np.imag(coef_est[1:(1+freq_num)]).T

    amplitude = np.sqrt(c ** 2 + d ** 2)
    phase = np.arctan(d/c)

    fitted_params = pd.DataFrame(np.concatenate([
        fitted_freq.reshape((-1,1)),
        amplitude.reshape((-1,1)),
        phase.reshape((-1,1))
    ], axis = 1), columns = ['freq', 'amplitude', 'phase'])

    return constant, fitted_params