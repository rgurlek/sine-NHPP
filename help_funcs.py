import numpy as np
import pandas as pd
np.random.seed(123)
from scipy.integrate import quad, simps

def center_periodogram(T, obs, freq_grid, a):
    def hann(t): # Equation 9 - Left
        return (1 - np.cos(2 * np.pi * t / T)) / 2 # sin^2(x) cos(2x) transformation

    def hann_ft(v): # Equation 9 - Right
        return (T / 2 * np.exp(np.array(-1j) * np.pi * T * v) * np.sinc(T * v) / (
                1 - (T * v * (np.abs(v) != (1 / T))) ** 2) *
                (np.abs(v) != (1 / T)) - T / 4 * (np.abs(v) == 1 / T))

    matrix_size = obs.size * freq_grid.size
    if matrix_size < 1000000000: # Equation 8
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
    threshold = 0.0181 * max_h + 1.06 * np.max(p) # Equation 10. The first coefficient
    # is not the same as the one from the paper but it is corrected in USbank script
    # that calls this function.
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
    # the two thresholds tau and 3/T.
    ind = np.logical_and(periodogram > np.concatenate((periodogram[1:], [[0]]), axis = 1),
                         periodogram > np.concatenate(([[0]], periodogram[0:-2]), axis = 1),
                         freq_grid > (3 / T),
                         periodogram > tau)
    ind = np.nonzero(ind).flatten()
    # TODO: Check the dimensions of periodogram and ind. Also the subsettings here


    I = periodogram[:, ind].argsort(axis = 1)[::-1]
    I = ind[I]  # I is the list the frequencies that satisfy the criteria. The list is sorted (descending).
    ind = []
    while I.size > 0:
        ind.append(I[0]) # Append the index with biggest density to ind.
        I = I[np.abs(freq_grid[:, I] - freq_grid[:, I[0]]) > (3 / T)] # Remove that index and its neighbours
    fitted_freq = freq_grid[:, ind]
    freq_num = fitted_freq.size

    freq_double = np.concatenate([[0]], fitted_freq, -fitted_freq, axis = 1)
    xty = np.zeros((freq_double.size, 1)) # x^T * y vector
    for j in range(freq_double.size):
        xty[j,1] = sum(np.exp(-freq_double[1,j] * 2 * np.pi * obs * np.array(1j)))

    xtx = np.zeros((freq_double.size, freq_double.size)) # X^T * X
    for i1 in range(freq_double.size):
        for i2 in range(freq_double.size):
            myeta = freq_double[i1] - freq_double[i2]
            xtx[i1, i2] = T * np.exp(np.array(-1j) * np.pi * T * myeta) * np.sinc(T * myeta)

    coef_est = np.linalg.solve(xtx, xty)
    a = np.abs(coef_est[0])
    c = 2 * np.real(coef_est[1:]).T
    # TODO: Matlab code has "2:(1+freq_num)". Check the dimension of coef_est and see if what I write make sense
    d = -2 * np.imag(coef_est[1:]).T

    return fitted_freq, a, c, d


def generate_data1_new(t1, t2, freq, const_term, cos_coef, sin_coef):
    # Using cos and sin, not mag and phase.
    # Generate the arrival process given freq, phase, mag, const_term,  Generate arrival data between t1 and t2
    ub = 3 * const_term  # upper bound of the rate
    exp_inter = np.random.exponential(1 / ub, (1, int(np.floor((t2 - t1) * ub * 2))))
    while np.sum(exp_inter) < (t2 - t1):
        exp_inter = np.random.exponential(1 / ub, (1, int(np.floor((t2 - t1) * ub * 2))))
    exp_cum = np.cumsum(exp_inter)
    exp_cum = exp_cum[exp_cum < (t2 - t1)].reshape(1, -1)
    pois_thinning_rate = rate(t1 + exp_cum, freq, const_term, cos_coef, sin_coef)
    pois_thinning = np.random.binomial(1, pois_thinning_rate / ub)
    obs = exp_cum[pois_thinning == 1] + t1
    return obs

def m_t_sine(params, mu, t):
    freq = params.Frequency.values
    constant = params.Constant[0]
    cos_coef = params.Cos.values
    sin_coef = params.Sin.values
    gamma = 2 * np.pi * freq

    m_t = 1 / mu * (constant + sum(cos_coef * (
            np.cos(gamma * t) * mu ** 2 / (mu ** 2 + gamma ** 2) + np.sin(gamma * t) * mu * gamma / (
                mu ** 2 + gamma ** 2))) + sum(sin_coef * (
            np.sin(gamma * t) * mu ** 2 / (mu ** 2 + gamma ** 2) - np.cos(gamma * t) * mu * gamma / (
                mu ** 2 + gamma ** 2))))

    return m_t

def m_t_sine_general_arrival(params, service_time_cdf, start, end, look_back):
    freq = params.Frequency.values
    constant = params.Constant[0]
    cos_coef = params.Cos.values
    sin_coef = params.Sin.values

    def m_t_sub(t):
        def int_fun(u):
            # Service time cdf function is in seconds. Start/end uses day as time unit
            return (1 - service_time_cdf((t-u)*24*60*60)) * rate(u, freq, constant, cos_coef, sin_coef)
        # Any u bigger than look_back will yield 0 for 1 - CDF
        x = np.linspace(t-look_back, t, 10000)
        y = [int_fun(u) for u in x]
        m_t = simps(y, x)
        return m_t

    return max([m_t_sub(t) for t in np.linspace(end, start, 10)])

def m_t_PC(lamb, mu, t):
    # lamb is an array for lambda(t). It should have arrival rate predictions for non-Sunday days.
    return 1 / mu * sum([lamb[k] * (np.exp(-mu * t + mu * (k + 1)) - np.exp(-mu * t + mu * k)) for k in range(t)])

def m_t_PC_general_arrival(lamb, service_time_cdf, start, end, look_back):
    lamb = lamb[int(start)]
    def m_t_sub(t):
        def int_fun(u):
            # Service time cdf function is in seconds. Start/end uses shift as time unit
            return (1 - service_time_cdf((t-u)*30*60)) * lamb
        # Any u bigger than look_back will yield 0 for 1 - CDF
        x = np.linspace(t-look_back*24*2, t, 10000)
        y = [int_fun(u) for u in x]
        m_t = simps(y, x)
        return m_t

    return max([m_t_sub(t) for t in np.linspace(end, start, 10)])

def delay_prob(c, lamb, mu):
    # https://lucidmanager.org/data-science/call-centre-workforce-planning-erlang-c-in-r/
    # my original code was having overflow. Used this one from the link instead
    c = int(c)
    rate = lamb / mu
    erlang_b_inv = 1
    for i in range(c):
        erlang_b_inv = 1 + erlang_b_inv * (i+1) / rate
    erlang_b = 1 / erlang_b_inv
    return c * erlang_b / (c - rate * (1 - erlang_b))

def cost(c, lamb, mu, waiting_cost, salary):
    expected_mean_time = delay_prob(c, lamb, mu) / (c * mu - lamb)
    return waiting_cost * lamb * expected_mean_time + salary * c

def eps_k(lamb, mu, k):
    if k == 0:
        return 1
    else:
        return 1 + k*mu/lamb * eps_k(lamb, mu, k-1)

def aband_cost(c, lamb, mu, theta, salary, conversion, profit):
    # http://ie.technion.ac.il/serveng/References/MMNG_formulae.pdf
    # Mandelbaum, A., & Zeltyn, S. (2009). The M/M/n+ G queue: Summary of performance measures. Technical Note, Technion, Israel Institute of Technology.
    x = c*mu/theta
    y = lamb/theta
    def int_fun(t):
        return t**(x-1) * np.exp(-t)
    J = np.exp(lamb / theta) / theta * (theta / lamb) ** (c * mu / theta) * quad(int_fun, 0, y)[0]
    eps = eps_k(lamb, mu, c-1)
    aband_prob = (1 + (lamb - c*mu)*J) / (eps + lamb *J)
    labor_cost = c * salary
    opport_cost = lamb * aband_prob * conversion * profit
    return labor_cost + opport_cost


def generate_hetero_arrivals(lamb, T):
    t = 0
    hist = []
    while True:
        arr_rate = lamb(t)
        if (arr_rate == 0) & (t % 1 == 0):
            t += 1
            continue
        elif arr_rate == 0:
            t = np.ceil(t)
            continue
        else:
            t += np.random.exponential(1 / arr_rate)
        if t >= T:
            break
        hist.append(t)
    return hist

def sim_Mt_M_st(arrivals, mu, s, T):
    t = 0
    dep_times = np.array([]) # queue
    hist = pd.DataFrame(columns = ["n", "t", "s_t"])
    while True:
        if len(arrivals) > 0:
            time_to_arr = arrivals[0] - t
        else:
            time_to_arr = 1000
        c = s(t)
        if min(c, sum(~np.isnan(dep_times))) == 0:
            time_to_dep = 1000
        else:
            time_to_dep = np.nanmin(dep_times) - t
        time_to_next_day = np.ceil(t) - t
        day = np.floor(t)

        def hist_append(after_w_hours, t, is_last = False):
            return (hist.append(pd.DataFrame(
                {"day"    : [day], "after_w_hours": [after_w_hours], "n": [len(dep_times)], "t": [t], "s_t": [c],
                 "is_last": is_last}), ignore_index = True))

        if time_to_next_day < min(time_to_arr, time_to_dep):
            if len(dep_times) > 0:
                t_minor = t + time_to_dep
                while True:
                    # A work-day is 10 hours. The extra hours are from 6:30pm to 9pm.
                    if t_minor > (9 - 6.5) / 10:
                        hist = hist_append(True, t_minor, is_last = True)
                        dep_times = np.array([])
                        break
                    else:
                        t_minor = np.nanmin(dep_times)
                        dep_times = np.delete(dep_times, np.nanargmin(dep_times))
                        # TODO CORRECT available object!! Should be max of 0 and the value. Not an absolute value
                        available = int(np.abs(c - sum(~np.isnan(dep_times))))
                        size = int(min(sum(np.isnan(dep_times)), available))
                        if size > 0:
                            dep_times[np.nonzero(np.isnan(dep_times))[0][:size]] =\
                                t_minor + np.random.exponential(1 / mu, size = size)
                        if len(dep_times) == 0:
                            hist = hist_append(True, t_minor, is_last = True)
                            break
                        else:
                            hist = hist_append(True, t_minor)
            t = np.ceil(t)
        elif time_to_arr < time_to_dep:
            t = arrivals[0]
            arrivals.pop(0)
            if sum(~np.isnan(dep_times)) < c:
                dep_times = np.append(dep_times, t + np.random.exponential(1 / mu))
            else:
                dep_times = np.append(dep_times, np.NaN)
        else:
            t += time_to_dep
            dep_times = np.delete(dep_times, np.nanargmin(dep_times))
        # TODO CORRECT available object!! Should be max of 0 and the value. Not an absolute value
        available = int(np.abs(c - sum(~np.isnan(dep_times))))
        size = int(min(sum(np.isnan(dep_times)), available))
        if size > 0:
            dep_times[np.nonzero(np.isnan(dep_times))[0][:size]] = t + np.random.exponential(1 / mu, size = size)
        if t >= T:
            break
        hist = hist_append(False, t)
    hist["duration"] = hist.t - hist.t.shift(1)
    hist.loc[0, "duration"] = hist.loc[0, "t"]
    return hist

def sim_Mt_M_st_simple(arrivals, mu, s, T, cont_op, theta = 0,
                       n = 0, t = 0, my_seed = None):
    if my_seed:
        np.random.seed(my_seed)
    hist = pd.DataFrame()

    def hist_append(after_w_hours, t, is_abandon, is_last = False):
        return (hist.append(pd.DataFrame(
            {"day"    : [day], "after_w_hours": [after_w_hours], "n": [int(n)], "t": [t], "s_t": [c],
             "is_abandon": [is_abandon], "is_last": [is_last]}), ignore_index = True))

    np.seterr(divide = "ignore")
    day = np.floor(t)
    while True:
        c = s(t)
        if len(arrivals) > 0:
            time_to_arr = arrivals[0] - t
        else:
            time_to_arr = np.inf
        time_to_dep = np.random.exponential(1 / np.array(min(c, n) * mu))
        time_to_aband = np.random.exponential(1 / np.array(max(n - c, 0) * theta))
        time_to_next_day = np.floor(t+1) - t
        day = np.floor(t)

        is_abandon = False
        if (not cont_op) & (time_to_next_day < min(time_to_arr, time_to_dep)):
            # TODO: you gotta modify this part if you want to have continues operations and abandonment at the same time (now, no abandonment).
            if n > 0:
                t_minor = t + time_to_dep
                while True:
                    # A work-day is 10 hours. The extra hours are from 6:30pm to 9pm.
                    if t_minor >= np.ceil(t) + (9 - 6.5) / 10:
                        hist = hist_append(True, t_minor, is_last = True)
                        n = 0
                        break
                    else:
                        n -= 1
                        if n == 0:
                            hist = hist_append(True, t_minor, is_last = True)
                            break
                        else:
                            hist = hist_append(True, t_minor)
                            if c == 0:
                                t_minor = np.ceil(t) + (9 - 6.5) / 10
                            else:
                                t_minor += np.random.exponential(1 / (min(c, n) * mu))
            t = np.floor(t+1)
        elif time_to_arr < min(time_to_dep, time_to_aband):
            n += 1
            t = arrivals[0]
            arrivals.pop(0)
        else:
            n -= 1
            if time_to_dep < time_to_aband:
                t += time_to_dep
            else:
                t += time_to_aband
                is_abandon = True

        if t >= T:
            break

        hist = hist_append(False, t, is_abandon)
    # Duration: Length of the current event = Time to next event
    hist["duration"] = hist.t.shift(-1) - hist.t
    hist.duration.fillna(0, inplace=True)
    return hist, arrivals


def sim_Mt_G_st(arrivals, service_times, s, T,
                       n = 0, t = 0, my_seed = None):
    if my_seed:
        np.random.seed(my_seed)
    hist = pd.DataFrame()

    def hist_append(t, is_last = False):
        return (pd.concat([hist,
                           pd.DataFrame({"day": [day], "n": [int(n)], "t": [t], "s_t": [c], "is_last": [is_last]})],
                          ignore_index = True))

    np.seterr(divide = "ignore")
    day = np.floor(t)
    dep_times = np.array([])  # queue
    while True:
        c = s(t)
        if len(arrivals) > 0:
            time_to_arr = arrivals[0] - t
        else:
            time_to_arr = np.inf
        if min(c, sum(~np.isnan(dep_times))) == 0:
            time_to_dep = np.inf
        else:
            time_to_dep = np.nanmin(dep_times) - t
        day = np.floor(t)

        if time_to_arr < time_to_dep:
            t = arrivals[0]
            arrivals.pop(0)
            if sum(~np.isnan(dep_times)) < c:
                dep_times = np.append(dep_times, t + np.random.choice(service_times, 1))
            else:
                dep_times = np.append(dep_times, np.NaN)
        else:
            t += time_to_dep
            dep_times = np.delete(dep_times, np.nanargmin(dep_times))
        n = len(dep_times)
        available = int(max(c - sum(~np.isnan(dep_times)), 0))
        size = int(min(sum(np.isnan(dep_times)), available))
        if size > 0:
            dep_times[np.nonzero(np.isnan(dep_times))[0][:size]] = t + np.random.choice(service_times, size = size)
        if t >= T:
            break
        hist = hist_append(t)

    # Duration: Length of the current event = Time to next event
    hist["duration"] = hist.t.shift(-1) - hist.t
    hist.duration.fillna(0, inplace=True)
    return hist, arrivals