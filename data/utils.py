import numpy as np

def add_noise(data, noisy_percent):
    if noisy_percent == 0:
        return data

    noisy_data = np.copy(data)
    for i in range(data.shape[0]):
        signal_std = np.std(data[i, :])
        noise_std = noisy_percent * signal_std
        noise = np.random.normal(0, noise_std, data.shape[1])
        noisy_data[i, :] = data[i, :] + noise
    return noisy_data


def get_partial_observations(data, fraction_observed, seed=None):
    return "TODO"