import numpy as np

def add_noise(data, noise_percent):
    if noise_percent == 0:
        return data

    noisy_data = np.copy(data)
    for i in range(data.shape[0]):
        signal_std = np.std(data[i, :])
        if signal_std == 0:
            noise_std = 0
        else:
            noise_std = noise_percent * signal_std
        
        noise = np.random.normal(0, noise_std, data.shape[1])
        noisy_data[i, :] = data[i, :] + noise
    return noisy_data


def get_partial_observations(data, frac_obs, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if frac_obs < 0 or frac_obs > 1:
        raise ValueError("frac_obs must be between 0 and 1.")
    if frac_obs == 1.0:
        return data, np.arange(data.shape[0])

    n_vars = data.shape[0]
    n_obs_vars = int(np.ceil(n_vars * frac_obs))
    if n_obs_vars == 0 and n_vars > 0 and frac_obs > 0:
        n_obs_vars = 1
    if n_obs_vars > n_vars:
        n_obs_vars = n_vars

    if n_vars == 0: 
        return data, np.array([], dtype=int)
        
    if n_obs_vars > n_vars :
        n_obs_vars = n_vars 
        
    obs_indices = np.random.choice(n_vars, n_obs_vars, replace=False)
    obs_indices.sort() 
    
    partial_data = data[obs_indices, :]
    return partial_data, obs_indices