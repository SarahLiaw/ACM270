# configurations.py

MY_LORENZ_PARAMS = {
    'simulation_duration': 50, 
    'time_step_size': 0.01,
    'initial_conditions': [1, 1, 1],
    'lorenz_sigma': 10,
    'lorenz_rho': 28,
    'lorenz_beta': 8/3,
}

EXPERIMENT_PARAMS = {
    'system': 'my_lorenz',
    'system_config': MY_LORENZ_PARAMS,
    'noise_levels': [0.0, 0.01, 0.05, 0.1, 0.2],
    'fractions_observed': [1.0, 0.8, 0.5, 0.3],
    'svd_rank_settings': {
        'my_lorenz': 3,
        'my_lorenz_partial_obs_heuristic': 'full_rank_of_partial',
        'default_rank': 0
    },
    'num_trials': 100,
    'trial_size': 0.7,
    'num_snapshots': 1000
}
