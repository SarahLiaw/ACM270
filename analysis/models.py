from pydmd import BOPDMD
import numpy as np

def perform_opt_dmd(snapshots, times, svd_rank):
    optdmd_model = BOPDMD(svd_rank=svd_rank, num_trials=0)
    optdmd_model.fit(snapshots, times)
    return optdmd_model

def perform_bop_dmd(snapshots, times, svd_rank, num_trials=100, trial_size=0.7):
    bopdmd_model = BOPDMD(svd_rank=svd_rank, num_trials=num_trials, trial_size=trial_size)
    bopdmd_model.fit(snapshots, times)
    return bopdmd_model