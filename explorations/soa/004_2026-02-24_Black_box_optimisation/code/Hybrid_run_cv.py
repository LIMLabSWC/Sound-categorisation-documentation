import argparse
import sys
import os

sys.path.append('/nfs/nhome/live/apourdehghan')
import Hybrid_hpc_utils as hpc_ut

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--p_id', type=str, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--trials', type=int, default=200, help="Number of Optuna guesses")
# ---> NEW: Allow choosing the sampler directly from the command line
parser.add_argument('--sampler', type=str, default='TPE', choices=['TPE', 'CMA-ES'], help="Optuna sampler to use")
args = parser.parse_args()

# ONLY Hybrid Search Space
optuna_search_space = {
    'sigma_noise':    {'low': 0.05, 'high': 0.45},
    'A_repulsion':    {'low': 0.0,  'high': 0.5},
    'gamma':          {'low': 0.01, 'high': 1.0},
    'sigma_update':   {'low': 0.1,  'high': 0.7},
    'eta_learning':   {'low': 0.1,  'high': 1.0},
    'sigma_boundary': {'low': 0.01, 'high': 0.5},
    'alpha':          {'low': 0.0,  'high': 1.0}
}

# Create results folder if it doesn't exist
subject_folder = f'/nfs/nhome/live/apourdehghan/results/{args.p_id}'
os.makedirs(subject_folder, exist_ok=True)

# Build a unique save path automatically using the PID and Seed
save_name = f"results_pid_{args.p_id}_seed_{args.seed}_{args.sampler}.pkl"
save_path = os.path.join(subject_folder, save_name)

# Run the Optuna HPC Pipeline
hpc_ut.run_pid_cv(
    species='mouse',
    p_id=args.p_id,
    data_path='/nfs/nhome/live/apourdehghan/preprocessed_Data_mouse_March2025.csv',
    search_space=optuna_search_space,
    seed=args.seed,           
    cv_folds=2,
    optuna_trials=args.trials,
    sampler_type=args.sampler,
    save_path=save_path
)