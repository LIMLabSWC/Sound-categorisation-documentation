import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from functools import wraps
import logging

# Set the path to the modules
cwd = os.getcwd()
models_dir = os.path.join(cwd, 'Models')
sys.path.append(models_dir)

# Import ONLY Hybrid and Optuna Fitter modules
from Hybrid import Hybrid_model
from Fitter import post_correct_update_matrix
from Gradient_free_Opt import run_cv_optimization, run_full_optimization

def profile_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.2f} seconds.")
        return result
    return wrapper

def save_results(cv_results, save_path):
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(cv_results, f)
        print(f"  ✔ Results saved to {save_path}")
    except Exception as e:
        print(f"Error saving results to {save_path}: {e}")
        raise

@profile_time
def run_pid_cv(species, p_id, data_path, search_space, seed, cv_folds=5, optuna_trials=50, sampler_type='TPE', save_path=None):
    """
    Run Optuna optimization for a specific participant ID and save to a SINGLE seed file.
    """
    # Cluster-Safe Logging (Unique per PID and Seed)
    log_name = f"errors_pid_{p_id}_seed_{seed}.log"
    logging.basicConfig(
        filename=log_name,   
        filemode="w", 
        level=logging.WARNING,               
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True
    )

    # Load data
    data = pd.read_csv(data_path)
    df = pd.DataFrame(data)
    pid_df = df[df['Participant_ID'] == p_id].reset_index(drop=True)
    pid_df = pid_df[pid_df['Distribution'].isin(['Asym_left', 'Uniform'])].reset_index(drop=True)
    
    # Calculate block starts
    pid_df['is_not_start_of_block'] = pid_df['block'].eq(pid_df['block'].shift())

    # Prepare results directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"  ✱ Running Optuna for PID {p_id}, Seed {seed}, Sampler: {sampler_type}...")

    # Set up the dictionary to hold JUST this seed's results
    seed_results = {
        'metadata': {
            'Participant_ID': p_id, 'Model': 'Hybrid', 
            'cv_folds': cv_folds, 'trials': optuna_trials, 
            'seed': seed, 'sampler': sampler_type
        }
    }

    try:
        if cv_folds == 0:
            print("  ✱ Running FULL DATA FIT...")
            best_error, best_params, lapse_rates, matrices = run_full_optimization(
                df=pid_df, 
                model=Hybrid_model, 
                func=post_correct_update_matrix, 
                seed=seed, 
                search_space=search_space, 
                n_trials=optuna_trials, 
                mode_pre='simulated',         
                fit_with='conditional', 
                sampler_type=sampler_type 
            )
            
            seed_results.update({
                'best_error': best_error,
                'best_params': best_params,
                'lapse_rates': lapse_rates,
                'final_matrices': matrices,
            })
            seed_results['metadata']['type'] = 'full_fit'
            
        else:
            print(f"  ✱ Running {cv_folds}-FOLD CV...")
            mean_cv_error, train_errs, test_errs, best_params, lapse_rates, matrices = run_cv_optimization(
                df=pid_df, 
                model=Hybrid_model, 
                func=post_correct_update_matrix, 
                seed=seed, 
                search_space=search_space, 
                k=cv_folds, 
                n_trials=optuna_trials, 
                mode_pre='simulated',         
                fit_with='conditional', 
                sampler_type=sampler_type 
            )
            
            seed_results.update({
                'mean_cv_error': mean_cv_error,
                'train_errors': train_errs,
                'test_errors': test_errs,
                'best_params_per_fold': best_params,
                'lapse_rates_per_fold': lapse_rates,
                'matrices_per_fold': matrices,
            })
            seed_results['metadata']['type'] = 'cv'

    except Exception as e:
        logging.exception(f"Error running optimization for seed {seed}: {e}")
        print(f"  ✘ Error for seed {seed}. See {log_name} for traceback.")
        seed_results['error'] = str(e)

    # Save directly to the unique file
    save_results(seed_results, save_path)
    print(f"  ✔ Processing completed for Participant ID {p_id}, Seed {seed}.")