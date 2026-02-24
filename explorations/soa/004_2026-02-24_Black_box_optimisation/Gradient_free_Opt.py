import numpy as np
import optuna
from scipy.stats import uniform
import warnings
import logging
from BE import Noise_generator, Delta_repulsion, Delta_learning
from Fitter import total_psychometric, matrix_error, merge_smallest_adjacent, select_and_concatenate

# ───────────────────────────────────────────────────────────────────────────────
# Configure logging for Fitter.py:
logging.basicConfig(
    filename="fitter_errors.log",   
    filemode="w",                        
    level=logging.WARNING,               
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s"
)
logging.warning("🔍 Logging initialized in Gradient_free_Opt.py")
# ───────────────────────────────────────────────────────────────────────────────


#warnings.filterwarnings("ignore")


def train_objective(trial, train_df, model, func, seed, search_space, mode_pre, fit_with, lambda_A, lambda_B):
    """
    Optuna uses this function to minimize the matrix error on the TRAINING data.
    """
    # 1. Dynamically suggest parameters
    params = {}
    for param_name, bounds in search_space.items():
        params[param_name] = trial.suggest_float(
            param_name, bounds['low'], bounds['high'], log=bounds.get('log', False)
        )

    # 2. Extract Training Data
    s_train = train_df['stim_relative'].to_numpy()
    chooseB_train = train_df['choice'].to_numpy()
    rewards_train = train_df['correct'].to_numpy()
    No_response_train = train_df['No_response'].to_numpy()
    Not_Blockstart_train = train_df['is_not_start_of_block'].to_numpy()
    categories_train = np.where(s_train > 0, 1, 0)
    
    # Set burn-in trials based on mode, or crash if the mode is unrecognized
    if mode_pre == 'real':
        n_burn = 200
    elif mode_pre == 'simulated':
        n_burn = 0
    else:
        raise ValueError(f"Invalid mode_pre option: '{mode_pre}'. Must be 'real' or 'simulated'.")


    # 3. Target Training Matrix (Empirical)
    train_data, train_psychs = func(
        s_train[n_burn:], chooseB_train[n_burn:], rewards_train[n_burn:], 
        No_response_train[n_burn:], Not_Blockstart_train[n_burn:]
    )


    if fit_with == 'conditional':
        target_train_matrix = train_psychs[::-1]
    elif fit_with == 'update':
        target_train_matrix = train_data[::-1]
    else:
        raise ValueError(f"Invalid fit_with: '{fit_with}'. Use 'conditional' or 'update'.")

    # 4. Prepare Stimulus Space
    sigma_noise = params['sigma_noise']
    A_repulsion = params['A_repulsion']
    
    max_range = 1 + 6 * sigma_noise + 2 * A_repulsion * (1 + 6 * sigma_noise)
    min_range = -1 - 6 * sigma_noise - 2 * A_repulsion * (1 + 6 * sigma_noise)
    num_points = round((max_range - min_range) * 1000)
    x = np.linspace(min_range, max_range, num_points)
    
    s_train_tilde = s_train + Noise_generator(len(s_train), seed, sigma_noise)
    s_train_hat = Delta_repulsion(A_repulsion, s_train_tilde)
    y_BE_initial = uniform.pdf(x, loc=min_range, scale=max_range - min_range)

# 5. Run Model (Matches your exact signature)
    try:
        Final_update_matrix, Final_conditional_matrix, SC_update_matrix, \
        SC_conditional_matrix, BE_update_matrix, BE_conditional_matrix = model(
            func=func, 
            x=x, 
            s=s_train, 
            s_hat=s_train_hat, 
            categories=categories_train, 
            y_BE=y_BE_initial.copy(), 
            Delta_learning=Delta_learning, 
            lambda_A=lambda_A, 
            lambda_B=lambda_B, 
            no_response=No_response_train, 
            Not_Blockstart=Not_Blockstart_train, 
            seed=seed, 
            mode=mode_pre, 
            **params  # <--- Unpacks sigma_noise, A_repulsion, gamma, sigma_update, eta_learning, sigma_boundary, alpha
        )
    except Exception as e:
        logging.error(f"Training crashed! Params: {params} | Error: {str(e)}")
        raise optuna.exceptions.TrialPruned()

    # 6. Extract the correct matrix for scoring (No need to run func() again!)
    if fit_with == 'conditional':
        sim_matrix = Final_conditional_matrix
    elif fit_with == 'update':
        sim_matrix = Final_update_matrix
    else:
        raise ValueError(f"Invalid fit_with: '{fit_with}'.")
    
    # 7. Return the error for Optuna to minimize
    return matrix_error(sim_matrix, target_train_matrix)



def run_cv_optimization(df, model, func, seed, search_space, k=5, n_trials=50, mode_pre='simulated', fit_with='conditional', sampler_type='TPE'):
    """
    Splits the data into folds, runs Optuna to find best params on the training fold, 
    and tests those params on the test fold.
    Returns: mean_test_error, train_errors, test_errors, best_params, lapse_rates, fold_matrices_list
    """
    # Fail-fast guards
    if fit_with not in ['conditional', 'update']:
        raise ValueError(f"Invalid fit_with option: '{fit_with}'. Must be 'conditional' or 'update'.")
    
    if mode_pre == 'real':
        n_burn = 200
    elif mode_pre == 'simulated':
        n_burn = 0
    else:
        raise ValueError(f"Invalid mode_pre option: '{mode_pre}'. Must be 'real' or 'simulated'.")

    block_sizes = df.groupby('block')['Trial'].count().reset_index(name='count')
    sizes = block_sizes['count'].to_numpy()
    label = block_sizes['block'].to_numpy()

    if len(sizes) == 1:
        raise ValueError("Only one block found. Cannot perform k-fold.")
    elif len(sizes) < k:
        k = len(sizes)

    blocks_in_folds = merge_smallest_adjacent(sizes, label, k)
    
    # Lists to store our outputs
    fold_train_errors = []
    fold_test_errors = []
    fold_best_params = []   
    fold_lapse_rates = []   
    fold_matrices_list = [] # <--- Will now store Final, SC, and BE matrices

    for fold_idx in range(k):
        print(f"\n--- Starting Fold {fold_idx + 1}/{k} ---")
        test_df, train_df = select_and_concatenate(df, blocks_in_folds, fold_idx)

        # ---------------------------------------------------------------------
        # PHASE 1: FIT LAPSE RATES ON TRAINING SET
        # ---------------------------------------------------------------------
        s_train = train_df['stim_relative'].to_numpy()
        chooseB_train = train_df['choice'].to_numpy()
        No_response_train = train_df['No_response'].to_numpy()
        Not_Blockstart_train = train_df['is_not_start_of_block'].to_numpy()
        categories_train = np.where(s_train > 0, 1, 0)
        
        _, fit_params = total_psychometric(s_train[n_burn:], chooseB_train[n_burn:], No_response_train[n_burn:])
        lambda_A = np.clip(fit_params[2], 0.0, 0.5) 
        lambda_B = np.clip(fit_params[3], 0.0, 0.5) 
        
        fold_lapse_rates.append({'lambda_A': lambda_A, 'lambda_B': lambda_B})

        # Calculate empirical Training Target Matrices
        train_data, train_psychs = func(
            s_train[n_burn:], chooseB_train[n_burn:], train_df['correct'].to_numpy()[n_burn:], 
            No_response_train[n_burn:], Not_Blockstart_train[n_burn:]
        )
        
        # ---------------------------------------------------------------------
        # PHASE 2: OPTIMIZE HYPERPARAMETERS ON TRAINING SET
        # ---------------------------------------------------------------------
        obj = lambda trial: train_objective(
            trial, train_df, model, func, seed, search_space, 
            mode_pre, fit_with, lambda_A, lambda_B
        )
        
        if sampler_type.upper() == 'CMA-ES':
            sampler = optuna.samplers.CmaEsSampler(seed=seed)
        elif sampler_type.upper() == 'TPE':
            sampler = optuna.samplers.TPESampler(seed=seed)
        else:
            raise ValueError("Invalid sampler_type. Choose 'TPE' or 'CMA-ES'.")
            
        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(obj, n_trials=n_trials)
        
        best_params = study.best_params
        best_train_error = study.best_value  
        
        fold_best_params.append(best_params) 
        fold_train_errors.append(best_train_error) 
        
        print(f"Empirical Lapse Rates: lambda_A={lambda_A:.4f}, lambda_B={lambda_B:.4f}")
        print(f"Best Params for Fold {fold_idx + 1}: {best_params}")
        print(f"Training Error for Fold {fold_idx + 1}: {best_train_error:.4f}")

        # ---------------------------------------------------------------------
        # PHASE 2b: GENERATE BEST MATRICES FOR TRAINING SET
        # ---------------------------------------------------------------------
        sigma_noise_best = best_params['sigma_noise']
        A_repulsion_best = best_params['A_repulsion']
        
        max_range_best = 1 + 6 * sigma_noise_best + 2 * A_repulsion_best * (1 + 6 * sigma_noise_best)
        min_range_best = -1 - 6 * sigma_noise_best - 2 * A_repulsion_best * (1 + 6 * sigma_noise_best)
        num_points_best = round((max_range_best - min_range_best) * 1000)
        x_best = np.linspace(min_range_best, max_range_best, num_points_best)
        y_BE_initial_best = uniform.pdf(x_best, loc=min_range_best, scale=max_range_best - min_range_best)

        s_train_tilde_best = s_train + Noise_generator(len(s_train), seed, sigma_noise_best)
        s_train_hat_best = Delta_repulsion(A_repulsion_best, s_train_tilde_best)
        
        # ---> CHANGED: Unpack ALL matrices from the training run
        Final_up_train, Final_cond_train, SC_up_train, SC_cond_train, BE_up_train, BE_cond_train = model(
            func=func, x=x_best, s=s_train, s_hat=s_train_hat_best, categories=categories_train, 
            y_BE=y_BE_initial_best.copy(), Delta_learning=Delta_learning, lambda_A=lambda_A, 
            lambda_B=lambda_B, no_response=No_response_train, Not_Blockstart=Not_Blockstart_train, 
            seed=seed, mode=mode_pre, **best_params
        )

        # ---------------------------------------------------------------------
        # PHASE 3: EVALUATE BEST PARAMS ON TEST SET
        # ---------------------------------------------------------------------
        s_test = test_df['stim_relative'].to_numpy()
        chooseB_test = test_df['choice'].to_numpy()
        rewards_test = test_df['correct'].to_numpy()
        No_response_test = test_df['No_response'].to_numpy()
        Not_Blockstart_test = test_df['is_not_start_of_block'].to_numpy()
        categories_test = np.where(s_test > 0, 1, 0)

        # Target Test Matrix (Sliced by n_burn)
        test_data, test_psychs = func(
            s_test[n_burn:], chooseB_test[n_burn:], rewards_test[n_burn:], 
            No_response_test[n_burn:], Not_Blockstart_test[n_burn:]
        )
        
        if fit_with == 'conditional':
            target_test_matrix = test_psychs[::-1]
        elif fit_with == 'update':
            target_test_matrix = test_data[::-1]
        else:
            raise ValueError(f"Invalid fit_with: '{fit_with}'.")

        s_test_tilde = s_test + Noise_generator(len(s_test), seed, sigma_noise_best)
        s_test_hat = Delta_repulsion(A_repulsion_best, s_test_tilde)

        # ---> CHANGED: Unpack ALL matrices from the test run
        Final_up_test, Final_cond_test, SC_up_test, SC_cond_test, BE_up_test, BE_cond_test = model(
            func=func, x=x_best, s=s_test, s_hat=s_test_hat, categories=categories_test, 
            y_BE=y_BE_initial_best.copy(), Delta_learning=Delta_learning, lambda_A=lambda_A, 
            lambda_B=lambda_B, no_response=No_response_test, Not_Blockstart=Not_Blockstart_test, 
            seed=seed, mode=mode_pre, **best_params  
        )

        # Score only the Final Hybrid Matrix
        if fit_with == 'conditional':
            sim_test_matrix = Final_cond_test
        elif fit_with == 'update':
            sim_test_matrix = Final_up_test
        else:
            raise ValueError(f"Invalid fit_with: '{fit_with}'.")
        
        test_error = matrix_error(sim_test_matrix, target_test_matrix)
        fold_test_errors.append(test_error)
        print(f"Test Error for Fold {fold_idx + 1}: {test_error:.4f}\n")

        # ---------------------------------------------------------------------
        # PHASE 4: SAVE ALL MATRICES TO DICTIONARY
        # ---------------------------------------------------------------------
        fold_matrices = {
            # Empirical Human Data
            'train_target_update': train_data[::-1],
            'train_target_conditional': train_psychs[::-1],
            'test_target_update': test_data[::-1],
            'test_target_conditional': test_psychs[::-1],
            
            # Simulated Model Data (TRAIN)
            'train_sim_Final_update': Final_up_train,
            'train_sim_Final_conditional': Final_cond_train,
            'train_sim_SC_update': SC_up_train,
            'train_sim_SC_conditional': SC_cond_train,
            'train_sim_BE_update': BE_up_train,
            'train_sim_BE_conditional': BE_cond_train,
            
            # Simulated Model Data (TEST)
            'test_sim_Final_update': Final_up_test,
            'test_sim_Final_conditional': Final_cond_test,
            'test_sim_SC_update': SC_up_test,
            'test_sim_SC_conditional': SC_cond_test,
            'test_sim_BE_update': BE_up_test,
            'test_sim_BE_conditional': BE_cond_test
        }
        fold_matrices_list.append(fold_matrices)

    mean_cv_error = np.mean(fold_test_errors)
    
    return mean_cv_error, fold_train_errors, fold_test_errors, fold_best_params, fold_lapse_rates, fold_matrices_list


def run_full_optimization(df, model, func, seed, search_space, n_trials=50, mode_pre='simulated', fit_with='conditional', sampler_type='TPE'):
    """
    Fits the model to the entire dataset using Optuna.
    Returns: best_error, best_params, lapse_rates, final_matrices
    """
    # Fail-fast guards
    if fit_with not in ['conditional', 'update']:
        raise ValueError(f"Invalid fit_with option: '{fit_with}'. Must be 'conditional' or 'update'.")
    
    if mode_pre == 'real':
        n_burn = 200
    elif mode_pre == 'simulated':
        n_burn = 0
    else:
        raise ValueError(f"Invalid mode_pre option: '{mode_pre}'. Must be 'real' or 'simulated'.")

    print("\n--- Starting Full Dataset Optimization ---")

    # ---------------------------------------------------------------------
    # PHASE 1: FIT LAPSE RATES ON FULL DATA
    # ---------------------------------------------------------------------
    s_full = df['stim_relative'].to_numpy()
    chooseB_full = df['choice'].to_numpy()
    No_response_full = df['No_response'].to_numpy()
    rewards_full = df['correct'].to_numpy()
    Not_Blockstart_full = df['is_not_start_of_block'].to_numpy()
    categories_full = np.where(s_full > 0, 1, 0)
    
    _, fit_params = total_psychometric(s_full[n_burn:], chooseB_full[n_burn:], No_response_full[n_burn:])
    lambda_A = np.clip(fit_params[2], 0.0, 0.5) 
    lambda_B = np.clip(fit_params[3], 0.0, 0.5) 
    
    lapse_rates = {'lambda_A': lambda_A, 'lambda_B': lambda_B}

    # ---------------------------------------------------------------------
    # PHASE 2: OPTIMIZE HYPERPARAMETERS ON FULL DATA
    # ---------------------------------------------------------------------
    # We reuse the exact same train_objective, but pass the full df
    obj = lambda trial: train_objective(
        trial, df, model, func, seed, search_space, 
        mode_pre, fit_with, lambda_A, lambda_B
    )
    
    if sampler_type.upper() == 'CMA-ES':
        sampler = optuna.samplers.CmaEsSampler(seed=seed)
        print("Using CMA-ES Sampler...")
    elif sampler_type.upper() == 'TPE':
        sampler = optuna.samplers.TPESampler(seed=seed)
        print("Using TPE Sampler...")
    else:
        raise ValueError("Invalid sampler_type. Choose 'TPE' or 'CMA-ES'.")
        
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(obj, n_trials=n_trials)
    
    best_params = study.best_params
    best_error = study.best_value  
    
    print(f"\nOptimization Complete!")
    print(f"Empirical Lapse Rates: lambda_A={lambda_A:.4f}, lambda_B={lambda_B:.4f}")
    print(f"Best Params: {best_params}")
    print(f"Final Minimum Error: {best_error:.4f}")

    # ---------------------------------------------------------------------
    # PHASE 3: GENERATE BEST MATRICES FOR FULL DATA
    # ---------------------------------------------------------------------
    # Calculate empirical Target Matrices for the full dataset
    full_data, full_psychs = func(
        s_full[n_burn:], chooseB_full[n_burn:], rewards_full[n_burn:], 
        No_response_full[n_burn:], Not_Blockstart_full[n_burn:]
    )

    sigma_noise_best = best_params['sigma_noise']
    A_repulsion_best = best_params['A_repulsion']
    
    max_range_best = 1 + 6 * sigma_noise_best + 2 * A_repulsion_best * (1 + 6 * sigma_noise_best)
    min_range_best = -1 - 6 * sigma_noise_best - 2 * A_repulsion_best * (1 + 6 * sigma_noise_best)
    num_points_best = round((max_range_best - min_range_best) * 1000)
    x_best = np.linspace(min_range_best, max_range_best, num_points_best)
    y_BE_initial_best = uniform.pdf(x_best, loc=min_range_best, scale=max_range_best - min_range_best)

    s_full_tilde_best = s_full + Noise_generator(len(s_full), seed, sigma_noise_best)
    s_full_hat_best = Delta_repulsion(A_repulsion_best, s_full_tilde_best)
    
    Final_up, Final_cond, SC_up, SC_cond, BE_up, BE_cond = model(
        func=func, x=x_best, s=s_full, s_hat=s_full_hat_best, categories=categories_full, 
        y_BE=y_BE_initial_best.copy(), Delta_learning=Delta_learning, lambda_A=lambda_A, 
        lambda_B=lambda_B, no_response=No_response_full, Not_Blockstart=Not_Blockstart_full, 
        seed=seed, mode=mode_pre, **best_params
    )

    # ---------------------------------------------------------------------
    # PHASE 4: SAVE ALL MATRICES TO DICTIONARY
    # ---------------------------------------------------------------------
    final_matrices = {
        # Empirical Human Data
        'target_update': full_data[::-1],
        'target_conditional': full_psychs[::-1],
        
        # Simulated Model Data
        'sim_Final_update': Final_up,
        'sim_Final_conditional': Final_cond,
        'sim_SC_update': SC_up,
        'sim_SC_conditional': SC_cond,
        'sim_BE_update': BE_up,
        'sim_BE_conditional': BE_cond
    }

    return best_error, best_params, lapse_rates, final_matrices