import numpy as np
import optuna
import logging
from scipy.stats import uniform
import pandas as pd

from BE import Noise_generator, Delta_repulsion, Delta_learning
from Fitter import (
    total_psychometric,
    matrix_error,
    create_random_folds,
    select_and_concatenate,
    post_correct_update_matrix_binned,
    post_correct_binned_counts,
    matrices_from_binned_counts,
)

logging.basicConfig(
    filename="fitter_errors.log",
    filemode="a",
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s"
)

MAX_SEED = 2**32 - 1

def _safe_seed(seed):
    return int(seed) % MAX_SEED


def _get_n_burn(mode_pre):
    if mode_pre == "real":
        return 200
    if mode_pre == "simulated":
        return 0
    raise ValueError("mode_pre must be 'real' or 'simulated'.")


def _complete_params(params, fixed_alpha=None):
    completed = {
        "sigma_noise": 0.1,
        "A_repulsion": 0.0,
        "gamma": 0.5,
        "sigma_update": 0.2,
        "eta_learning": 0.1,
        "sigma_boundary": 1.0,
        "alpha": 0.5,
    }
    completed.update(params)

    if fixed_alpha is not None:
        completed["alpha"] = fixed_alpha

    return completed


def _make_x_and_y(params):
    sigma_noise = params["sigma_noise"]
    A_repulsion = params["A_repulsion"]

    max_range = 1 + 6 * sigma_noise + 2 * A_repulsion * (1 + 6 * sigma_noise)
    min_range = -1 - 6 * sigma_noise - 2 * A_repulsion * (1 + 6 * sigma_noise)

    num_points = max(10, round((max_range - min_range) * 1000))
    x = np.linspace(min_range, max_range, num_points)
    y_BE_initial = uniform.pdf(x, loc=min_range, scale=max_range - min_range)

    return x, y_BE_initial


def _extract_arrays(df):
    s = df["stim_relative"].to_numpy()
    chooseB = df["choice"].to_numpy()
    rewards = df["correct"].to_numpy()
    no_response = df["No_response"].to_numpy()
    not_blockstart = df["is_not_start_of_block"].to_numpy()
    categories = np.where(s > 0, 1, 0)

    return s, chooseB, rewards, no_response, not_blockstart, categories


def _empirical_target_matrix(df, fit_with, mode_pre, n_bins=8):
    n_burn = _get_n_burn(mode_pre)

    s, chooseB, rewards, no_response, not_blockstart, _ = _extract_arrays(df)

    update, conditional = post_correct_update_matrix_binned(
        s=s[n_burn:],
        chooseB=chooseB[n_burn:],
        reward=rewards[n_burn:],
        No_response=no_response[n_burn:],
        Not_Blockstart=not_blockstart[n_burn:],
        n_bins=n_bins,
    )

    update = update[::-1]
    conditional = conditional[::-1]

    if fit_with == "conditional":
        return conditional, update, conditional
    if fit_with == "update":
        return update, update, conditional

    raise ValueError("fit_with must be 'conditional' or 'update'.")


def _pooled_model_matrices(
    df,
    model_outputs,
    params,
    lambda_A,
    lambda_B,
    base_seed,
    model_seed_count,
    mode_pre,
    n_bins=8,
):
    params = _complete_params(params)

    s, _, _, no_response, not_blockstart, categories = _extract_arrays(df)

    x, y_BE_initial = _make_x_and_y(params)

    N_final = B_final = Nt_final = Bt_final = None
    N_sc = B_sc = Nt_sc = Bt_sc = None
    N_be = B_be = Nt_be = Bt_be = None

    for m in range(model_seed_count):
        model_seed = _safe_seed(base_seed * 10000 + 1000 + m)

        s_tilde = s + Noise_generator(len(s), model_seed, params["sigma_noise"])
        s_hat = Delta_repulsion(params["A_repulsion"], s_tilde)

        out = model_outputs(
            x=x,
            s=s,
            s_hat=s_hat,
            categories=categories,
            sigma_noise=params["sigma_noise"],
            A_repulsion=params["A_repulsion"],
            gamma=params["gamma"],
            sigma_update=params["sigma_update"],
            y_BE=y_BE_initial.copy(),
            Delta_learning=Delta_learning,
            eta_learning=params["eta_learning"],
            sigma_boundary=params["sigma_boundary"],
            alpha=params["alpha"],
            lambda_A=lambda_A,
            lambda_B=lambda_B,
            no_response=no_response,
            Not_Blockstart=not_blockstart,
            seed=model_seed,
            mode=mode_pre,
        )

        counts_final = post_correct_binned_counts(
            s=out["s"],
            chooseB=out["Final_choice"],
            reward=out["Final_reward"],
            No_response=out["no_response"],
            Not_Blockstart=out["Not_Blockstart"],
            n_bins=n_bins,
        )

        counts_sc = post_correct_binned_counts(
            s=out["s"],
            chooseB=out["SC_choice"],
            reward=out["SC_reward"],
            No_response=out["no_response"],
            Not_Blockstart=out["Not_Blockstart"],
            n_bins=n_bins,
        )

        counts_be = post_correct_binned_counts(
            s=out["s"],
            chooseB=out["BE_choice"],
            reward=out["BE_reward"],
            No_response=out["no_response"],
            Not_Blockstart=out["Not_Blockstart"],
            n_bins=n_bins,
        )

        if N_final is None:
            N_final, B_final, Nt_final, Bt_final = [c.copy() for c in counts_final]
            N_sc, B_sc, Nt_sc, Bt_sc = [c.copy() for c in counts_sc]
            N_be, B_be, Nt_be, Bt_be = [c.copy() for c in counts_be]
        else:
            for acc, val in zip(
                [N_final, B_final, Nt_final, Bt_final],
                counts_final
            ):
                acc += val

            for acc, val in zip(
                [N_sc, B_sc, Nt_sc, Bt_sc],
                counts_sc
            ):
                acc += val

            for acc, val in zip(
                [N_be, B_be, Nt_be, Bt_be],
                counts_be
            ):
                acc += val

    Final_update, Final_conditional, Final_total = matrices_from_binned_counts(
        N_final, B_final, Nt_final, Bt_final
    )
    SC_update, SC_conditional, SC_total = matrices_from_binned_counts(
        N_sc, B_sc, Nt_sc, Bt_sc
    )
    BE_update, BE_conditional, BE_total = matrices_from_binned_counts(
        N_be, B_be, Nt_be, Bt_be
    )

    matrices = {
        "Final_update": Final_update[::-1],
        "Final_conditional": Final_conditional[::-1],
        "SC_update": SC_update[::-1],
        "SC_conditional": SC_conditional[::-1],
        "BE_update": BE_update[::-1],
        "BE_conditional": BE_conditional[::-1],
        "Final_total": Final_total[::-1],
        "SC_total": SC_total[::-1],
        "BE_total": BE_total[::-1],
    }

    return matrices


def _agent_choices_table(
    df,
    model_outputs,
    params,
    lambda_A,
    lambda_B,
    base_seed,
    model_seed_count,
    mode_pre,
):
    params = _complete_params(params)

    s, _, _, no_response, not_blockstart, categories = _extract_arrays(df)
    x, y_BE_initial = _make_x_and_y(params)

    id_col = "Participant_ID" if "Participant_ID" in df.columns else "Participant_Private_ID"

    table = pd.DataFrame({
        id_col: df[id_col].to_numpy(),
        "Trial": df["Trial"].to_numpy(),
        "block": df["block"].to_numpy(),
        "relative_stim": df["stim_relative"].to_numpy(),
    })

    for m in range(model_seed_count):
        model_seed = _safe_seed(base_seed * 10000 + 1000 + m)

        s_tilde = s + Noise_generator(len(s), model_seed, params["sigma_noise"])
        s_hat = Delta_repulsion(params["A_repulsion"], s_tilde)

        out = model_outputs(
            x=x,
            s=s,
            s_hat=s_hat,
            categories=categories,
            sigma_noise=params["sigma_noise"],
            A_repulsion=params["A_repulsion"],
            gamma=params["gamma"],
            sigma_update=params["sigma_update"],
            y_BE=y_BE_initial.copy(),
            Delta_learning=Delta_learning,
            eta_learning=params["eta_learning"],
            sigma_boundary=params["sigma_boundary"],
            alpha=params["alpha"],
            lambda_A=lambda_A,
            lambda_B=lambda_B,
            no_response=no_response,
            Not_Blockstart=not_blockstart,
            seed=model_seed,
            mode=mode_pre,
        )

        table[f"seed_{m + 1}"] = out["Final_choice"]

    return table


def _suggest_params(trial, search_space):
    params = {}
    for name, bounds in search_space.items():
        params[name] = trial.suggest_float(
            name,
            bounds["low"],
            bounds["high"],
            log=bounds.get("log", False),
        )
    return params


def train_objective_pooled(
    trial,
    train_df,
    model_outputs,
    search_space,
    fit_with,
    mode_pre,
    lambda_A,
    lambda_B,
    base_seed,
    model_seed_count,
    fixed_alpha=None,
    n_bins=8,
):
    params = _suggest_params(trial, search_space)
    params = _complete_params(params, fixed_alpha=fixed_alpha)

    target, _, _ = _empirical_target_matrix(
        train_df,
        fit_with=fit_with,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    matrices = _pooled_model_matrices(
        df=train_df,
        model_outputs=model_outputs,
        params=params,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=base_seed,
        model_seed_count=model_seed_count,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    if fit_with == "conditional":
        sim = matrices["Final_conditional"]
    elif fit_with == "update":
        sim = matrices["Final_update"]
    else:
        raise ValueError("fit_with must be 'conditional' or 'update'.")

    return matrix_error(sim, target)


def _make_sampler(sampler_type, seed):
    sampler_type = sampler_type.upper()

    if sampler_type == "TPE":
        return optuna.samplers.TPESampler(seed=seed)

    if sampler_type == "CMA-ES":
        return optuna.samplers.CmaEsSampler(seed=seed)

    raise ValueError("sampler_type must be 'TPE' or 'CMA-ES'.")


def _fit_one_fold(
    train_df,
    test_df,
    model_outputs,
    search_space,
    repetition_idx,
    fold_idx,
    partition_seed,
    sampler_type,
    n_trials,
    mode_pre,
    fit_with,
    model_seed_count,
    fixed_alpha=None,
    n_bins=8,
):
    n_burn = _get_n_burn(mode_pre)

    s_train, chooseB_train, _, no_response_train, _, _ = _extract_arrays(train_df)

    _, fit_params = total_psychometric(
        s_train[n_burn:],
        chooseB_train[n_burn:],
        no_response_train[n_burn:],
    )

    lambda_A = np.clip(fit_params[2], 0.0, 0.5)
    lambda_B = np.clip(fit_params[3], 0.0, 0.5)

    optuna_seed = int(7 * partition_seed + 100 * fold_idx)

    objective = lambda trial: train_objective_pooled(
        trial=trial,
        train_df=train_df,
        model_outputs=model_outputs,
        search_space=search_space,
        fit_with=fit_with,
        mode_pre=mode_pre,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=partition_seed * 100 + fold_idx,
        model_seed_count=model_seed_count,
        fixed_alpha=fixed_alpha,
        n_bins=n_bins,
    )

    sampler = _make_sampler(sampler_type, optuna_seed)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = _complete_params(study.best_params, fixed_alpha=fixed_alpha)

    train_target, train_target_update, train_target_conditional = _empirical_target_matrix(
        train_df,
        fit_with=fit_with,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    test_target, test_target_update, test_target_conditional = _empirical_target_matrix(
        test_df,
        fit_with=fit_with,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    train_matrices = _pooled_model_matrices(
        df=train_df,
        model_outputs=model_outputs,
        params=best_params,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=partition_seed * 100 + fold_idx,
        model_seed_count=model_seed_count,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    test_matrices = _pooled_model_matrices(
        df=test_df,
        model_outputs=model_outputs,
        params=best_params,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=partition_seed * 100 + fold_idx + 500000,
        model_seed_count=model_seed_count,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    train_agent_choices = _agent_choices_table(
        df=train_df,
        model_outputs=model_outputs,
        params=best_params,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=partition_seed * 100 + fold_idx,
        model_seed_count=model_seed_count,
        mode_pre=mode_pre,
    )

    test_agent_choices = _agent_choices_table(
        df=test_df,
        model_outputs=model_outputs,
        params=best_params,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=partition_seed * 100 + fold_idx + 500000,
        model_seed_count=model_seed_count,
        mode_pre=mode_pre,
    )

    train_agent_choices["repetition"] = repetition_idx
    train_agent_choices["fold"] = fold_idx + 1
    train_agent_choices["set"] = "train"

    test_agent_choices["repetition"] = repetition_idx
    test_agent_choices["fold"] = fold_idx + 1
    test_agent_choices["set"] = "test"

    if fit_with == "conditional":
        train_sim = train_matrices["Final_conditional"]
        test_sim = test_matrices["Final_conditional"]
    else:
        train_sim = train_matrices["Final_update"]
        test_sim = test_matrices["Final_update"]

    train_error = matrix_error(train_sim, train_target)
    test_error = matrix_error(test_sim, test_target)

    return {
        "repetition_idx": repetition_idx,
        "fold_idx": fold_idx,
        "partition_seed": partition_seed,
        "best_params": best_params,
        "train_error": train_error,
        "test_error": test_error,
        "lambda_A": lambda_A,
        "lambda_B": lambda_B,
        "train_target_update": train_target_update,
        "train_target_conditional": train_target_conditional,
        "test_target_update": test_target_update,
        "test_target_conditional": test_target_conditional,
        "train_model_matrices": train_matrices,
        "test_model_matrices": test_matrices,
        "study_best_value": study.best_value,
        "train_agent_choices": train_agent_choices,
        "test_agent_choices": test_agent_choices,
        "n_train_trials": len(train_df),
        "n_test_trials": len(test_df),
    }



def run_one_repetition_optimization(
    df,
    model_outputs,
    search_space,
    repetition_idx,
    partition_seed,
    k=2,
    n_trials=200,
    mode_pre="simulated",
    fit_with="conditional",
    sampler_type="TPE",
    model_seed_count=5,
    fixed_alpha=None,
    n_bins=8,
):
    block_sizes = df.groupby("block")["Trial"].count().reset_index(name="count")
    sizes = block_sizes["count"].to_numpy()
    labels = block_sizes["block"].to_numpy()

    if len(sizes) == 1:
        raise ValueError("Only one block found. Cannot perform cross-validation.")

    if len(sizes) < k:
        k = len(sizes)

    blocks_in_folds = create_random_folds(
        sizes,
        labels,
        num_folds=k,
        seed=4 * int(partition_seed),
    )

    partition = {
        "repetition_idx": repetition_idx,
        "partition_seed": int(partition_seed),
        "blocks_in_folds": blocks_in_folds,
    }

    fits = []

    for fold_idx in range(k):
        print(f"\n--- Repetition {repetition_idx}, Fold {fold_idx + 1}/{k} ---")

        test_df, train_df = select_and_concatenate(
            df,
            blocks_in_folds,
            fold_idx,
        )

        fit_result = _fit_one_fold(
            train_df=train_df,
            test_df=test_df,
            model_outputs=model_outputs,
            search_space=search_space,
            repetition_idx=repetition_idx,
            fold_idx=fold_idx + 1,
            partition_seed=int(partition_seed),
            sampler_type=sampler_type,
            n_trials=n_trials,
            mode_pre=mode_pre,
            fit_with=fit_with,
            model_seed_count=model_seed_count,
            fixed_alpha=fixed_alpha,
            n_bins=n_bins,
        )

        fit_result["train_blocks"] = sorted(train_df["block"].unique().tolist())
        fit_result["test_blocks"] = sorted(test_df["block"].unique().tolist())

        fits.append(fit_result)

    return partition, fits


def run_repeated_cv_optimization(
    df,
    model_outputs,
    search_space,
    repetition_seeds,
    k=2,
    n_trials=500,
    mode_pre="simulated",
    fit_with="conditional",
    sampler_type="TPE",
    model_seed_count=10,
    fixed_alpha=None,
    n_bins=8,
):
    if fit_with not in ["conditional", "update"]:
        raise ValueError("fit_with must be 'conditional' or 'update'.")

    block_sizes = df.groupby("block")["Trial"].count().reset_index(name="count")
    sizes = block_sizes["count"].to_numpy()
    labels = block_sizes["block"].to_numpy()

    if len(sizes) == 1:
        raise ValueError("Only one block found. Cannot perform cross-validation.")

    if len(sizes) < k:
        k = len(sizes)

    all_fits = []
    partitions = []

    for r_idx, partition_seed in enumerate(repetition_seeds, start=1):
        blocks_in_folds = create_random_folds(
            sizes,
            labels,
            num_folds=k,
            seed=4 * int(partition_seed),
        )

        partitions.append({
            "repetition_idx": r_idx,
            "partition_seed": int(partition_seed),
            "blocks_in_folds": blocks_in_folds,
        })

        for fold_idx in range(k):
            print(f"\n--- Repetition {r_idx}/{len(repetition_seeds)}, Fold {fold_idx + 1}/{k} ---")

            test_df, train_df = select_and_concatenate(
                df,
                blocks_in_folds,
                fold_idx,
            )

            fit_result = _fit_one_fold(
                train_df=train_df,
                test_df=test_df,
                model_outputs=model_outputs,
                search_space=search_space,
                repetition_idx=r_idx,
                fold_idx=fold_idx + 1,
                partition_seed=int(partition_seed),
                sampler_type=sampler_type,
                n_trials=n_trials,
                mode_pre=mode_pre,
                fit_with=fit_with,
                model_seed_count=model_seed_count,
                fixed_alpha=fixed_alpha,
                n_bins=n_bins,
            )

            fit_result["train_blocks"] = sorted(train_df["block"].unique().tolist())
            fit_result["test_blocks"] = sorted(test_df["block"].unique().tolist())

            all_fits.append(fit_result)

    train_errors = np.array([x["train_error"] for x in all_fits], dtype=float)
    test_errors = np.array([x["test_error"] for x in all_fits], dtype=float)

    summary = {
        "mean_train_error": np.nanmean(train_errors),
        "mean_test_error": np.nanmean(test_errors),
        "std_train_error": np.nanstd(train_errors, ddof=1),
        "std_test_error": np.nanstd(test_errors, ddof=1),
        "sem_train_error": np.nanstd(train_errors, ddof=1) / np.sqrt(np.sum(~np.isnan(train_errors))),
        "sem_test_error": np.nanstd(test_errors, ddof=1) / np.sqrt(np.sum(~np.isnan(test_errors))),
        "n_fits": len(all_fits),
    }

    return {
        "summary": summary,
        "fits": all_fits,
        "partitions": partitions,
    }


def run_full_dataset_optimization(
    df,
    model_outputs,
    search_space,
    n_trials=200,
    mode_pre="simulated",
    fit_with="conditional",
    sampler_type="TPE",
    model_seed_count=5,
    fixed_alpha=None,
    n_bins=8,
    seed=1,
):
    if fit_with not in ["conditional", "update"]:
        raise ValueError("fit_with must be 'conditional' or 'update'.")

    n_burn = _get_n_burn(mode_pre)

    s, chooseB, _, no_response, _, _ = _extract_arrays(df)

    _, fit_params = total_psychometric(
        s[n_burn:],
        chooseB[n_burn:],
        no_response[n_burn:],
    )

    lambda_A = np.clip(fit_params[2], 0.0, 0.5)
    lambda_B = np.clip(fit_params[3], 0.0, 0.5)

    objective = lambda trial: train_objective_pooled(
        trial=trial,
        train_df=df,
        model_outputs=model_outputs,
        search_space=search_space,
        fit_with=fit_with,
        mode_pre=mode_pre,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=seed,
        model_seed_count=model_seed_count,
        fixed_alpha=fixed_alpha,
        n_bins=n_bins,
    )

    sampler = _make_sampler(sampler_type, seed)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    best_params = _complete_params(study.best_params, fixed_alpha=fixed_alpha)

    target, target_update, target_conditional = _empirical_target_matrix(
        df,
        fit_with=fit_with,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    model_matrices = _pooled_model_matrices(
        df=df,
        model_outputs=model_outputs,
        params=best_params,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=seed,
        model_seed_count=model_seed_count,
        mode_pre=mode_pre,
        n_bins=n_bins,
    )

    if fit_with == "conditional":
        sim = model_matrices["Final_conditional"]
    else:
        sim = model_matrices["Final_update"]

    train_error = matrix_error(sim, target)

    agent_choices = _agent_choices_table(
        df=df,
        model_outputs=model_outputs,
        params=best_params,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        base_seed=seed,
        model_seed_count=model_seed_count,
        mode_pre=mode_pre,
    )

    agent_choices["set"] = "full"

    return {
        "best_params": best_params,
        "train_error": train_error,
        "lambda_A": lambda_A,
        "lambda_B": lambda_B,
        "target_update": target_update,
        "target_conditional": target_conditional,
        "model_matrices": model_matrices,
        "study_best_value": study.best_value,
        "agent_choices": agent_choices,
        "n_trials_data": len(df),
        "optuna_n_trials": n_trials,
        "fit_with": fit_with,
        "mode_pre": mode_pre,
        "model_seed_count": model_seed_count,
        "fixed_alpha": fixed_alpha,
        "n_bins": n_bins,
        "seed": seed,
    }
