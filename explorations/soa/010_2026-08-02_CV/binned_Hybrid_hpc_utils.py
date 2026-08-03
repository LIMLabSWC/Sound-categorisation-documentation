import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from functools import wraps
import logging
from joblib import Parallel, delayed

# =========================
# EDIT THESE IF NEEDED
# =========================
USERNAME = "athenaa"
HOME_ROOT = f"/nfs/nhome/live/{USERNAME}"
CEPH_ROOT = "/ceph/akrami/Amir"
PROJECT_NAME = "Hybrid_repeated_cv_binned"
# =========================

logging.basicConfig(
    filename="run_pid_cv_errors.log",
    filemode="a",
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s"
)

cwd = os.getcwd()
models_dir = os.path.join(cwd, "binned_Models")
sys.path.append(models_dir)

from Hybrid import Hybrid_model_outputs
from binned_Gradient_free_Opt import run_repeated_cv_optimization, run_one_repetition_optimization



def profile_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time:.2f} seconds.")
        return result
    return wrapper


def load_data(data_path):
    if data_path.endswith(".pkl"):
        return pd.read_pickle(data_path)
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path, low_memory=False)
    raise ValueError(f"Unsupported data file type: {data_path}")


def get_species_config(species):
    species = species.lower()

    configs = {
        "human": {
            "data_path": os.path.join(CEPH_ROOT, "Data_human_first_dists_V2.pkl"),
            "id_col": "Participant_Private_ID",
            "output_species": "Human",
        },
        "rat": {
            "data_path": os.path.join(CEPH_ROOT, "Data_rat_first_dists_V2.pkl"),
            "id_col": "Participant_ID",
            "output_species": "Rat",
        },
        "mouse": {
            "data_path": os.path.join(CEPH_ROOT, "SC_Data_Cleaned.pkl"),
            "id_col": "Participant_ID",
            "output_species": "Mouse",
        },
    }

    if species not in configs:
        raise ValueError("species must be one of: human, rat, mouse")

    return configs[species]


def save_results(results, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  ✔ Results saved to {save_path}")


def get_variant_specs(model_mode="all", custom_alpha=None):
    hybrid_space = {
        "sigma_noise": {"low": 0.01, "high": 0.45},
        "A_repulsion": {"low": 0.0, "high": 0.5},
        "gamma": {"low": 0.01, "high": 1.0},
        "sigma_update": {"low": 0.01, "high": 0.7},
        "eta_learning": {"low": 0.01, "high": 1.0},
        "sigma_boundary": {"low": 0.01, "high": 10.0},
        "alpha": {"low": 0.0, "high": 1.0},
    }

    be_space = {
        "sigma_noise": {"low": 0.01, "high": 0.45},
        "A_repulsion": {"low": 0.0, "high": 0.5},
        "eta_learning": {"low": 0.01, "high": 1.0},
        "sigma_boundary": {"low": 0.01, "high": 10.0},
    }

    sc_space = {
        "sigma_noise": {"low": 0.01, "high": 0.45},
        "A_repulsion": {"low": 0.0, "high": 0.5},
        "gamma": {"low": 0.01, "high": 1.0},
        "sigma_update": {"low": 0.01, "high": 0.7},
    }

    specs = {
        "hybrid": {
            "Hybrid": {
                "search_space": hybrid_space,
                "fixed_alpha": None,
            }
        },
        "be": {
            "Hybrid_alpha0_BE": {
                "search_space": be_space,
                "fixed_alpha": 0.0,
            }
        },
        "sc": {
            "Hybrid_alpha1_SC": {
                "search_space": sc_space,
                "fixed_alpha": 1.0,
            }
        },
        "all": {
            "Hybrid": {
                "search_space": hybrid_space,
                "fixed_alpha": None,
            },
            "Hybrid_alpha0_BE": {
                "search_space": be_space,
                "fixed_alpha": 0.0,
            },
            "Hybrid_alpha1_SC": {
                "search_space": sc_space,
                "fixed_alpha": 1.0,
            },
        },
    }

    model_mode = model_mode.lower()

    if model_mode == "fixed_alpha":
        if custom_alpha is None:
            raise ValueError("custom_alpha must be provided when model_mode='fixed_alpha'")
        if not (0.0 <= custom_alpha <= 1.0):
            raise ValueError("custom_alpha must be between 0 and 1")

        return {
            f"Hybrid_fixed_alpha_{custom_alpha:.3f}".replace(".", "p"): {
                "search_space": be_space | sc_space,
                "fixed_alpha": float(custom_alpha),
            }
        }

    if model_mode not in specs:
        raise ValueError("model_mode must be one of: hybrid, be, sc, all, fixed_alpha")

    return specs[model_mode]


def normalise_pid_for_matching(pid, species):
    if species.lower() == "human":
        return float(pid)
    return str(pid)


def filter_subject_df(df, p_id, species, id_col):
    if species.lower() == "human":
        pid_value = float(p_id)
        subject_df = df[df[id_col].astype(float) == pid_value].reset_index(drop=True)
    else:
        pid_value = str(p_id)
        subject_df = df[df[id_col].astype(str) == pid_value].reset_index(drop=True)

    return subject_df


@profile_time
def run_pid_repeated_cv(
    species,
    p_id,
    save_path,
    model_mode="all",
    custom_alpha=None,
    repetition_seeds=None,
    cv_folds=2,
    optuna_trials=500,
    sampler_type="TPE",
    mode_pre="simulated",
    fit_with="conditional",
    model_seed_count=10,
    n_bins=8,
    parallel_jobs=1,
):
    species_config = get_species_config(species)

    data_path = species_config["data_path"]
    id_col = species_config["id_col"]

    if repetition_seeds is None:
        repetition_seeds = list(range(1, 11))

    df = load_data(data_path)

    pid_df = filter_subject_df(
        df=df,
        p_id=p_id,
        species=species,
        id_col=id_col,
    )

    if len(pid_df) == 0:
        raise ValueError(
            f"No data found for species={species}, {id_col}={p_id}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if "is_not_start_of_block" not in pid_df.columns:
        pid_df["is_not_start_of_block"] = pid_df["block"].eq(pid_df["block"].shift())

    if "No_response" not in pid_df.columns:
        pid_df["No_response"] = False

    variants = get_variant_specs(
        model_mode=model_mode,
        custom_alpha=custom_alpha,
    )

    print(f"  ✱ Running repeated CV")
    print(f"  ✱ Species: {species}")
    print(f"  ✱ Subject: {p_id}")
    print(f"  ✱ ID column: {id_col}")
    print(f"  ✱ Data path: {data_path}")
    print(f"  ✱ Model mode: {model_mode}")
    print(f"  ✱ Custom alpha: {custom_alpha}")
    print(f"  ✱ Repetitions: {len(repetition_seeds)}")
    print(f"  ✱ Folds: {cv_folds}")
    print(f"  ✱ Model seeds per objective: {model_seed_count}")
    print(f"  ✱ Sampler: {sampler_type}")

    results = {
        "metadata": {
            "Participant_ID_used": p_id,
            "species": species,
            "id_col": id_col,
            "data_path": data_path,
            "cv_folds": cv_folds,
            "optuna_trials": optuna_trials,
            "sampler": sampler_type,
            "mode_pre": mode_pre,
            "fit_with": fit_with,
            "model_seed_count": model_seed_count,
            "repetition_seeds": repetition_seeds,
            "n_bins": n_bins,
            "model_mode": model_mode,
            "custom_alpha": custom_alpha,
            "save_path": save_path,
        },
        "variants": {},
    }

    tasks = []

    for variant_name, spec in variants.items():
        for r_idx, partition_seed in enumerate(repetition_seeds, start=1):
            tasks.append(
                delayed(run_one_repetition_optimization)(
                    df=pid_df,
                    model_outputs=Hybrid_model_outputs,
                    search_space=spec["search_space"],
                    repetition_idx=r_idx,
                    partition_seed=int(partition_seed),
                    k=cv_folds,
                    n_trials=optuna_trials,
                    mode_pre=mode_pre,
                    fit_with=fit_with,
                    sampler_type=sampler_type,
                    model_seed_count=model_seed_count,
                    fixed_alpha=spec["fixed_alpha"],
                    n_bins=n_bins,
                )
            )

    task_keys = [
        (variant_name, r_idx)
        for variant_name in variants.keys()
        for r_idx, _ in enumerate(repetition_seeds, start=1)
    ]

    outputs = Parallel(n_jobs=parallel_jobs, backend="loky")(tasks)

    grouped = {
        variant_name: {
            "fits": [],
            "partitions": [],
        }
        for variant_name in variants.keys()
    }

    for (variant_name, r_idx), (partition, fits) in zip(task_keys, outputs):
        grouped[variant_name]["partitions"].append(partition)
        grouped[variant_name]["fits"].append(fits)

    for variant_name, variant_data in grouped.items():

        all_fits = [fit for rep_fits in variant_data["fits"] for fit in rep_fits]

        train_errors = np.array([x["train_error"] for x in all_fits], dtype=float)
        test_errors  = np.array([x["test_error"]  for x in all_fits], dtype=float)

        variant_data["summary"] = {
            "mean_train_error": np.nanmean(train_errors),
            "mean_test_error": np.nanmean(test_errors),
            "std_train_error": np.nanstd(train_errors, ddof=1),
            "std_test_error": np.nanstd(test_errors, ddof=1),
            "sem_train_error": np.nanstd(train_errors, ddof=1) / np.sqrt(np.sum(~np.isnan(train_errors))),
            "sem_test_error": np.nanstd(test_errors, ddof=1) / np.sqrt(np.sum(~np.isnan(test_errors))),
            "n_fits": len(variant_data["fits"]),
            "parallel_jobs": parallel_jobs,
        }

        results["variants"][variant_name] = variant_data

    save_results(results, save_path)

    print(f"  ✔ Completed {species} subject {p_id}")
    return results


def build_save_path(species, p_id, sampler_type="TPE", model_mode="all", custom_alpha=None):
    species_config = get_species_config(species)
    species_folder = species_config["output_species"]

    if model_mode == "fixed_alpha":
        model_folder = f"fixed_alpha_{float(custom_alpha):.3f}".replace(".", "p")
    else:
        model_folder = model_mode

    save_root = os.path.join(
        CEPH_ROOT,
        PROJECT_NAME,
        species_folder,
        "results",
        sampler_type,
        model_folder,
        str(p_id),
    )

    save_name = f"results_{species}_{p_id}_{sampler_type}_{model_folder}_all_repetitions.pkl"
    return os.path.join(save_root, save_name)