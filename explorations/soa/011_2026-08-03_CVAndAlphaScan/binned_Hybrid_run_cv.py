import argparse
import sys
import os

# =========================
# EDIT USERNAME ONCE HERE IF NEEDED
# =========================
USERNAME = "apourdehghan"
HOME_ROOT = f"/nfs/nhome/live/{USERNAME}"
# =========================

sys.path.append(HOME_ROOT)
import binned_Hybrid_hpc_utils as hpc_ut


parser = argparse.ArgumentParser()

parser.add_argument(
    "--fit_mode",
    type=str,
    default="cv",
    choices=["cv", "full", "full_alpha_scan"],
)


parser.add_argument("--p_id", type=str, required=True)

parser.add_argument(
    "--species",
    type=str,
    default="mouse",
    choices=["human", "rat", "mouse"],
)

parser.add_argument(
    "--model_mode",
    type=str,
    default="all",
    choices=["hybrid", "be", "sc", "all", "fixed_alpha"],
)

parser.add_argument(
    "--custom_alpha",
    type=float,
    default=None,
)

parser.add_argument(
    "--sampler",
    type=str,
    default="TPE",
    choices=["TPE", "CMA-ES"],
)

parser.add_argument("--trials", type=int, default=500)
parser.add_argument("--folds", type=int, default=2)
parser.add_argument("--repetitions", type=int, default=10)
parser.add_argument("--model_seeds", type=int, default=10)

parser.add_argument("--parallel_jobs", type=int, default=1)

parser.add_argument(
    "--alpha_values",
    type=str,
    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
)

args = parser.parse_args()

if (
    args.model_mode == "fixed_alpha"
    and args.fit_mode != "full_alpha_scan"
    and args.custom_alpha is None
):
    raise ValueError("--custom_alpha is required when --model_mode fixed_alpha")

save_path = hpc_ut.build_save_path(
    species=args.species,
    p_id=args.p_id,
    sampler_type=args.sampler,
    model_mode=args.model_mode,
    custom_alpha=args.custom_alpha,
    fit_mode=args.fit_mode,
)

repetition_seeds = list(range(1, args.repetitions + 1))

alpha_values = [
    float(x) for x in args.alpha_values.split(",")
]


hpc_ut.run_pid_repeated_cv(
    species=args.species,
    p_id=args.p_id,
    save_path=save_path,
    model_mode=args.model_mode,
    custom_alpha=args.custom_alpha,
    repetition_seeds=repetition_seeds,
    cv_folds=args.folds,
    optuna_trials=args.trials,
    sampler_type=args.sampler,
    mode_pre="simulated",
    fit_with="conditional",
    model_seed_count=args.model_seeds,
    n_bins=8,
    fit_mode=args.fit_mode,
    alpha_values=alpha_values,
    parallel_jobs=args.parallel_jobs,
)