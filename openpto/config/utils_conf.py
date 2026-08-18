import argparse
import ast
import os
import random

from datetime import datetime

import numpy as np
import ruamel.yaml as yaml
import torch

###################################### Args ###############################################


def get_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument(
        "--problem",
        type=str,
        choices=[
            "budgetalloc",
            "bipartitematching",
            "cubic",
            "portfolio",
            "knapsack",
            "energy",
            "shortestpath",
            "sp_synth",
            "sp_planted",
            "asurv",
            "cook_county",
            "speed_humps",
            "pg_misspec",
        ],
        default="knapsack",
    )
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument(
        "--method_path", type=str, default="openpto/config/models/default.yaml"
    )
    parser.add_argument("--trained_path", type=str, default="")
    parser.add_argument("--loss_path", type=str, default="")
    parser.add_argument(
        "--opt_model",
        type=str,
        choices=[
            "mse",
            "mse_train",
            "mse_val",
            "dfl",
            "bce",
            "ce",
            "mae",
            "spo",
            "pointLTR",
            "pairLTR",
            "listLTR",
            "blackbox",
            "blackboxSolver",
            "identity",
            "identitySolver",
            "lodl",
            "nce",
            "qptl",
            "lodl",
            "perturb",
            "cpLayer",
            "pg",
            "dad",
        ],
        default="mse",
    )
    parser.add_argument(
        "--pred_model",
        type=str,
        choices=[
            "dense",
            "cvr",
            "cv_mlp",
            "ConvNet",
            "Resnet18",
            "CombResnet18",
            "PureConvNet",
        ],
        default="dense",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=[
            "gurobi",
            "neural",
            "heuristic",
            "cvxpy",
            "qptl",
        ],
        default="gurobi",
    )
    parser.add_argument("--gpu", type=str, default="-1", help="Visible GPU")
    # training
    parser.add_argument("--loadnew", type=ast.literal_eval, default=False)
    parser.add_argument("--opt_name", type=str, default="gd", choices=["gd", "sgd"])
    parser.add_argument("--n_epochs", type=int, default=0)
    parser.add_argument("--n_ptr_epochs", type=int, default=0)
    parser.add_argument("--earlystopping", type=ast.literal_eval, default=True)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--init_seed", type=int, default=None,
                        help="If set, re-seed torch/np/random after problem construction. "
                             "Controls model init + training-time RNG only; dataset still "
                             "uses --seed. Used for multi-seed runs that share a dataset.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--use_lr_scheduling", action="store_true")
    parser.add_argument("--lr_milestone_1", type=int, default=100)
    parser.add_argument("--lr_milestone_2", type=int, default=200)
    parser.add_argument("--l1_weight", type=float, default=0)
    parser.add_argument("--l2_weight", type=float, default=0)
    # data
    parser.add_argument("--data_dir", type=str, default="./openpto/data/")
    parser.add_argument("--do_debug", action="store_true")
    parser.add_argument("--instances", type=int, default=400)
    parser.add_argument("--testinstances", type=int, default=200)
    parser.add_argument("--val_frac", type=float, default=0.2)
    # debug
    parser.add_argument("--valfreq", type=int, default=1)
    parser.add_argument("--skip_solver_eval", action="store_true",
                        help="Skip solver calls during training; use val MSE for early stopping. "
                             "Solver is still called once at the end for final test evaluation.")
    parser.add_argument("--solver_valfreq", type=int, default=0,
                        help="With --skip_solver_eval: run solver on val every N epochs for "
                             "real val regret checkpoint selection. 0 = use val MSE only.")
    parser.add_argument("--selection_signal", type=str,
                        choices=["val_regret", "val_mse", "train_mse"],
                        default="val_regret",
                        help="Signal driving checkpoint selection and early stopping. "
                             "val_regret (default): existing behavior. "
                             "val_mse: equivalent to --skip_solver_eval --solver_valfreq 0. "
                             "train_mse: select on training MSE; skip all solver eval during training.")
    parser.add_argument("--savefreq", type=int, default=-1)
    parser.add_argument("--prefix", type=str, default="default")
    # model
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=32)
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel_size", type=int, default=1)
    parser.add_argument("--warmstart_ckpt", type=str, default=None,
                        help="Path to a .pt checkpoint to warm-start the pred model before fine-tuning.")
    parser.add_argument("--train_subsample_n", type=int, default=0,
                        help="If > 0, keep only the first N training instances "
                             "(slice on axis 0 of each tensor returned by problem.get_train_data()). "
                             "For CookCounty this means keep N timesteps (years).")
    parser.add_argument("--grad_surgery", action="store_true",
                        help="Gradient surgery: remove MSE-misaligned component of PnO gradient "
                             "and inject surgery_weight * g_mse. Requires --opt_name sgd.")
    parser.add_argument("--surgery_weight", type=float, default=1.0,
                        help="Weight for injected MSE gradient in surgery: "
                             "g_update = (g_p - proj_{g_m}(g_p)) + surgery_weight * g_m.")

    args = parser.parse_args()

    args.data_dir = os.path.join(args.data_dir, args.problem)
    return args


###################################### Configs ###############################################


def load_conf(prob_path: str = None, method_path: str = None, prob_name: str = None):
    """
    Function to load config file.

    Parameters
    ----------
    prob_path : str
        Path to load config file. Load default configuration if set to `None`.
    method_path : str
        Path to load method config file. Necessary if ``path`` is set to `None`.
    prob_name : str
        Name of the corresponding problem. Necessary if ``path`` is set to `None`.

    Returns
    -------
    conf : argparse.Namespace
        The config file converted to Namespace.

    """
    if prob_path == "":
        dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/probs/"
        )
        prob_path = os.path.join(dir, prob_name + ".yaml")

    if os.path.exists(prob_path) is False:
        raise ValueError(f"The configuration file, [{prob_path}] is not provided.")

    conf = yaml.safe_load(open(prob_path, "r").read())
    conf["models"] = yaml.safe_load(open(method_path, "r").read())
    # conf = argparse.Namespace(**conf)
    return conf


def save_conf(path, conf):
    """
    Function to save the config file.

    Parameters
    ----------
    path : str
        Path to save config file.
    conf : dict
        The config dict.

    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(vars(conf), f)


def get_logger(args, conf):
    import logging

    #
    log_dir = os.path.join(
        "saved_records",
        args.problem + "-" + conf["dataset"]["prob_version"],
        args.opt_model,
        args.prefix,
    )
    args.log_dir = log_dir
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

    now = datetime.now()
    formatted_time = f"{now.year:04d}-{now.month:02d}-{now.day:02d} {now.hour:02d}-{now.minute:02d}-{now.second:02d}.{now.microsecond:06d}"
    bkup_log_dir = os.path.join(
        "saved_records",
        "timed_logs",
        args.problem + "-" + conf["dataset"]["prob_version"],
        args.opt_model,
        args.prefix,
        formatted_time,
    )
    args.bkup_log_dir = bkup_log_dir
    os.makedirs(os.path.join(bkup_log_dir, "checkpoints"), exist_ok=True)
    if args.do_debug:
        os.makedirs(os.path.join(bkup_log_dir, "tensors"), exist_ok=True)

    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s-%(levelname)s:  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # FileHandler: write to file
    fh = logging.FileHandler(f"{log_dir}/log.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # FileHandler2:
    fh2 = logging.FileHandler(f"{bkup_log_dir}/log.txt")
    fh2.setLevel(logging.DEBUG)
    fh2.setFormatter(formatter)

    # StreamHandler: output to shell
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add Handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.addHandler(fh2)
    return logger


################################### Seed ###################################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
