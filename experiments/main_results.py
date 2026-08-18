import os
import sys
import warnings

import numpy  # noqa: F401  (import before torch: avoids libstdc++ conflicts in mixed conda/pip envs)
import torch

from openpto.config import get_args, get_logger, load_conf, setup_seed
from openpto.expmanager import ExpManager
from openpto.method.Models.wrapper_loss import get_loss_fn
from openpto.method.Solvers.wrapper_solver import solver_wrapper
from openpto.problems.wrapper_prob import problem_wrapper

warnings.filterwarnings("ignore")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Makes sure hashes are consistent
hashseed = os.getenv("PYTHONHASHSEED")
if not hashseed:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

if __name__ == "__main__":
    # get configs
    args = get_args()
    conf = load_conf(args.config_path, args.method_path, args.problem)

    # set seed
    setup_seed(args.seed)

    # set logger
    logger = get_logger(args, conf)
    logger.info(f" {args.bkup_log_dir}\n {args.log_dir}\n args: {args} \n")

    # Load problem
    logger.info(f" dataset configs: {conf['dataset']} \n")
    logger.info(f" model configs: {conf['models'][args.opt_model]} \n")
    logger.info(f" Loading [{args.problem}] Problem...")
    problem = problem_wrapper(args, conf)

    # Re-seed AFTER problem construction so the dataset stays pinned to --seed
    # but model init + training-time RNG (DPO/PG perturbations, LODL surrogate
    # sampling, dropout, minibatch order, ...) follows --init_seed.
    if args.init_seed is not None:
        logger.info(f" Re-seeding for model init: init_seed={args.init_seed}")
        setup_seed(args.init_seed)

    # Optional train-set subsampling (first N instances on axis 0).
    if getattr(args, "train_subsample_n", 0) and args.train_subsample_n > 0:
        _orig_get_train = problem.get_train_data
        _n_sub = args.train_subsample_n

        def _subsampled_get_train(*a, **kw):
            out = _orig_get_train(*a, **kw)
            return tuple(
                t[:_n_sub] if hasattr(t, "__getitem__") and hasattr(t, "__len__") else t
                for t in out
            )

        problem.get_train_data = _subsampled_get_train
        logger.info(f" Train subsample: first {_n_sub} instances")

    # Load solver
    logger.info(f" Loading [{args.solver}] solver ...")
    ptoSolver = solver_wrapper(args, conf, problem)

    # Load loss function
    logger.info(f" Loading [{args.opt_model}] Loss Function...")
    loss_fn = get_loss_fn(args, ptoSolver, conf)

    # load exp manager
    pred_model_args = {
        "ipdim": problem.get_model_shape()[0],
        "opdim": problem.get_model_shape()[1],
        "out_act": problem.get_output_activation(),
    }
    exp = ExpManager(pred_model_args, args=args, conf=conf, logger=logger)

    # Optionally warm-start the prediction model from a saved checkpoint
    if args.warmstart_ckpt:
        sd = torch.load(args.warmstart_ckpt, map_location="cpu")
        exp.pred_model.load_state_dict(sd)
        logger.info(f" Warm-started pred model from: {args.warmstart_ckpt}")

    # Train neural network with a given loss function
    logger.info(
        f" Start training [{args.pred_model}] model on [{args.opt_model}] loss..."
    )
    exp.run(problem, loss_fn, ptoSolver, n_epochs=args.n_epochs, do_debug=args.do_debug)
