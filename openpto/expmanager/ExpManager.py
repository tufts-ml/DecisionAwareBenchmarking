import csv
import os
import signal
import sys
import time

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

# from torchsummary import summary
from openpto.expmanager.utils_manager import (
    ExpDataset,
    add_log,
    compare_result,
    print_metrics,
    prob_to_gpu,
    save_pd,
)
from openpto.method.Models.utils_loss import l1_penalty, l2_penalty, str2twoStageLoss
from openpto.method.Predicts.wrapper_predicts import pred_model_wrapper
from openpto.method.utils_method import do_reduction, ndiv, to_array, to_device


def _append_csv(path, row_dict):
    """Append one row to a CSV log. Creates the file with header if it doesn't exist."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


class ExpManager:
    """
    Experiment management class to enable running multiple experiment.

    Parameters
    ----------
    """

    def __init__(self, pred_model_args, args, conf, logger):
        self.args = args
        self.conf = conf
        self.logger = logger
        self.batch_size = self.args.batch_size
        self.model_args = self.conf["models"][self.args.opt_model]
        self.device = torch.device(
            f"cuda:{args.gpu}"
            if torch.cuda.is_available() and int(args.gpu) >= 0
            else "cpu"
        )
        self.logger.info(f"--- Running on {self.device}")
        # prediction model
        self.pred_model = pred_model_wrapper(args, pred_model_args)
        print("self.pred_model: ", self.pred_model)
        self.logger.info(f"--- Built [{args.pred_model}] Prediction Model")
        # optimizer
        # optimizer:
        self.optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=args.lr)
        if args.use_lr_scheduling:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[args.lr_milestone_1, args.lr_milestone_2],
                gamma=0.3162,
            )

    def run(self, problem, loss_fn, ptoSolver=None, n_epochs=1, do_debug=False):
        #   Move everything to device
        prob_to_gpu(problem, self.device)
        prob_to_gpu(ptoSolver, self.device)
        problem.device = self.device
        self.pred_model = self.pred_model.to(self.device)

        # ---- Checkpoint / resume setup ----
        _ckpt_path = os.path.join(self.args.log_dir, "checkpoints", "checkpoint_latest.pt")
        _ckpt_best_path = os.path.join(self.args.log_dir, "checkpoints", "checkpoint_best.pt")
        _train_log_path = os.path.join(self.args.log_dir, "train_logs.csv")
        _val_log_path = os.path.join(self.args.log_dir, "val_logs.csv")

        # Mutable state shared with the SIGTERM handler (updated each epoch).
        _ckpt_state = {
            "epoch": 0,
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "best_val": float("inf"),
            "best_model_state_dict": None,
            "time_since_best": 0,
            "best_epoch": 0,
        }

        def _save_latest():
            _ckpt_state["model_state_dict"] = self.pred_model.state_dict()
            _ckpt_state["optimizer_state_dict"] = self.optimizer.state_dict()
            torch.save(_ckpt_state, _ckpt_path)
            self.logger.info(f"  [ckpt] saved checkpoint_latest at epoch {_ckpt_state['epoch']}")

        # Register SIGTERM handler so preempted jobs save state and exit cleanly.
        # SLURM sends SIGTERM 30 s before kill; --requeue causes SLURM to requeue automatically.
        _orig_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum, frame):
            self.logger.info("  [ckpt] SIGTERM received — saving checkpoint and exiting")
            _save_latest()
            signal.signal(signal.SIGTERM, _orig_sigterm)
            sys.exit(0)

        signal.signal(signal.SIGTERM, _sigterm_handler)

        # Try to resume from a previous (preempted) run.
        _resume_epoch = 0
        if os.path.exists(_ckpt_path):
            _prev = torch.load(_ckpt_path, map_location="cpu")
            _resume_epoch = _prev.get("epoch", 0)
            if _resume_epoch > 0:
                self.pred_model.load_state_dict(_prev["model_state_dict"])
                self.pred_model = self.pred_model.to(self.device)
                _ckpt_state.update(_prev)
                self.logger.info(
                    f"  [ckpt] Resuming from epoch {_resume_epoch} "
                    f"(best_val={_prev['best_val']:.6f}, time_since_best={_prev['time_since_best']})"
                )

        ############################## Data ##############################
        # Get data
        X_train, Y_train, Y_train_aux = problem.get_train_data()
        X_val, Y_val, Y_val_aux = problem.get_val_data()
        X_test, Y_test, Y_test_aux = problem.get_test_data()

        ############################## Preliminary Evaluation ##############################
        #   Document the optimal value
        # TODO: !!! use exact sovler for optimal
        Z_train_opt, Objs_train_opt = problem.get_decision(
            Y_train,
            params=Y_train_aux,
            ptoSolver=ptoSolver,
            isTrain=False,
            **problem.init_API(),
        )
        Z_val_opt, Objs_val_opt = problem.get_decision(
            Y_val,
            params=Y_val_aux,
            ptoSolver=ptoSolver,
            isTrain=False,
            **problem.init_API(),
        )

        Objs_val_opt = to_array(Objs_val_opt)
        #
        Z_test_opt, Objs_test_opt = problem.get_decision(
            Y_test,
            params=Y_test_aux,
            ptoSolver=ptoSolver,
            isTrain=False,
            **problem.init_API(),
        )
        Objs_test_opt = to_array(Objs_test_opt)
        # save
        problem.z_train_opt = Z_train_opt
        problem.z_val_opt = Z_val_opt
        problem.z_test_opt = Z_test_opt
        ###   Document the value of a random guess
        # objs_rand = list()
        # for _ in range(10):
        #     rand_Y = rand_like(Y_test, device=self.device)
        #     Z_test_rand, Objs_test_rand = problem.get_decision(
        #         rand_Y,
        #         params=Y_test_aux,
        #         ptoSolver=ptoSolver,
        #         isTrain=False,
        #         **problem.init_API(),
        #     )
        #     objs_rand.append(Objs_test_rand)
        # objs_rand = torch.stack(to_tensor(objs_rand))
        objs_rand = torch.zeros(10)
        ############################# Load previous model #############################
        if self.args.trained_path != "":
            self.pred_model.load_state_dict(torch.load(self.args.trained_path))
            self.logger.info(f"--- Loaded model from {self.args.trained_path}")
        ############################# Pretrain #############################
        # fetch pretrain data:
        if hasattr(problem, "get_pretrain_data"):
            X_pretrain, Y_pretrain, Y_pretrain_aux = problem.get_pretrain_data()
        else:
            X_pretrain, Y_pretrain, Y_pretrain_aux = X_train, Y_train, Y_train_aux

        # Pretrain prediction model
        total_train_time = 0.0
        best = (float("inf"), None)
        time_since_best = 0
        train_logs = {
            "epoch": [],
            "obj": [],
            "loss": [],
            "pred_loss": [],
            "eval": [],
        }
        val_logs = {
            "epoch": [],
            "obj": [],
            "loss": [],
            "pred_loss": [],
            "eval": [],
        }
        # loss function
        twostage_criterion = str2twoStageLoss(problem)
        self.logger.info("Pretraining Prediction Model...")
        self.pred_model.train()
        best_epoch = 0
        for ptr_epoch in range(1, self.args.n_ptr_epochs + 1):
            ###### one-shot training
            time_train_start = time.time()
            preds = self.pred_model(X_pretrain)
            loss = twostage_criterion(problem, preds, Y_pretrain, **self.model_args)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update time
            time_since_best += 1
            total_train_time += time.time() - time_train_start

            if do_debug:
                torch.save(
                    preds.detach().cpu(),
                    os.path.join(
                        self.args.bkup_log_dir, "tensors", f"preds-ptr-EP{ptr_epoch}.pt"
                    ),
                )
            ###### Check metrics on val set
            self.logger.info(
                f"Previous best epoch: {best_epoch}, time since best: {time_since_best}"
            )
            if ptr_epoch % self.args.valfreq != 0:
                datasets = [
                    (X_pretrain, Y_pretrain, Y_pretrain_aux, "pretrain"),
                ]
                metrics = print_metrics(
                    datasets,
                    self.pred_model,
                    problem,
                    twostage_criterion,
                    twostage_criterion,
                    ptoSolver,
                    f"Ptr iter {ptr_epoch},",
                    self.logger,
                    do_debug=do_debug,
                    batch_size=self.args.batch_size,
                    **self.model_args,
                )
                add_log(train_logs, "Ptr-" + str(ptr_epoch), metrics, "pretrain")
            else:
                # Compute metrics
                datasets = [
                    (X_pretrain, Y_pretrain, Y_pretrain_aux, "pretrain"),
                    (X_val, Y_val, Y_val_aux, "val"),
                ]
                metrics = print_metrics(
                    datasets,
                    self.pred_model,
                    problem,
                    twostage_criterion,
                    twostage_criterion,
                    ptoSolver,
                    f"Ptr iter {ptr_epoch},",
                    self.logger,
                    do_debug=do_debug,
                    batch_size=self.args.batch_size,
                    **self.model_args,
                )
                add_log(train_logs, "Ptr-" + str(ptr_epoch), metrics, "pretrain")
                add_log(val_logs, "Ptr-" + str(ptr_epoch), metrics, "val")
                # Save model if it's the best one
                if best[1] is None or compare_result(metrics["val"], best):
                    best = (metrics["val"]["eval"]["value"], deepcopy(self.pred_model))
                    time_since_best = 0
                    best_epoch = ptr_epoch
                    # save
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(
                            self.args.log_dir, "checkpoints", "ptr_pred_best.pt"
                        ),
                    )
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(
                            self.args.bkup_log_dir, "checkpoints", "ptr_pred_best.pt"
                        ),
                    )
            # Stop if model hasn't improved for patience steps
            if self.args.earlystopping and time_since_best > self.args.patience:
                break

        if best[1]:
            self.pred_model = deepcopy(best[1])

        ############################# Train #############################
        # optimizer:
        self.optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=self.args.lr)
        # If resuming, restore optimizer state and best-model state.
        if _resume_epoch > 0 and _ckpt_state.get("optimizer_state_dict") is not None:
            try:
                self.optimizer.load_state_dict(_ckpt_state["optimizer_state_dict"])
            except Exception as e:
                self.logger.warning(f"  [ckpt] Could not restore optimizer state: {e}")
        if self.args.use_lr_scheduling:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[self.args.lr_milestone_1, self.args.lr_milestone_2],
                gamma=0.3162,
            )
        # dataset loader
        train_dataset = ExpDataset(X_train, Y_train, Y_train_aux)
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        # Train PTO — restore best-model state for checkpoint-resume
        time_since_best = _ckpt_state["time_since_best"] if _resume_epoch > 0 else 0
        self.logger.info("Training Model...")
        self.pred_model.train()
        best_epoch = _ckpt_state["best_epoch"] if _resume_epoch > 0 else 0
        # Restore best=(best_val_tensor, best_model) from checkpoint if resuming
        if _resume_epoch > 0 and _ckpt_state.get("best_model_state_dict") is not None:
            _best_model = deepcopy(self.pred_model)
            _best_model.load_state_dict(_ckpt_state["best_model_state_dict"])
            best = (torch.tensor(_ckpt_state["best_val"]), _best_model)
        for iter_idx in range(1, n_epochs + 1):
            # Skip epochs already completed in a previous (preempted) run.
            if iter_idx <= _resume_epoch:
                continue
            time_train_start = time.time()
            losses = list()

            # Surgery: minibatch (--opt_name sgd) or full-batch (default/gd).
            if getattr(self.args, "grad_surgery", False):
                if self.args.opt_name == "sgd":
                    # --- Minibatch surgery: one optimizer step per batch ---
                    for batch_id, batch in enumerate(train_loader):
                        X_batch = batch["X"].to(self.device)
                        Y_batch = batch["Y"].to(self.device)
                        Y_aux_batch = batch["Y_aux"]
                        inst_idx = batch["idx"]
                        preds = self.pred_model(X_batch)
                        loss = loss_fn(
                            problem,
                            coeff_hat=preds,
                            coeff_true=Y_batch,
                            params=Y_aux_batch,
                            inst_idx=inst_idx,
                            partition="train",
                            index=batch_id,
                            do_debug=do_debug,
                            **self.model_args,
                        )
                        loss = do_reduction(loss, self.model_args["reduction"])
                        self.optimizer.zero_grad()
                        g_p = torch.autograd.grad(loss, preds, retain_graph=True)[0]
                        g_m = torch.autograd.grad(
                            F.mse_loss(preds, Y_batch), preds
                        )[0]
                        gp_flat = g_p.reshape(-1).float()
                        gm_flat = g_m.reshape(-1).float()
                        dot = (gp_flat * gm_flat).sum()
                        norm_sq = (gm_flat * gm_flat).sum() + 1e-12
                        g_proj = (dot / norm_sq) * gm_flat
                        g_update = (
                            gp_flat - g_proj + self.args.surgery_weight * gm_flat
                        ).reshape(g_p.shape)
                        preds.backward(g_update)
                        self.optimizer.step()
                        if self.args.use_lr_scheduling:
                            self.scheduler.step()
                else:
                    # --- Full-batch surgery: single forward pass over all training data ---
                    preds = self.pred_model(X_train.to(self.device))
                    loss = loss_fn(
                        problem,
                        coeff_hat=preds,
                        coeff_true=Y_train.to(self.device),
                        params=Y_train_aux,
                        inst_idx=torch.arange(len(X_train)),
                        partition="train",
                        index=0,
                        do_debug=do_debug,
                        **self.model_args,
                    )
                    loss = do_reduction(loss, self.model_args["reduction"])
                    self.optimizer.zero_grad()
                    g_p = torch.autograd.grad(loss, preds, retain_graph=True)[0]
                    g_m = torch.autograd.grad(
                        F.mse_loss(preds, Y_train.to(preds.device)), preds
                    )[0]
                    gp_flat = g_p.reshape(-1).float()
                    gm_flat = g_m.reshape(-1).float()
                    dot = (gp_flat * gm_flat).sum()
                    norm_sq = (gm_flat * gm_flat).sum() + 1e-12
                    g_proj = (dot / norm_sq) * gm_flat
                    g_update = (gp_flat - g_proj + self.args.surgery_weight * gm_flat
                                ).reshape(g_p.shape)
                    preds.backward(g_update)
                    self.optimizer.step()
                    if self.args.use_lr_scheduling:
                        self.scheduler.step()
            else:
                for batch_id, batch in enumerate(train_loader):
                    X_batch, Y_batch, Y_aux_batch = batch["X"], batch["Y"], batch["Y_aux"]
                    inst_idx = batch["idx"]
                    preds = self.pred_model(X_batch)
                    loss = loss_fn(
                        problem,
                        coeff_hat=preds,
                        coeff_true=Y_batch,
                        params=Y_aux_batch,
                        inst_idx=inst_idx,
                        partition="train",
                        index=batch_id,
                        do_debug=do_debug,
                        **self.model_args,
                    )
                    if self.args.opt_name == "sgd":
                        loss = do_reduction(loss, self.model_args["reduction"])
                        # add penalty
                        if self.args.l1_weight > 0:
                            loss += self.args.l1_weight * l1_penalty(self.pred_model)
                        if self.args.l2_weight > 0:
                            loss += self.args.l2_weight * l2_penalty(self.pred_model)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        if self.args.use_lr_scheduling:
                            self.scheduler.step()
                    elif self.args.opt_name == "gd":
                        losses.append(loss)
                    else:
                        raise NotImplementedError

            if self.args.opt_name == "gd" and not getattr(self.args, "grad_surgery", False):
                if losses and losses[0].dim() == 0:
                    losses = do_reduction(torch.stack(losses), self.model_args["reduction"])
                else:
                    losses = do_reduction(torch.cat(losses), self.model_args["reduction"])
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()
                if self.args.use_lr_scheduling:
                    self.scheduler.step()

            # accuracy = (preds.round() * Y_train).sum() / Y_train.sum()
            # print("accuracy: ", accuracy)
            # loss = torch.stack(losses)
            time_since_best += 1
            total_train_time += time.time() - time_train_start

            # Step loss_fn scheduler (e.g. sigma schedule for perturbed)
            if hasattr(loss_fn, "step"):
                new_val = loss_fn.step(iter_idx)
                self.logger.info(f"  loss_fn.step({iter_idx}) -> {new_val}")

            ###### Check metrics on val set
            self.logger.info(
                f"Previous best epoch: {best_epoch}, time since best: {time_since_best}"
            )
            _should_stop = False
            if getattr(self.args, "selection_signal", "val_regret") == "train_mse":
                # train_mse path: select on training MSE, skip all solver eval
                # during training. Used by the mse_train pseudo-method in the
                # Phase 1/2 sweep.
                self.pred_model.eval()
                with torch.no_grad():
                    preds_train = self.pred_model(X_train)
                    train_mse_losses = twostage_criterion(
                        problem, preds_train, Y_train, **self.model_args
                    )
                    train_mse = float(do_reduction(train_mse_losses, "mean").item())
                self.pred_model.train()
                self.logger.info(f"Iter {iter_idx}, train MSE (no solver): {train_mse:.6f}")
                _append_csv(_train_log_path,
                            {"epoch": "Tr-" + str(iter_idx),
                             "obj": 0.0,
                             "loss": round(train_mse, 6),
                             "pred_loss": round(train_mse, 6),
                             "eval": round(train_mse, 6)})
                if best[1] is None or train_mse < (
                    float(best[0].mean()) if hasattr(best[0], "mean") else float(best[0])
                ):
                    best = (train_mse, deepcopy(self.pred_model))
                    time_since_best = 0
                    best_epoch = iter_idx
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(self.args.log_dir, "checkpoints", "tr_pred_best.pt"),
                    )
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(
                            self.args.bkup_log_dir, "checkpoints", "tr_pred_best.pt"
                        ),
                    )
                if self.args.earlystopping and time_since_best > self.args.patience:
                    _should_stop = True
            elif getattr(self.args, "skip_solver_eval", False):
                # Cheap path: skip train solver eval entirely.
                # Val MSE is computed every epoch for logging.
                # If solver_valfreq > 0, real val regret is computed every N epochs
                # and used for checkpoint selection / early stopping.
                # If solver_valfreq == 0, val MSE is used for checkpoint selection.
                self.pred_model.eval()
                with torch.no_grad():
                    preds_val = self.pred_model(X_val)
                    val_mse_losses = twostage_criterion(
                        problem, preds_val, Y_val, **self.model_args
                    )
                    val_mse = float(do_reduction(val_mse_losses, "mean").item())
                self.pred_model.train()
                self.logger.info(f"Iter {iter_idx}, val MSE (no solver): {val_mse:.6f}")

                solver_valfreq = getattr(self.args, "solver_valfreq", 0)
                if solver_valfreq > 0 and iter_idx % solver_valfreq == 0:
                    # Run solver on val set to get real val regret.
                    val_metrics = print_metrics(
                        [(X_val, Y_val, Y_val_aux, "val")],
                        self.pred_model,
                        problem,
                        loss_fn,
                        twostage_criterion,
                        ptoSolver,
                        f"Iter {iter_idx},",
                        self.logger,
                        do_debug=do_debug,
                        batch_size=self.args.batch_size,
                        **self.model_args,
                    )
                    add_log(val_logs, "Tr-" + str(iter_idx), val_metrics, "val")
                    _append_csv(_val_log_path,
                                {"epoch": "Tr-" + str(iter_idx),
                                 "obj": round(val_metrics["val"]["objective"].mean().item(), 6),
                                 "loss": round(val_metrics["val"]["loss"], 6),
                                 "pred_loss": round(val_metrics["val"]["pred_loss"], 6),
                                 "eval": round(float(val_metrics["val"]["eval"]["value"].mean()), 6)})
                    if best[1] is None or compare_result(val_metrics["val"], best):
                        best = (val_metrics["val"]["eval"]["value"], deepcopy(self.pred_model))
                        time_since_best = 0
                        best_epoch = iter_idx
                        torch.save(
                            self.pred_model.state_dict(),
                            os.path.join(self.args.log_dir, "checkpoints", "tr_pred_best.pt"),
                        )
                        torch.save(
                            self.pred_model.state_dict(),
                            os.path.join(
                                self.args.bkup_log_dir, "checkpoints", "tr_pred_best.pt"
                            ),
                        )
                elif solver_valfreq == 0:
                    # No solver valfreq: use val MSE for checkpoint selection.
                    if best[1] is None or val_mse < best[0]:
                        best = (val_mse, deepcopy(self.pred_model))
                        time_since_best = 0
                        best_epoch = iter_idx
                        torch.save(
                            self.pred_model.state_dict(),
                            os.path.join(self.args.log_dir, "checkpoints", "tr_pred_best.pt"),
                        )
                        torch.save(
                            self.pred_model.state_dict(),
                            os.path.join(
                                self.args.bkup_log_dir, "checkpoints", "tr_pred_best.pt"
                            ),
                        )
                if self.args.earlystopping and time_since_best > self.args.patience:
                    _should_stop = True
            elif iter_idx % self.args.valfreq != 0:
                datasets = [
                    (X_train, Y_train, Y_train_aux, "train"),
                ]
                metrics = print_metrics(
                    datasets,
                    self.pred_model,
                    problem,
                    loss_fn,
                    twostage_criterion,
                    ptoSolver,
                    f"Iter {iter_idx},",
                    self.logger,
                    do_debug=do_debug,
                    batch_size=self.args.batch_size,
                    **self.model_args,
                )
                add_log(train_logs, "Tr-" + str(iter_idx), metrics, "train")
                _append_csv(_train_log_path,
                            {"epoch": "Tr-" + str(iter_idx),
                             "obj": round(metrics["train"]["objective"].mean().item(), 6),
                             "loss": round(metrics["train"]["loss"], 6),
                             "pred_loss": round(metrics["train"]["pred_loss"], 6),
                             "eval": round(float(metrics["train"]["eval"]["value"].mean()), 6)})
            else:
                # Compute metrics
                datasets = [
                    (X_train, Y_train, Y_train_aux, "train"),
                    (X_val, Y_val, Y_val_aux, "val"),
                ]
                metrics = print_metrics(
                    datasets,
                    self.pred_model,
                    problem,
                    loss_fn,
                    twostage_criterion,
                    ptoSolver,
                    f"Iter {iter_idx},",
                    self.logger,
                    do_debug=do_debug,
                    batch_size=self.args.batch_size,
                    **self.model_args,
                )
                add_log(train_logs, "Tr-" + str(iter_idx), metrics, "train")
                add_log(val_logs, "Tr-" + str(iter_idx), metrics, "val")
                _append_csv(_train_log_path,
                            {"epoch": "Tr-" + str(iter_idx),
                             "obj": round(metrics["train"]["objective"].mean().item(), 6),
                             "loss": round(metrics["train"]["loss"], 6),
                             "pred_loss": round(metrics["train"]["pred_loss"], 6),
                             "eval": round(float(metrics["train"]["eval"]["value"].mean()), 6)})
                _append_csv(_val_log_path,
                            {"epoch": "Tr-" + str(iter_idx),
                             "obj": round(metrics["val"]["objective"].mean().item(), 6),
                             "loss": round(metrics["val"]["loss"], 6),
                             "pred_loss": round(metrics["val"]["pred_loss"], 6),
                             "eval": round(float(metrics["val"]["eval"]["value"].mean()), 6)})
                # Save model if it's the best one
                if best[1] is None or compare_result(metrics["val"], best):
                    best = (metrics["val"]["eval"]["value"], deepcopy(self.pred_model))
                    time_since_best = 0
                    best_epoch = iter_idx
                    # save
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(self.args.log_dir, "checkpoints", "tr_pred_best.pt"),
                    )
                    torch.save(
                        self.pred_model.state_dict(),
                        os.path.join(
                            self.args.bkup_log_dir, "checkpoints", "tr_pred_best.pt"
                        ),
                    )
                # Stop if model hasn't improved for patience steps
                if self.args.earlystopping and time_since_best > self.args.patience:
                    _should_stop = True

            # ---- Save checkpoint_latest after every completed epoch ----
            _ckpt_state["epoch"] = iter_idx
            _ckpt_state["time_since_best"] = time_since_best
            _ckpt_state["best_epoch"] = best_epoch
            if best[1] is not None:
                _best_val = best[0]
                _ckpt_state["best_val"] = (
                    float(_best_val.mean()) if hasattr(_best_val, "mean") else float(_best_val)
                )
                _ckpt_state["best_model_state_dict"] = best[1].state_dict()
            _save_latest()

            if _should_stop:
                break

        if best[1]:
            self.pred_model = deepcopy(best[1])

        ############################# Evaluate final model #############################
        # Document how well this trained model does
        self.logger.info("Benchmarking Model...")
        # Print final metrics
        datasets = [
            (X_train, Y_train, Y_train_aux, "train"),
            (X_val, Y_val, Y_val_aux, "val"),
            (X_test, Y_test, Y_test_aux, "test"),
        ]
        results = print_metrics(
            datasets,
            self.pred_model,
            problem,
            loss_fn,
            twostage_criterion,
            ptoSolver,
            "Final",
            self.logger,
            do_debug=do_debug,
            batch_size=self.args.batch_size,
            **self.model_args,
        )
        total_test_time = results["test"]["time"]
        eval_value = results["test"]["eval"]["value"]
        ############################ Save to file ############################
        # save logs — log_dir CSVs are written incrementally; bkup gets a full snapshot
        # (bkup_log_dir is timestamped and only covers this run's epochs)
        save_pd(train_logs, os.path.join(self.args.bkup_log_dir, "train_logs.csv"))
        save_pd(val_logs, os.path.join(self.args.bkup_log_dir, "val_logs.csv"))
        # save objectives
        np.save(
            os.path.join(self.args.log_dir, "results.npy"),
            [to_device(Objs_test_opt, "cpu"), to_device(eval_value, "cpu")],
        )
        np.save(
            os.path.join(self.args.bkup_log_dir, "results.npy"),
            [to_device(Objs_test_opt, "cpu"), to_device(eval_value, "cpu")],
        )
        # save solutions
        if do_debug:
            Z_test_opt_array = to_array(Z_test_opt)
            np.save(
                os.path.join(self.args.bkup_log_dir, "tensors", "solution.npy"),
                Z_test_opt_array,
            )
            torch.save(
                results["test"]["preds"].cpu().detach(),
                os.path.join(self.args.bkup_log_dir, "tensors", "preds.pt"),
            )

        ############################ Logging ############################
        avg_train_time = ndiv(total_train_time, (self.args.n_ptr_epochs + n_epochs))
        avg_test_time = total_test_time
        self.logger.info(
            f"[Random Obj]: {objs_rand.mean().item():.5f} "
            f"[Optimal Obj]: {Objs_test_opt.mean().item():.5f} "
            f"[{problem.get_eval_metric()}]: {eval_value.mean():.5f} "
            f"[avg Train Time]: {avg_train_time:.5f} "
            f"[avg Test Time]: {avg_test_time:.5f} "
        )
        self.logger.info(
            f"[{self.args.opt_model}]  {results['test']['objective'].mean():.5f}  {eval_value.mean():.5f}  "
            f"{avg_train_time:.5f}  {avg_test_time:.5f}"
        )
        return True
