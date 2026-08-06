import os
from copy import deepcopy
from typing import Dict, Optional

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer,
                               seed_everything)
from pytorch_lightning.loggers import LightningLoggerBase
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchmtlr import mtlr_survival

from src import utils
from src.postprocessors.adapter import SurvivalModelAdapter, TASK_NAMES
from src.postprocessors.utils import (PostprocessorConfig,
                                      get_postprocessor)
from src.utils.evaluation_metrics import (bootstrap_ood_metrics,
                                        mixed_ood_metric_row)


log = utils.get_logger(__name__)


def get_focus_task() -> Optional[str]:
    focus_task = os.environ.get("FOCUS_TASK", "").strip().upper()
    if focus_task in TASK_NAMES:
        return focus_task
    return None


def to_metric_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def log_focus_task_summary(trainer: Trainer) -> None:
    focus_task = get_focus_task()
    if not focus_task:
        return

    callback_metrics = trainer.callback_metrics
    test_ci = to_metric_float(callback_metrics.get(f"test/{focus_task}_CI"))
    id_ci = to_metric_float(callback_metrics.get(f"test/id/{focus_task}_CI"))
    ood_ci = to_metric_float(callback_metrics.get(f"test/ood/{focus_task}_CI"))

    log.info("=" * 80)
    log.info(f"FOCUS TASK SUMMARY: {focus_task}")
    if test_ci is not None:
        log.info(f"[FOCUS TASK: {focus_task}] Test CI: {test_ci:.4f}")
    if id_ci is not None:
        log.info(f"[FOCUS TASK: {focus_task}] ID-only CI: {id_ci:.4f}")
    if ood_ci is not None:
        log.info(f"[FOCUS TASK: {focus_task}] OOD-only CI: {ood_ci:.4f}")
    log.info("=" * 80)


class TaskSpecificWrapper(torch.nn.Module):
    """
    Wrapper that extracts logits from a specific task.

    This wrapper intercepts the adapter's forward calls and extracts
    only the logits from the specified task index.
    """

    def __init__(self, adapter, task_idx: int):
        super().__init__()
        self.adapter = adapter
        self.task_idx = task_idx

    def forward(self, data, return_feature=False, return_feature_list=False):
        """Extract logits for specific task."""
        # Get per-task logits
        logits_list, features = self.adapter(data, return_feature=True, return_per_task=True)

        # Extract the specific task logits
        task_logits = logits_list[self.task_idx]

        # Return in the expected format
        if return_feature and return_feature_list:
            return task_logits, [features]
        if return_feature:
            return task_logits, features
        if return_feature_list:
            return task_logits, [features]
        return task_logits

    def forward_threshold(self, data, percentile):
        """Extract logits for specific task with threshold (for ASH, ReAct, etc.)."""
        # Get per-task logits
        logits_list = self.adapter.forward_threshold(data, percentile, return_per_task=True)

        # Extract the specific task logits
        return logits_list[self.task_idx]

    def _get_features(self, data):
        """Delegate to adapter."""
        return self.adapter._get_features(data)

    def __getattr__(self, name):
        """Forward any other attributes to the adapter."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.adapter, name)


def test(config: DictConfig) -> None:
    """Evaluation entrypoint leveraging Hydra configuration."""

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    if not config.get("ckpt_path"):
        raise ValueError("ckpt_path is required. Pass ckpt_path=/absolute/path/to/model.ckpt")

    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(),
                                        config.ckpt_path)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        config.datamodule)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    if config.ckpt_path:
        log.info(f"Loading checkpoint from {config.ckpt_path}")
        checkpoint = torch.load(config.ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)

    logger_instances = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger_instances.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer,
                                               logger=logger_instances)

    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})

    log.info("Running Lightning test loop.")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=None)
    log_focus_task_summary(trainer)

    if config.get("survival_dist_metrics") and config.survival_dist_metrics.get("enable", False):
        log.info("Distribution-aware survival metrics enabled. Starting analysis...")
        run_distribution_metrics(config, model, datamodule)

    if config.get("survival_hazarddev") and config.survival_hazarddev.get("enable", False):
        log.info("DeepHit HazardDev metrics enabled. Starting analysis...")
        run_hazarddev_metrics(config, model, datamodule)

    if config.postprocessor and config.postprocessor.get("enable", False):
        log.info("Postprocessor evaluation enabled. Starting analysis...")
        # Check if multi-task evaluation is requested
        evaluate_per_task = config.postprocessor.get("evaluate_per_task", True)
        run_postprocessor_evaluation(config, model, datamodule, evaluate_per_task)


def _move_to_device(obj, device: torch.device):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_move_to_device(v, device) for v in obj]
        return type(obj)(converted)
    return torch.as_tensor(obj, device=device)


def make_eval_loader(datamodule: LightningDataModule, dataset):
    if hasattr(dataset, "mini_batch_assignments"):
        dataset.mini_batch_assignments = None
    return DataLoader(
        dataset,
        batch_size=getattr(datamodule, "batch_size", 32),
        shuffle=False,
        num_workers=getattr(datamodule, "num_workers", 0),
        pin_memory=getattr(datamodule, "pin_memory", False),
    )


def time_grid_for_dataset(dataset, task_idx: int, n_time_bins: int):
    attr = f"time_bins{task_idx + 1}"
    source = f"{attr}"
    grid = getattr(dataset, attr, None)
    if grid is not None:
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().numpy()
        else:
            grid = np.asarray(grid)
        grid = grid.astype(float)
    if grid is None or len(grid) != n_time_bins:
        source = "fallback_arange"
        grid = np.arange(1, n_time_bins + 1, dtype=float)
    order = np.argsort(grid)
    grid = grid[order]
    for i in range(1, len(grid)):
        if grid[i] <= grid[i - 1]:
            grid[i] = grid[i - 1] + 1e-6
    return grid, source, order


def collect_distribution_predictions(adapter, dataloader, device: torch.device) -> Dict[str, Dict[str, np.ndarray]]:
    collected = {
        task: {"surv": [], "time": [], "event": []}
        for task in TASK_NAMES
    }
    adapter.eval()
    with torch.no_grad():
        for batch in dataloader:
            (sample, clin_var), _, labels = batch
            sample = _move_to_device(sample, device)
            clin_var = _move_to_device(clin_var, device)
            logits_list = adapter((sample, clin_var), return_per_task=True)
            is_deephit = getattr(adapter, "is_deephit", False)
            for task_idx, task in enumerate(TASK_NAMES[:len(logits_list)]):
                if is_deephit:
                    pmf = torch.softmax(logits_list[task_idx], dim=1)
                    surv = (1.0 - pmf.cumsum(dim=1)).clamp(min=0.0).detach().cpu().numpy()
                else:
                    surv = mtlr_survival(logits_list[task_idx]).detach().cpu().numpy()
                collected[task]["surv"].append(surv)
                collected[task]["time"].append(labels[f"time{task_idx + 1}"].detach().cpu().numpy())
                collected[task]["event"].append(labels[f"event{task_idx + 1}"].detach().cpu().numpy())

    out = {}
    for task, values in collected.items():
        if not values["surv"]:
            continue
        out[task] = {
            "surv": np.concatenate(values["surv"], axis=0),
            "time": np.concatenate(values["time"], axis=0).astype(float),
            "event": np.concatenate(values["event"], axis=0).astype(bool),
        }
    return out


def compute_pycox_metrics(surv: np.ndarray, durations: np.ndarray, events: np.ndarray, time_grid: np.ndarray):
    from pycox.evaluation import EvalSurv

    surv_df = pd.DataFrame(surv.T, index=time_grid)
    evaluator = EvalSurv(surv_df, durations, events.astype(bool), censor_surv="km")
    return {
        "ibs": float(evaluator.integrated_brier_score(time_grid)),
        "antolini_cindex": float(evaluator.concordance_td(method="adj_antolini")),
    }


def distribution_metric_row(task: str, domain: str, values: Dict[str, np.ndarray], dataset, task_idx: int) -> Dict[str, object]:
    surv = values["surv"]
    durations = values["time"]
    events = values["event"]
    time_grid, grid_source, order = time_grid_for_dataset(dataset, task_idx, surv.shape[1])
    surv = surv[:, order]

    row = {
        "task": task,
        "domain": domain,
        "ibs": "",
        "antolini_cindex": "",
        "harrell_cindex": "",
        "n": int(len(durations)),
        "n_events": int(events.sum()),
        "censoring_rate": float(1.0 - events.mean()) if len(events) else "",
        "time_grid_source": grid_source,
        "status": "SUCCESS",
        "notes": "",
    }
    try:
        row.update(compute_pycox_metrics(surv, durations, events, time_grid))
    except Exception as exc:
        row["status"] = "PYCOX_FAILED"
        row["notes"] = str(exc)
    try:
        row["harrell_cindex"] = float(concordance_index(durations, surv.sum(axis=1), event_observed=events))
    except Exception as exc:
        row["status"] = "HARRELL_FAILED" if row["status"] == "SUCCESS" else row["status"]
        row["notes"] = "; ".join(filter(None, [row["notes"], f"harrell: {exc}"]))
    return row


def run_distribution_metrics(config: DictConfig,
                             model: LightningModule,
                             datamodule: LightningDataModule) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    adapter = SurvivalModelAdapter(model).to(device)
    adapter.eval()

    datamodule.setup(stage="test")
    if getattr(datamodule, "_id_dataset", None) is None or getattr(datamodule, "_ood_dataset", None) is None:
        raise RuntimeError("Distribution metrics require RadCureMixedDataModule ID and OOD datasets.")

    id_dataset = datamodule._id_dataset
    ood_dataset = datamodule._ood_dataset
    id_predictions = collect_distribution_predictions(adapter, make_eval_loader(datamodule, id_dataset), device)
    ood_predictions = collect_distribution_predictions(adapter, make_eval_loader(datamodule, ood_dataset), device)

    rows = []
    for task_idx, task in enumerate(TASK_NAMES):
        if task not in id_predictions or task not in ood_predictions:
            continue
        rows.append(distribution_metric_row(task, "ID", id_predictions[task], id_dataset, task_idx))
        rows.append(distribution_metric_row(task, "OOD", ood_predictions[task], id_dataset, task_idx))
        all_values = {
            "surv": np.concatenate([id_predictions[task]["surv"], ood_predictions[task]["surv"]], axis=0),
            "time": np.concatenate([id_predictions[task]["time"], ood_predictions[task]["time"]], axis=0),
            "event": np.concatenate([id_predictions[task]["event"], ood_predictions[task]["event"]], axis=0),
        }
        rows.append(distribution_metric_row(task, "ALL", all_values, id_dataset, task_idx))

    output_path = config.survival_dist_metrics.get("output") or "Test_DistMetrics.csv"
    output_path = hydra.utils.to_absolute_path(output_path) if not os.path.isabs(output_path) else output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info("Saved distribution-aware survival metrics to %s", output_path)


def deephit_hazard_from_logits(logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Convert DeepHit time-bin logits into a discrete hazard curve.

    pmf = softmax(logits); S(t_i) = 1 - cumsum(pmf); h(t_i) = pmf(t_i) / S(t_{i-1}).
    """
    pmf = torch.softmax(logits, dim=1)
    surv = 1.0 - pmf.cumsum(dim=1)
    surv_prev = torch.cat([torch.ones_like(surv[:, :1]), surv[:, :-1]], dim=1)
    return pmf / surv_prev.clamp(min=eps)


def collect_deephit_hazard(adapter, dataloader, device: torch.device) -> np.ndarray:
    hazards = []
    adapter.eval()
    with torch.no_grad():
        for batch in dataloader:
            (sample, clin_var), _, _ = batch
            sample = _move_to_device(sample, device)
            clin_var = _move_to_device(clin_var, device)
            logits_list = adapter((sample, clin_var), return_per_task=True)
            hazard = deephit_hazard_from_logits(logits_list[0]).detach().cpu().numpy()
            hazards.append(hazard)
    if not hazards:
        return np.empty((0, 0), dtype=float)
    return np.concatenate(hazards, axis=0)


def run_hazarddev_metrics(config: DictConfig,
                          model: LightningModule,
                          datamodule: LightningDataModule) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    adapter = SurvivalModelAdapter(model).to(device)
    adapter.eval()
    if not getattr(adapter, "is_deephit", False):
        raise RuntimeError("survival_hazarddev currently supports DeepHit heads only.")

    mode = str(config.survival_hazarddev.get("mode", "none"))
    ref_dir = config.survival_hazarddev.get("dir")
    if ref_dir and not os.path.isabs(ref_dir):
        ref_dir = hydra.utils.to_absolute_path(ref_dir)
    if not ref_dir:
        raise RuntimeError("survival_hazarddev.dir must be set for compute/use modes.")
    ref_path = os.path.join(ref_dir, "deephit_hazard_mean_OS.npy")

    datamodule.setup(stage="test")
    if getattr(datamodule, "_id_dataset", None) is None:
        raise RuntimeError("HazardDev metrics require RadCureMixedDataModule ID dataset.")

    if mode == "compute":
        id_hazard = collect_deephit_hazard(
            adapter, make_eval_loader(datamodule, datamodule._id_dataset), device
        )
        if id_hazard.size == 0:
            raise RuntimeError("No ID samples collected for HazardDev reference computation.")
        os.makedirs(ref_dir, exist_ok=True)
        reference = id_hazard.mean(axis=0)
        np.save(ref_path, reference)
        log.info("Saved DeepHit hazard reference (shape=%s, n=%d) to %s",
                 reference.shape, id_hazard.shape[0], ref_path)
        return

    if mode != "use":
        log.warning("survival_hazarddev.mode=%s is not 'compute' or 'use'; skipping.", mode)
        return

    if getattr(datamodule, "_ood_dataset", None) is None:
        raise RuntimeError("HazardDev use-mode requires RadCureMixedDataModule OOD dataset.")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f"HazardDev reference not found: {ref_path}. Run survival_hazarddev.mode=compute first."
        )

    reference = np.load(ref_path)
    id_hazard = collect_deephit_hazard(
        adapter, make_eval_loader(datamodule, datamodule._id_dataset), device
    )
    ood_hazard = collect_deephit_hazard(
        adapter, make_eval_loader(datamodule, datamodule._ood_dataset), device
    )
    if id_hazard.size == 0 or ood_hazard.size == 0:
        raise RuntimeError("HazardDev use-mode requires non-empty ID and OOD predictions.")

    hazard = np.concatenate([id_hazard, ood_hazard], axis=0)
    diff = hazard - reference[None, :]
    n_id = id_hazard.shape[0]
    n_ood = ood_hazard.shape[0]
    binary_targets = np.concatenate([np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)])
    id_mask = binary_targets == 0
    ood_mask = binary_targets == 1

    score_defs = {
        "HazardDev_OS": diff.mean(axis=1),
        "HazardDevL1_OS": np.abs(diff).mean(axis=1),
        "HazardDevL2_OS": np.sqrt(np.square(diff).mean(axis=1)),
    }

    n_bootstrap = int(config.survival_hazarddev.get("n_bootstrap", 1000))
    seed = int(config.survival_hazarddev.get("seed", 786))

    rows = []
    for score_name, values in score_defs.items():
        row = mixed_ood_metric_row(
            score_name,
            values,
            binary_targets,
            id_mask,
            ood_mask,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        row["task"] = "OS"
        row["reference_path"] = ref_path
        rows.append(row)

    output_path = config.survival_hazarddev.get("output") or "Test_HazardDev.csv"
    if not os.path.isabs(output_path):
        output_path = hydra.utils.to_absolute_path(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info("Saved DeepHit HazardDev metrics (n_id=%d, n_ood=%d) to %s", n_id, n_ood, output_path)


def run_postprocessor_evaluation(config: DictConfig,
                                 model: LightningModule,
                                 datamodule: LightningDataModule,
                                 evaluate_per_task: bool = True) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    adapter = SurvivalModelAdapter(model).to(device)
    adapter.eval()

    pp_config = load_postprocessor_config(config)
    postprocessor = get_postprocessor(pp_config)

    # Wrap adapter with method-specific wrappers (following OpenMIBOOD pattern)
    postprocessor_name = config.postprocessor.name
    if postprocessor_name == 'ash':
        from src.postprocessors.ash_wrapper import ASHWrapper
        adapter = ASHWrapper(adapter)
        log.info("Wrapped adapter with ASHWrapper")
    elif postprocessor_name == 'scale':
        from src.postprocessors.scale_wrapper import ScaleWrapper
        adapter = ScaleWrapper(adapter)
        log.info("Wrapped adapter with ScaleWrapper")
    # Add other wrappers here as needed (react, etc.)

    # Setup datamodule for test stage
    datamodule.setup(stage="test")

    def make_loader(dataset):
        if hasattr(dataset, "mini_batch_assignments"):
            dataset.mini_batch_assignments = None
        return DataLoader(
            dataset,
            batch_size=getattr(datamodule, "batch_size", 32),
            shuffle=False,
            num_workers=getattr(datamodule, "num_workers", 0),
            pin_memory=getattr(datamodule, "pin_memory", False),
        )

    id_loader = None
    if getattr(datamodule, "_id_dataset", None) is not None:
        id_loader = make_loader(datamodule._id_dataset)
    else:
        id_loader = datamodule.test_dataloader()
        log.warning("Postprocessor ID loader fallback: using mixed test dataloader.")

    runtime_dir = os.getcwd()
    focus_task = get_focus_task()

    ood_loader = None
    if getattr(datamodule, "_ood_dataset", None) is not None:
        ood_loader = make_loader(datamodule._ood_dataset)

    # Some postprocessors estimate feature statistics before scoring.
    # For the mixed test datamodule we only have ID/OOD test datasets available,
    # so reuse the ID loader as the reference loader across train/val/test keys.
    if hasattr(postprocessor, "setup"):
        id_loader_dict = {"train": id_loader, "val": id_loader, "test": id_loader}
        ood_loader_dict = {"train": ood_loader, "val": ood_loader, "test": ood_loader}
        postprocessor.setup(adapter, id_loader_dict, ood_loader_dict)
        log.info("Postprocessor setup completed.")

    if evaluate_per_task:
        # Evaluate each task independently
        log.info("=" * 80)
        log.info("MULTI-TASK OOD DETECTION - Evaluating each task independently")
        log.info("=" * 80)

        num_tasks = len(adapter.time_bins)
        task_names = TASK_NAMES[:num_tasks]

        for task_idx, task_name in enumerate(task_names):
            log.info(f"\n{'='*80}")
            log.info(f"Task {task_idx + 1}/{num_tasks}: {task_name}")
            if task_name == focus_task:
                log.info(f"[FOCUS TASK] {task_name} is the primary task for this checkpoint.")
            log.info(f"{'='*80}")

            # Collect scores for this specific task
            id_results = collect_scores(adapter, postprocessor, id_loader, device, task_idx=task_idx)

            ood_results = None
            if ood_loader is not None:
                ood_results = collect_scores(adapter, postprocessor,
                                             ood_loader,
                                             device, task_idx=task_idx)
            else:
                ood_path = config.postprocessor.get("ood_data_path")
                if ood_path:
                    ood_cfg = deepcopy(config)
                    ood_cfg.datamodule.data_dir = ood_path
                    ood_datamodule: LightningDataModule = hydra.utils.instantiate(
                        ood_cfg.datamodule)
                    ood_datamodule.setup(stage="test")
                    ood_loader = ood_datamodule.test_dataloader()
                    ood_results = collect_scores(adapter, postprocessor, ood_loader,
                                                 device, task_idx=task_idx)

            # Save results for this task
            save_and_report(
                config,
                id_results,
                ood_results,
                runtime_dir,
                task_name=task_name,
                focus_task=focus_task,
            )

        log.info("\n" + "=" * 80)
        log.info("Multi-task evaluation completed!")
        log.info("=" * 80)

    else:
        # Original behavior: evaluate all tasks together (concatenated logits)
        log.info("Evaluating all tasks together (concatenated logits)")

        id_results = collect_scores(adapter, postprocessor, id_loader, device)

        ood_results = None
        if ood_loader is not None:
            ood_results = collect_scores(adapter, postprocessor,
                                         ood_loader,
                                         device)
        else:
            ood_path = config.postprocessor.get("ood_data_path")
            if ood_path:
                log.info(f"Preparing OOD datamodule with data at {ood_path}")
                ood_cfg = deepcopy(config)
                ood_cfg.datamodule.data_dir = ood_path
                ood_datamodule: LightningDataModule = hydra.utils.instantiate(
                    ood_cfg.datamodule)
                ood_datamodule.setup(stage="test")
                ood_loader = ood_datamodule.test_dataloader()
                ood_results = collect_scores(adapter, postprocessor, ood_loader,
                                             device)

        save_and_report(config, id_results, ood_results, runtime_dir, focus_task=focus_task)


def load_postprocessor_config(config: DictConfig) -> PostprocessorConfig:
    if config.postprocessor.get("config_path"):
        cfg_path = hydra.utils.to_absolute_path(config.postprocessor.config_path)
    else:
        cfg_path = os.path.join(
            hydra.utils.to_absolute_path("configs/postprocessors"),
            f"{config.postprocessor.name}.yml")
    cfg_dict = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if "postprocessor" not in cfg_dict:
        cfg_dict["postprocessor"] = {"name": config.postprocessor.name}
    else:
        cfg_dict["postprocessor"].setdefault("name",
                                             config.postprocessor.name)
    cfg_dict.setdefault("dataset", {})
    cfg_dict["dataset"].setdefault("name",
                                   config.postprocessor.get(
                                       "dataset_name",
                                       "cancer_survival"))
    cfg_dict.setdefault("exp_name", config.name)
    cfg_dict.setdefault("output_dir", config.output_dir)
    return PostprocessorConfig.from_dict(cfg_dict)


def collect_scores(net: torch.nn.Module,
                   postprocessor,
                   dataloader,
                   device: torch.device,
                   task_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Collect OOD scores from the network.

    Args:
        net: The network (adapter)
        postprocessor: The postprocessor for OOD detection
        dataloader: Data loader
        device: Device to use
        task_idx: If specified, only use logits from this task (0=OS, 1=LFFS, 2=RFFS, 3=DFFS).
                 If None, use all tasks (concatenated logits).

    Returns:
        Dictionary with 'pred' and 'score' tensors
    """
    preds = []
    scores = []

    def move_to_device(obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [move_to_device(v) for v in obj]
            return type(obj)(converted)
        return torch.as_tensor(obj, device=device)

    needs_grad = getattr(postprocessor, "needs_grad", False)
    context = torch.enable_grad if needs_grad else torch.no_grad

    # Create a wrapper to handle task-specific extraction
    if task_idx is not None:
        net = TaskSpecificWrapper(net, task_idx)

    with context():
        for batch in dataloader:
            (sample, clin_var), _, _ = batch
            sample = move_to_device(sample)
            clin_var = move_to_device(clin_var)
            data = (sample, clin_var)
            batch_pred, batch_score = postprocessor.postprocess(net, data)
            if needs_grad:
                net.zero_grad(set_to_none=True)
            preds.append(batch_pred.detach().cpu())
            scores.append(batch_score.detach().cpu())

    return {
        "pred": torch.cat(preds),
        "score": torch.cat(scores),
    }


def save_and_report(config: DictConfig,
                    id_results: Dict[str, torch.Tensor],
                    ood_results: Optional[Dict[str, torch.Tensor]],
                    base_dir: str,
                    task_name: Optional[str] = None,
                    focus_task: Optional[str] = None) -> None:
    """
    Save and report OOD detection results.

    Args:
        config: Configuration
        id_results: ID data results
        ood_results: OOD data results
        base_dir: Base directory for saving
        task_name: If specified, append task name to filenames (e.g., "OS", "LFFS")
    """
    os.makedirs(base_dir, exist_ok=True)

    # Prepare filename suffix
    suffix = f"_{task_name}" if task_name else ""
    method_name = config.postprocessor.name

    id_scores = id_results["score"].numpy()
    id_path = os.path.join(base_dir,
                           f"{method_name}{suffix}_scores_id.csv")
    pd.DataFrame({"score": id_scores}).to_csv(id_path, index=False)
    log.info(f"Saved ID scores to {id_path}")

    if ood_results is None:
        return

    ood_scores = ood_results["score"].numpy()
    ood_path = os.path.join(base_dir,
                            f"{method_name}{suffix}_scores_ood.csv")
    pd.DataFrame({"score": ood_scores}).to_csv(ood_path, index=False)
    log.info(f"Saved OOD scores to {ood_path}")

    labels = np.concatenate(
        [np.ones_like(id_scores), np.zeros_like(ood_scores)])
    scores = np.concatenate([id_scores, ood_scores])
    auroc = roc_auc_score(labels, scores)
    aupr = average_precision_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    if np.any(tpr >= 0.95):
        fpr95 = float(np.interp(0.95, tpr, fpr))
    else:
        fpr95 = 1.0

    bootstrap_cfg = config.postprocessor.get("bootstrap") or {}
    bootstrap_enabled = bool(bootstrap_cfg.get("enabled", False))
    bootstrap_metrics = None
    if bootstrap_enabled:
        bootstrap_metrics = bootstrap_ood_metrics(
            id_scores,
            ood_scores,
            n_bootstrap=int(bootstrap_cfg.get("n_bootstrap", 1000)),
            ci_level=float(bootstrap_cfg.get("ci_level", 0.95)),
            seed=int(bootstrap_cfg.get("seed", 786)),
            ood_positive=False,
        )

    # Print metrics with task name if available
    is_focus_task = bool(task_name and focus_task and task_name == focus_task)
    if is_focus_task:
        task_label = f"[FOCUS TASK: {task_name}] "
    else:
        task_label = f"[{task_name}] " if task_name else ""
    if bootstrap_metrics is not None:
        log.info(
            f"{task_label}OOD metrics — AUROC: {auroc:.4f} "
            f"[{bootstrap_metrics['auroc_ci_low']:.4f}, {bootstrap_metrics['auroc_ci_high']:.4f}], "
            f"AUPR: {aupr:.4f} "
            f"[{bootstrap_metrics['aupr_ci_low']:.4f}, {bootstrap_metrics['aupr_ci_high']:.4f}], "
            f"FPR95: {fpr95:.4f} "
            f"[{bootstrap_metrics['fpr95_ci_low']:.4f}, {bootstrap_metrics['fpr95_ci_high']:.4f}]")
    else:
        log.info(
            f"{task_label}OOD metrics — AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr95:.4f}")

    metrics = {
        "Task": task_name if task_name else "All",
        "Method": method_name,
        "AUROC": auroc,
        "AUPR": aupr,
        "FPR95": fpr95,
    }

    if bootstrap_metrics is not None:
        metrics.update({
            "AUROC_CI_LOW": bootstrap_metrics["auroc_ci_low"],
            "AUROC_CI_HIGH": bootstrap_metrics["auroc_ci_high"],
            "AUROC_BOOT_STD": bootstrap_metrics["auroc_boot_std"],
            "AUPR_CI_LOW": bootstrap_metrics["aupr_ci_low"],
            "AUPR_CI_HIGH": bootstrap_metrics["aupr_ci_high"],
            "AUPR_BOOT_STD": bootstrap_metrics["aupr_boot_std"],
            "FPR95_CI_LOW": bootstrap_metrics["fpr95_ci_low"],
            "FPR95_CI_HIGH": bootstrap_metrics["fpr95_ci_high"],
            "FPR95_BOOT_STD": bootstrap_metrics["fpr95_boot_std"],
            "N_ID": bootstrap_metrics["n_id"],
            "N_OOD": bootstrap_metrics["n_ood"],
            "N_BOOTSTRAP": bootstrap_metrics["n_bootstrap"],
            "N_BOOTSTRAP_VALID": bootstrap_metrics["n_bootstrap_valid"],
            "POSITIVE_LABEL": bootstrap_metrics["positive_label"],
            "SCORE_ORIENTATION": bootstrap_metrics["score_orientation"],
        })

    metrics_path = config.postprocessor.get("metrics_output")
    if metrics_path:
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        # Append task name to metrics path if specified
        if task_name:
            base, ext = os.path.splitext(metrics_path)
            metrics_path = f"{base}_{task_name}{ext}"
    else:
        metrics_path = os.path.join(
            base_dir,
            f"{method_name}{suffix}_metrics.csv")

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    log.info(f"Saved OOD metrics to {metrics_path}")
