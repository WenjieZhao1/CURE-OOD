import os
from pathlib import Path
from typing import Any, List
import pandas as pd

import torch
from torch import nn
from torchmtlr import MTLR

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn as nn

from hydra.utils import to_absolute_path
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torchmtlr import mtlr_hazard, mtlr_neg_log_likelihood, mtlr_survival, mtlr_risk
import numpy as np
from scipy.spatial import cKDTree

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold

from src.models.components.net_cli import Cli_MTLR 

from src.models.components.net_vit_img import UNETR as UNETR_Img
from src.models.components.net_cnn_img import CNN_MTLR as CNN_MTLR_Img
from src.models.components.net_cnn_img_v2 import CNN_MTLR as CNN_MTLR_Img_V2
from src.models.components.net_swin_img import Swin as Swin_Img

from src.models.components.net_vit import UNETR
from src.models.components.net_cnvit import UNETR as CNViT
from src.models.components.net_cnn import CNN_MTLR
from src.models.components.net_swin import Swin
from src.models.components.net_resnet import ResNet_MTLR
from src import utils
from src.utils.evaluation_metrics import (cindex_metric_rows,
                                        hazard_l1_l2_entries,
                                        mixed_ood_metric_row,
                                        extended_score_family_from_logits)

log = utils.get_logger(__name__)

class DEEP_MTLR(LightningModule):
    """Lightning module for task-specific survival prediction and OOD evaluation."""

    def __init__(
        self,
       
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()


        if self.hparams['model'] == 'UNETR':
            self.model = UNETR( hparams = self.hparams,
                                in_channels= self.hparams['in_channels'],
                                out_channels= self.hparams['out_channels'],
                                img_size = (self.hparams["patch_z"], self.hparams["patch_xy"], self.hparams["patch_xy"]),
                                feature_size = self.hparams['patch_size'],
                                hidden_size = self.hparams['hidden_size'],
                                mlp_dim = self.hparams['mlp_dim'],
                                num_heads = self.hparams['num_heads'],
                                pos_embed = "conv",
                                norm_name = "instance",
                                conv_block = True,
                                res_block = True,
                                dropout_rate = 0,
                                spatial_dims = 3,
                            )
        elif self.hparams['model'] == 'CNViT':
            self.model = CNViT( hparams = self.hparams,
                                in_channels= self.hparams['in_channels'],
                                out_channels= self.hparams['out_channels'],
                                img_size = (self.hparams["patch_z"], self.hparams["patch_xy"], self.hparams["patch_xy"]),
                                feature_size = self.hparams['patch_size'],
                                hidden_size = self.hparams['hidden_size'],
                                mlp_dim = self.hparams['mlp_dim'],
                                num_heads = self.hparams['num_heads'],
                                pos_embed = "conv",
                                norm_name = "instance",
                                conv_block = True,
                                res_block = True,
                                dropout_rate = 0,
                                spatial_dims = 3,
                            )
        
        elif self.hparams['model'] == 'Cli':
            self.model = Cli_MTLR(hparams = self.hparams)
        elif self.hparams['model'] == 'CNN':
            self.model = CNN_MTLR(hparams = self.hparams)
        elif self.hparams['model'] == 'Swin':
            self.model = Swin(hparams = self.hparams, embed_dim=96, patch_size=self.hparams['patch_size'], in_chans=self.hparams['in_channels'], img_size=(self.hparams["patch_z"], self.hparams["patch_xy"], self.hparams["patch_xy"]))
        elif self.hparams['model'] =='ResNet':
            self.model = ResNet_MTLR(hparams = self.hparams)
        elif self.hparams['model'] == 'UNETR_Img':
            self.model = UNETR_Img( hparams = self.hparams,
                                in_channels= self.hparams['in_channels'],
                                out_channels= self.hparams['out_channels'],
                                img_size = (self.hparams["patch_z"], self.hparams["patch_xy"], self.hparams["patch_xy"]),
                                feature_size = self.hparams['patch_size'],
                                hidden_size = self.hparams['hidden_size'],
                                mlp_dim = self.hparams['mlp_dim'],
                                num_heads = self.hparams['num_heads'],
                                pos_embed = "conv",
                                norm_name = "instance",
                                conv_block = True,
                                res_block = True,
                                dropout_rate = 0,
                                spatial_dims = 3,
                            )
        
        elif self.hparams['model'] == 'CNN_Img':
            self.model = CNN_MTLR_Img(hparams = self.hparams)
        elif self.hparams['model'] == 'CNN_Img_V2':
            self.model = CNN_MTLR_Img_V2(hparams = self.hparams)
        elif self.hparams['model'] == 'Swin_Img':
            self.model = Swin_Img(hparams = self.hparams, embed_dim=96, patch_size=self.hparams['patch_size'], in_chans=self.hparams['in_channels'], img_size=(self.hparams["patch_z"], self.hparams["patch_xy"], self.hparams["patch_xy"]))



        else:
            raise ValueError(f"Unsupported model architecture: {self.hparams['model']}")



    def forward(self, x: torch.Tensor):
        return self.model(x)

    @staticmethod
    def _safe_concordance_index(event_times, pred_prob, event_observed):
        pred_risk = mtlr_risk(pred_prob).detach().cpu().numpy()
        event_times = event_times.detach().cpu().numpy()
        event_observed = event_observed.detach().cpu().numpy()

        finite_mask = np.isfinite(pred_risk) & np.isfinite(event_times) & np.isfinite(event_observed)
        if finite_mask.sum() < 2:
            return np.nan, pred_risk

        event_times = event_times[finite_mask]
        pred_risk = pred_risk[finite_mask]
        event_observed = event_observed[finite_mask]

        if np.unique(event_observed).size < 2 or np.unique(event_times).size < 2:
            return np.nan, pred_risk

        try:
            ci_event = concordance_index(event_times, -pred_risk, event_observed=event_observed)
        except ValueError:
            ci_event = np.nan

        return ci_event, pred_risk

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

            nn.init.constant_(m.bias, -1.5214691)



    def step(self, batch: Any):
        (sample, clin_var), (y1,y2,y3,y4), labels = batch
        model_outputs = self.forward((sample, clin_var))
        logits1, logits2, logits3, logits4 = model_outputs[:4]
        loss_mtlr = mtlr_neg_log_likelihood(logits4, y4.float(), self.model, self.hparams['C1'], average=True)
        loss = loss_mtlr
        

        return loss, logits1, logits2,logits3, logits4, y1,y2,y3,y4, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds1, preds2, preds3, preds4, y1,y2,y3,y4, labels = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds1": preds1, "preds2": preds2, "preds3": preds3, "preds4": preds4,"labels": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        loss        = torch.stack([x["loss"] for x in outputs]).mean()

        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        ci_event1, pred_risk1 = self._safe_concordance_index(true_time1, pred_prob1, true_event1)
        
        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu() 
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        ci_event2, pred_risk2 = self._safe_concordance_index(true_time2, pred_prob2, true_event2)
        
        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu() 
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        ci_event3, pred_risk3 = self._safe_concordance_index(true_time3, pred_prob3, true_event3)
        
        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu() 
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        ci_event4, pred_risk4 = self._safe_concordance_index(true_time4, pred_prob4, true_event4)

        log = {"train/OS_CI": ci_event1,
               "train/LFFS_CI": ci_event2,
               "train/RFFS_CI": ci_event3,
               "train/DFFS_CI": ci_event4,
               "train/AVG_CI": np.nanmean([ci_event1, ci_event2, ci_event3, ci_event4]),
               }  
        self.log_dict(log)

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds1, preds2, preds3, preds4, y1,y2,y3,y4, labels = self.step(batch)
        return {"loss": loss, "preds1": preds1, "preds2": preds2, "preds3": preds3, "preds4": preds4,"labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        loss        = torch.stack([x["loss"] for x in outputs]).mean()

        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        ci_event1, pred_risk1 = self._safe_concordance_index(true_time1, pred_prob1, true_event1)
        
        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu() 
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        ci_event2, pred_risk2 = self._safe_concordance_index(true_time2, pred_prob2, true_event2)
        
        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu() 
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        ci_event3, pred_risk3 = self._safe_concordance_index(true_time3, pred_prob3, true_event3)
        
        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu() 
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        ci_event4, pred_risk4 = self._safe_concordance_index(true_time4, pred_prob4, true_event4)

        log = {"val/loss": loss,
               "val/OS_CI": ci_event1,
               "val/LFFS_CI": ci_event2,
               "val/RFFS_CI": ci_event3,
               "val/DFFS_CI": ci_event4,
               "val/AVG_CI": np.nanmean([ci_event1, ci_event2, ci_event3, ci_event4]),
               }  
        self.log_dict(log)


        PatientID  = [x['labels']['ID'] for x in outputs] 
        PatientID = sum(PatientID, []) #inefficient way to flatten a list


        results = pd.DataFrame({'ID':PatientID, 'OS_risk':pred_risk1, 'OS':true_time1,'Death':true_event1,\
                                'LFFS_risk':pred_risk2, 'LFFS':true_time2,'LF':true_event2,\
                                'RFFS_risk':pred_risk3, 'RFFS':true_time3,'RF':true_event3,\
                                'DFFS_risk':pred_risk4, 'DFFS':true_time4,'DF':true_event4})
        results.to_csv('Val_Predictions.csv')

        return {"loss": loss, "OS-CI": ci_event1, "LFFS-CI": ci_event2, "RFFS-CI": ci_event3, "DFFS-CI": ci_event4}

    def test_step(self, batch: Any, batch_idx: int):
        (sample, clin_var), (y1, y2, y3, y4), labels = batch

        if hasattr(self.model, "get_mtlr_features") and all(
            hasattr(self.model, name) for name in ("mtlr1", "mtlr2", "mtlr3", "mtlr4")
        ):
            features = self.model.get_mtlr_features((sample, clin_var))
            preds1, seprate_logits1 = self.model.mtlr1(features)
            preds2, seprate_logits2 = self.model.mtlr2(features)
            preds3, seprate_logits3 = self.model.mtlr3(features)
            preds4, seprate_logits4 = self.model.mtlr4(features)
            loss = mtlr_neg_log_likelihood(preds4, y4.float(), self.model, self.hparams['C1'], average=True)
        else:
            loss, preds1, preds2, preds3, preds4, y1, y2, y3, y4, labels = self.step(batch)
            features = None
            seprate_logits1 = preds1
            seprate_logits2 = preds2
            seprate_logits3 = preds3
            seprate_logits4 = preds4

        return {
            "loss": loss,
            "preds1": preds1,
            "preds2": preds2,
            "preds3": preds3,
            "preds4": preds4,
            "labels": labels,
            "features": features,
            "seprate_logits1": seprate_logits1,
            "seprate_logits2": seprate_logits2,
            "seprate_logits3": seprate_logits3,
            "seprate_logits4": seprate_logits4,
        }

    def compute_mcd_score_with_features(self, features, mtlr_layer):
        eps = 1e-9
        z = torch.matmul(features, mtlr_layer.mtlr_weight) + mtlr_layer.mtlr_bias
        mtlr_logits = torch.matmul(z, mtlr_layer.G)
        p = torch.softmax(mtlr_logits, dim=1)

        q = torch.sigmoid(z)
        num_intervals = z.shape[1] + 1
        ptilde_unnorm = torch.zeros(features.shape[0], num_intervals, device=z.device)

        for k in range(num_intervals):
            prefix = torch.ones(features.shape[0], device=z.device) if k == 0 else torch.prod(1 - q[:, :k], dim=1)
            suffix = torch.ones(features.shape[0], device=z.device) if k >= z.shape[1] else torch.prod(q[:, k:], dim=1)
            ptilde_unnorm[:, k] = prefix * suffix

        ptilde = ptilde_unnorm / (torch.sum(ptilde_unnorm, dim=1, keepdim=True) + eps)
        mixture = 0.5 * (p + ptilde) + eps

        def kl_divergence(lhs, rhs):
            return torch.sum(lhs * torch.log((lhs + eps) / (rhs + eps)), dim=1)

        return 0.5 * (kl_divergence(p, mixture) + kl_divergence(ptilde, mixture))

    def test_epoch_end(self, outputs: List[Any]):
        loss        = torch.stack([x["loss"] for x in outputs]).mean()

        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        ci_event1, pred_risk1 = self._safe_concordance_index(true_time1, pred_prob1, true_event1)
        pred_hazard1 = mtlr_hazard(pred_prob1).detach().numpy()
        
        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu() 
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        ci_event2, pred_risk2 = self._safe_concordance_index(true_time2, pred_prob2, true_event2)
        pred_hazard2 = mtlr_hazard(pred_prob2).detach().numpy()
        
        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu() 
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        ci_event3, pred_risk3 = self._safe_concordance_index(true_time3, pred_prob3, true_event3)
        pred_hazard3 = mtlr_hazard(pred_prob3).detach().numpy()
        
        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu() 
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        ci_event4, pred_risk4 = self._safe_concordance_index(true_time4, pred_prob4, true_event4)
        pred_hazard4 = mtlr_hazard(pred_prob4).detach().numpy()

        metrics_log = {"test/loss": loss,
                       "test/OS_CI": ci_event1,
                       "test/LFFS_CI": ci_event2,
                       "test/RFFS_CI": ci_event3,
                       "test/DFFS_CI": ci_event4,
                       "test/AVG_CI": np.nanmean([ci_event1, ci_event2, ci_event3, ci_event4]),
                       }
        self.log_dict(metrics_log)


        PatientID  = [x['labels']['ID'] for x in outputs] 
        PatientID = sum(PatientID, []) #inefficient way to flatten a list


        results = pd.DataFrame({'ID':PatientID, 'OS_risk':pred_risk1, 'OS':true_time1,'Death':true_event1,\
                                'LFFS_risk':pred_risk2, 'LFFS':true_time2,'LF':true_event2,\
                                'RFFS_risk':pred_risk3, 'RFFS':true_time3,'RF':true_event3,\
                                'DFFS_risk':pred_risk4, 'DFFS':true_time4,'DF':true_event4})
        results.to_csv('Test_Predictions.csv')

        domain_tensor = None
        if outputs and isinstance(outputs[0]["labels"], dict) and "domain" in outputs[0]["labels"]:
            try:
                domain_tensor = torch.cat([x["labels"]["domain"] for x in outputs]).cpu()
            except Exception as exc:
                log.warning("Failed to aggregate domain labels: %s", exc)

        features = None
        if outputs and outputs[0].get("features") is not None:
            features = torch.cat([x["features"] for x in outputs])

        separate_logits1 = torch.cat([x["seprate_logits1"] for x in outputs]).cpu()
        separate_logits2 = torch.cat([x["seprate_logits2"] for x in outputs]).cpu()
        separate_logits3 = torch.cat([x["seprate_logits3"] for x in outputs]).cpu()
        separate_logits4 = torch.cat([x["seprate_logits4"] for x in outputs]).cpu()

        entropy1 = -torch.sum(torch.softmax(pred_prob1, dim=-1) * torch.log(torch.softmax(pred_prob1, dim=-1) + 1e-9), dim=-1)
        entropy2 = -torch.sum(torch.softmax(pred_prob2, dim=-1) * torch.log(torch.softmax(pred_prob2, dim=-1) + 1e-9), dim=-1)
        entropy3 = -torch.sum(torch.softmax(pred_prob3, dim=-1) * torch.log(torch.softmax(pred_prob3, dim=-1) + 1e-9), dim=-1)
        entropy4 = -torch.sum(torch.softmax(pred_prob4, dim=-1) * torch.log(torch.softmax(pred_prob4, dim=-1) + 1e-9), dim=-1)
        energy1 = -(torch.logsumexp(pred_prob1, dim=-1))
        energy2 = -(torch.logsumexp(pred_prob2, dim=-1))
        energy3 = -(torch.logsumexp(pred_prob3, dim=-1))
        energy4 = -(torch.logsumexp(pred_prob4, dim=-1))

        probs_sep1 = torch.softmax(separate_logits1, dim=-1)
        probs_sep2 = torch.softmax(separate_logits2, dim=-1)
        probs_sep3 = torch.softmax(separate_logits3, dim=-1)
        probs_sep4 = torch.softmax(separate_logits4, dim=-1)
        entropy_sep1 = -torch.sum(probs_sep1 * torch.log(probs_sep1 + 1e-9), dim=-1)
        entropy_sep2 = -torch.sum(probs_sep2 * torch.log(probs_sep2 + 1e-9), dim=-1)
        entropy_sep3 = -torch.sum(probs_sep3 * torch.log(probs_sep3 + 1e-9), dim=-1)
        entropy_sep4 = -torch.sum(probs_sep4 * torch.log(probs_sep4 + 1e-9), dim=-1)
        energy_sep1 = -(torch.logsumexp(separate_logits1, dim=-1))
        energy_sep2 = -(torch.logsumexp(separate_logits2, dim=-1))
        energy_sep3 = -(torch.logsumexp(separate_logits3, dim=-1))
        energy_sep4 = -(torch.logsumexp(separate_logits4, dim=-1))

        if features is not None:
            mcd_scores1 = self.compute_mcd_score_with_features(features, self.model.mtlr1)
            mcd_scores2 = self.compute_mcd_score_with_features(features, self.model.mtlr2)
            mcd_scores3 = self.compute_mcd_score_with_features(features, self.model.mtlr3)
            mcd_scores4 = self.compute_mcd_score_with_features(features, self.model.mtlr4)
        else:
            zeros = torch.zeros_like(entropy1)
            mcd_scores1 = zeros
            mcd_scores2 = zeros
            mcd_scores3 = zeros
            mcd_scores4 = zeros

        extended_scores_enabled = bool(getattr(self.hparams, "extended_scores_enabled", False))
        metric_bootstrap_n = int(getattr(self.hparams, "metric_bootstrap_n", 0) or 0)
        metric_bootstrap_seed = int(getattr(self.hparams, "metric_bootstrap_seed", 786) or 786)
        extended_score_entries = []
        if extended_scores_enabled:
            extended_score_entries = extended_score_family_from_logits(
                {
                    "OS": pred_prob1.detach().cpu().numpy(),
                    "LFFS": pred_prob2.detach().cpu().numpy(),
                    "RFFS": pred_prob3.detach().cpu().numpy(),
                    "DFFS": pred_prob4.detach().cpu().numpy(),
                }
            )

        entropy_results = pd.DataFrame({
            "Entropy_OS": entropy1.detach().cpu().numpy(),
            "Entropy_LFFS": entropy2.detach().cpu().numpy(),
            "Entropy_RFFS": entropy3.detach().cpu().numpy(),
            "Entropy_DFFS": entropy4.detach().cpu().numpy(),
            "Energy_OS": energy1.detach().cpu().numpy(),
            "Energy_LFFS": energy2.detach().cpu().numpy(),
            "Energy_RFFS": energy3.detach().cpu().numpy(),
            "Energy_DFFS": energy4.detach().cpu().numpy(),
            "MCD_OS": mcd_scores1.detach().cpu().numpy(),
            "MCD_LFFS": mcd_scores2.detach().cpu().numpy(),
            "MCD_RFFS": mcd_scores3.detach().cpu().numpy(),
            "MCD_DFFS": mcd_scores4.detach().cpu().numpy(),
            "Sep_Entropy_OS": entropy_sep1.detach().cpu().numpy(),
            "Sep_Entropy_LFFS": entropy_sep2.detach().cpu().numpy(),
            "Sep_Entropy_RFFS": entropy_sep3.detach().cpu().numpy(),
            "Sep_Energy_DFFS": energy_sep4.detach().cpu().numpy(),
            "Sep_Energy_OS": energy_sep1.detach().cpu().numpy(),
            "Sep_Energy_LFFS": energy_sep2.detach().cpu().numpy(),
            "Sep_Energy_RFFS": energy_sep3.detach().cpu().numpy(),
            "Sep_Entropy_DFFS": entropy_sep4.detach().cpu().numpy(),
        })

        for name, values in extended_score_entries:
            entropy_results[name] = values

        hazard_reference_mode = getattr(self.hparams, "hazard_reference_mode", "none")
        hazard_reference_dir = getattr(self.hparams, "hazard_reference_dir", None)
        hazard_dev_entries = []
        if hazard_reference_dir:
            hazard_reference_dir = to_absolute_path(str(hazard_reference_dir))

        if hazard_reference_mode == "use" and hazard_reference_dir:
            try:
                ref_os = np.load(Path(hazard_reference_dir) / "hazard_mean_OS.npy")
                ref_lffs = np.load(Path(hazard_reference_dir) / "hazard_mean_LFFS.npy")
                ref_rffs = np.load(Path(hazard_reference_dir) / "hazard_mean_RFFS.npy")
                ref_dffs = np.load(Path(hazard_reference_dir) / "hazard_mean_DFFS.npy")
            except FileNotFoundError as exc:
                log.warning("Failed to load hazard reference means: %s", exc)
            else:
                diff_os = pred_hazard1 - ref_os
                diff_lffs = pred_hazard2 - ref_lffs
                diff_rffs = pred_hazard3 - ref_rffs
                diff_dffs = pred_hazard4 - ref_dffs

                pos_diff_os = np.maximum(diff_os, 0)
                pos_diff_lffs = np.maximum(diff_lffs, 0)
                pos_diff_rffs = np.maximum(diff_rffs, 0)
                pos_diff_dffs = np.maximum(diff_dffs, 0)

                hazard_dev_entries = [
                    ("HazardDev_OS", np.mean(diff_os, axis=1)),
                    ("HazardDev_LFFS", np.mean(diff_lffs, axis=1)),
                    ("HazardDev_RFFS", np.mean(diff_rffs, axis=1)),
                    ("HazardDev_DFFS", np.mean(diff_dffs, axis=1)),
                    ("HazardDevPos_OS", np.mean(pos_diff_os, axis=1)),
                    ("HazardDevPos_LFFS", np.mean(pos_diff_lffs, axis=1)),
                    ("HazardDevPos_RFFS", np.mean(pos_diff_rffs, axis=1)),
                    ("HazardDevPos_DFFS", np.mean(pos_diff_dffs, axis=1)),
                    ("HazardDevMax_OS", diff_os[np.arange(len(diff_os)), np.argmax(np.abs(diff_os), axis=1)]),
                    ("HazardDevMax_LFFS", diff_lffs[np.arange(len(diff_lffs)), np.argmax(np.abs(diff_lffs), axis=1)]),
                    ("HazardDevMax_RFFS", diff_rffs[np.arange(len(diff_rffs)), np.argmax(np.abs(diff_rffs), axis=1)]),
                    ("HazardDevMax_DFFS", diff_dffs[np.arange(len(diff_dffs)), np.argmax(np.abs(diff_dffs), axis=1)]),
                    ("HazardDevPosMax_OS", np.max(pos_diff_os, axis=1)),
                    ("HazardDevPosMax_LFFS", np.max(pos_diff_lffs, axis=1)),
                    ("HazardDevPosMax_RFFS", np.max(pos_diff_rffs, axis=1)),
                    ("HazardDevPosMax_DFFS", np.max(pos_diff_dffs, axis=1)),
                ]
                if extended_scores_enabled:
                    extended_hazard_entries = hazard_l1_l2_entries(
                        {
                            "OS": diff_os,
                            "LFFS": diff_lffs,
                            "RFFS": diff_rffs,
                            "DFFS": diff_dffs,
                        }
                    )
                    hazard_dev_entries.extend(extended_hazard_entries)
                    for name, values in extended_hazard_entries:
                        export_name = name.replace("HazardDevL1_", "Test_HazardDeviationL1_")
                        export_name = export_name.replace("HazardDevL2_", "Test_HazardDeviationL2_")
                        np.save(f"{export_name}.npy", values)

                for name, values in hazard_dev_entries:
                    entropy_results[name] = values

        if domain_tensor is not None and domain_tensor.numel() > 0:
            domain_np = domain_tensor.numpy().astype(int)
            if 0 in np.unique(domain_np) and 1 in np.unique(domain_np):
                id_mask = domain_np == 0
                ood_mask = domain_np == 1
                binary_targets = (domain_np == 1).astype(int)

                id_ci_logs = {}
                ood_ci_logs = {}
                ci_specs = [
                    ("OS", true_time1.numpy(), pred_risk1, true_event1.numpy()),
                    ("LFFS", true_time2.numpy(), pred_risk2, true_event2.numpy()),
                    ("RFFS", true_time3.numpy(), pred_risk3, true_event3.numpy()),
                    ("DFFS", true_time4.numpy(), pred_risk4, true_event4.numpy()),
                ]
                for task_name, times, risks, events in ci_specs:
                    if id_mask.sum() >= 2:
                        try:
                            id_ci_logs[f"test/id/{task_name}_CI"] = float(concordance_index(times[id_mask], -risks[id_mask], event_observed=events[id_mask]))
                        except Exception:
                            pass
                    if ood_mask.sum() >= 2:
                        try:
                            ood_ci_logs[f"test/ood/{task_name}_CI"] = float(concordance_index(times[ood_mask], -risks[ood_mask], event_observed=events[ood_mask]))
                        except Exception:
                            pass
                if id_ci_logs:
                    id_ci_logs["test/id/AVG_CI"] = float(np.mean(list(id_ci_logs.values())))
                    self.log_dict(id_ci_logs)
                if ood_ci_logs:
                    ood_ci_logs["test/ood/AVG_CI"] = float(np.mean(list(ood_ci_logs.values())))
                    self.log_dict(ood_ci_logs)

                if metric_bootstrap_n > 0:
                    cindex_rows = cindex_metric_rows(
                        ci_specs,
                        domain_labels=domain_np,
                        id_label=0,
                        ood_label=1,
                        n_bootstrap=metric_bootstrap_n,
                        seed=metric_bootstrap_seed,
                    )
                    pd.DataFrame(cindex_rows).to_csv("Test_CIndex_metrics.csv", index=False)

                score_arrays = [
                    ("Entropy_OS", entropy_results["Entropy_OS"].to_numpy()),
                    ("Entropy_LFFS", entropy_results["Entropy_LFFS"].to_numpy()),
                    ("Entropy_RFFS", entropy_results["Entropy_RFFS"].to_numpy()),
                    ("Entropy_DFFS", entropy_results["Entropy_DFFS"].to_numpy()),
                    ("Energy_OS", entropy_results["Energy_OS"].to_numpy()),
                    ("Energy_LFFS", entropy_results["Energy_LFFS"].to_numpy()),
                    ("Energy_RFFS", entropy_results["Energy_RFFS"].to_numpy()),
                    ("Energy_DFFS", entropy_results["Energy_DFFS"].to_numpy()),
                    ("MCD_OS", entropy_results["MCD_OS"].to_numpy()),
                    ("MCD_LFFS", entropy_results["MCD_LFFS"].to_numpy()),
                    ("MCD_RFFS", entropy_results["MCD_RFFS"].to_numpy()),
                    ("MCD_DFFS", entropy_results["MCD_DFFS"].to_numpy()),
                    ("Sep_Entropy_OS", entropy_results["Sep_Entropy_OS"].to_numpy()),
                    ("Sep_Entropy_LFFS", entropy_results["Sep_Entropy_LFFS"].to_numpy()),
                    ("Sep_Entropy_RFFS", entropy_results["Sep_Entropy_RFFS"].to_numpy()),
                    ("Sep_Entropy_DFFS", entropy_results["Sep_Entropy_DFFS"].to_numpy()),
                    ("Sep_Energy_OS", entropy_results["Sep_Energy_OS"].to_numpy()),
                    ("Sep_Energy_LFFS", entropy_results["Sep_Energy_LFFS"].to_numpy()),
                    ("Sep_Energy_RFFS", entropy_results["Sep_Energy_RFFS"].to_numpy()),
                    ("Sep_Energy_DFFS", entropy_results["Sep_Energy_DFFS"].to_numpy()),
                ]
                score_arrays.extend(hazard_dev_entries)
                score_arrays.extend(extended_score_entries)

                metrics_rows = []
                for score_name, values in score_arrays:
                    metrics_rows.append(
                        mixed_ood_metric_row(
                            score_name,
                            values,
                            binary_targets,
                            id_mask,
                            ood_mask,
                            n_bootstrap=metric_bootstrap_n,
                            seed=metric_bootstrap_seed,
                        )
                    )

                hazard_lda_columns = [
                    "HazardDev_OS", "HazardDev_LFFS", "HazardDev_RFFS", "HazardDev_DFFS",
                    "HazardDevPos_OS", "HazardDevPos_LFFS", "HazardDevPos_RFFS", "HazardDevPos_DFFS",
                ]
                available_hazard_cols = [col for col in hazard_lda_columns if col in entropy_results.columns]
                if len(available_hazard_cols) >= 2 and id_mask.any() and ood_mask.any():
                    hazard_features = entropy_results[available_hazard_cols].to_numpy()
                    cov = np.cov(hazard_features, rowvar=False) + np.eye(len(available_hazard_cols)) * 1e-6
                    lda_weights = np.linalg.pinv(cov) @ (hazard_features[ood_mask].mean(axis=0) - hazard_features[id_mask].mean(axis=0))
                    hazard_dev_scores = hazard_features @ lda_weights
                    entropy_results["HazardDev_LDA"] = hazard_dev_scores
                    try:
                        lda_auroc = roc_auc_score(binary_targets, hazard_dev_scores)
                        lda_aupr = average_precision_score(binary_targets, hazard_dev_scores)
                        fpr_lda, tpr_lda, _ = roc_curve(binary_targets, hazard_dev_scores)
                        lda_fpr95 = float(np.interp(0.95, tpr_lda, fpr_lda))
                    except ValueError:
                        lda_auroc = float("nan")
                        lda_aupr = float("nan")
                        lda_fpr95 = float("nan")
                    metrics_rows.append(
                        mixed_ood_metric_row(
                            "HazardDev_LDA",
                            hazard_dev_scores,
                            binary_targets,
                            id_mask,
                            ood_mask,
                            n_bootstrap=metric_bootstrap_n,
                            seed=metric_bootstrap_seed,
                        )
                    )

                mixed_metrics_df = pd.DataFrame(metrics_rows)
                mixed_metrics_df.to_csv("Test_Mixed_OOD_metrics.csv", index=False)

        if extended_scores_enabled:
            entropy_results.to_csv("Test_Entropy.csv", index=False)

        np.save("Test_Entropy_OS.npy", entropy_results["Entropy_OS"].to_numpy())
        np.save("Test_Entropy_LFFS.npy", entropy_results["Entropy_LFFS"].to_numpy())
        np.save("Test_Entropy_RFFS.npy", entropy_results["Entropy_RFFS"].to_numpy())
        np.save("Test_Entropy_DFFS.npy", entropy_results["Entropy_DFFS"].to_numpy())
        np.save("Test_MCD_OS.npy", entropy_results["MCD_OS"].to_numpy())

        return {"loss": loss, "OS-CI": ci_event1, "LFFS-CI": ci_event2, "RFFS-CI": ci_event3, "DFFS-CI": ci_event4}
   

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = make_optimizer(AdamW, self.model, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=0.001, factor=0.1, verbose=True),
          
            "monitor": "val/loss",
        }

        return [optimizer] , [scheduler]
    
    

def make_optimizer(opt_cls, model, **kwargs):
    """Creates a PyTorch optimizer for MTLR training."""
    params_dict = dict(model.named_parameters())
    weights = [v for k, v in params_dict.items() if "mtlr" not in k and "bias" not in k]
    biases = [v for k, v in params_dict.items() if "bias" in k]
    mtlr_weights = [v for k, v in params_dict.items() if "mtlr_weight" in k]
    # Don't use weight decay on the biases and MTLR parameters, which have
    # their own separate L2 regularization
    optimizer = opt_cls([
        {"params": weights},
        {"params": biases, "weight_decay": 0.},
        {"params": mtlr_weights, "weight_decay": 0.},
    ], **kwargs)
    return optimizer




class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
    

def dice(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()


def hausdorff_distance(image0, image1):
    """Code copied from 
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0.cpu()))
    b_points = np.transpose(np.nonzero(image1.cpu()))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))

def pysurvival_mtlr_loss(self, model, X_cens, X_uncens, Y_cens, Y_uncens, 
    Triangle, l2_reg, l2_smooth):
    """ Computes the loss function of the any MTLR model. 
        All the operations have been vectorized to ensure optimal speed

        Modified based on
        https://github.com/square/pysurvival/blob/master/pysurvival/models/multi_task.py
    """

    # Likelihood Calculations -- Uncensored
    score_uncens = model(X_uncens)
    phi_uncens = torch.exp( torch.mm(score_uncens, Triangle) )
    reduc_phi_uncens = torch.sum(phi_uncens*Y_uncens, dim = 1)

    # Likelihood Calculations -- Censored
    score_cens = model(X_cens)
    phi_cens = torch.exp( torch.mm(score_cens, Triangle) )
    reduc_phi_cens = torch.sum( phi_cens*Y_cens, dim = 1)

    # Likelihood Calculations -- Normalization
    z_uncens = torch.exp( torch.mm(score_uncens, Triangle) )
    reduc_z_uncens = torch.sum( z_uncens, dim = 1)

    z_cens = torch.exp( torch.mm(score_cens, Triangle) )
    reduc_z_cens = torch.sum( z_cens, dim = 1)

    # MTLR cost function
    loss = - (
                torch.sum( torch.log(reduc_phi_uncens) ) \
                + torch.sum( torch.log(reduc_phi_cens) )  \

                - torch.sum( torch.log(reduc_z_uncens) ) \
                - torch.sum( torch.log(reduc_z_cens) ) 
                )

    # Adding the regularized loss
    nb_set_parameters = len(list(model.parameters()))
    for i, w in enumerate(model.parameters()):
        loss += l2_reg*torch.sum(w*w)/2.
        
        if i >= nb_set_parameters - 2:
            loss += l2_smooth*norm_diff(w)
            
    return loss

def norm_diff(W):
    """ Special norm function for the last layer of the MTLR """
    dims=len(W.shape)
    if dims==1:
        diff = W[1:]-W[:-1]
    elif dims==2:
        diff = W[1:, :]-W[:-1, :]
    return torch.sum(diff*diff)
