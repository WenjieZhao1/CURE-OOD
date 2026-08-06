import os
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pytorch_lightning import LightningModule
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import scale
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification.accuracy import Accuracy
from torchmtlr import (MTLR, mtlr_hazard, mtlr_neg_log_likelihood, mtlr_risk,
                       mtlr_survival)

from src.utils.evaluation_metrics import (cindex_metric_rows,
                                        hazard_l1_l2_entries,
                                        mixed_ood_metric_row,
                                        extended_score_family_from_logits)
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

class DEEP_MTLR(LightningModule):
    """Lightning module for multi-label survival prediction and OOD evaluation."""

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
        .. [1] K. He et al. 'Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification',
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
        logits1, logits2, logits3, logits4, feat_norm, seprate_logits1, seprate_logits2, seprate_logits3, seprate_logits4 = self.forward((sample, clin_var))
        
        features = self.model.get_mtlr_features((sample, clin_var))

        loss_mtlr = mtlr_neg_log_likelihood(logits1, y1.float(), self.model, self.hparams['C1'], average=True)
        loss_mtlr += mtlr_neg_log_likelihood(logits2, y2.float(), self.model, self.hparams['C1'], average=True)
        loss_mtlr += mtlr_neg_log_likelihood(logits3, y3.float(), self.model, self.hparams['C1'], average=True)
        loss_mtlr += mtlr_neg_log_likelihood(logits4, y4.float(), self.model, self.hparams['C1'], average=True)
        loss = loss_mtlr
        

        return loss, logits1, logits2,logits3, logits4, y1,y2,y3,y4, labels, feat_norm, features, seprate_logits1, seprate_logits2, seprate_logits3, seprate_logits4

    def training_step(self, batch: Any, batch_idx: int):
        step_outputs = self.step(batch)
        loss, preds1, preds2, preds3, preds4, y1, y2, y3, y4, labels, feat_norm, features = step_outputs[:12]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds1": preds1, "preds2": preds2, "preds3": preds3, "preds4": preds4,"labels": labels}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        pred_risk1 = mtlr_risk(pred_prob1).detach().numpy()  
        ci_event1  = concordance_index(true_time1, -pred_risk1, event_observed=true_event1)
        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu() 
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        pred_risk2 = mtlr_risk(pred_prob2).detach().numpy()  
        ci_event2  = concordance_index(true_time2, -pred_risk2, event_observed=true_event2)
        
        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu() 
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        pred_risk3 = mtlr_risk(pred_prob3).detach().numpy()  
        ci_event3  = concordance_index(true_time3, -pred_risk3, event_observed=true_event3)
        
        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu() 
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        pred_risk4 = mtlr_risk(pred_prob4).detach().numpy()  
        ci_event4  = concordance_index(true_time4, -pred_risk4, event_observed=true_event4)

        import torch
        import torch.nn.functional as F

        cos12 = F.cosine_similarity(pred_prob1, pred_prob2, dim=1)
        cos13 = F.cosine_similarity(pred_prob1, pred_prob3, dim=1)
        cos14 = F.cosine_similarity(pred_prob1, pred_prob4, dim=1)
        cos23 = F.cosine_similarity(pred_prob2, pred_prob3, dim=1)
        cos24 = F.cosine_similarity(pred_prob2, pred_prob4, dim=1)
        cos34 = F.cosine_similarity(pred_prob3, pred_prob4, dim=1)
        cos_sims = torch.stack([cos12, cos13, cos14, cos23, cos24, cos34], dim=1)  # (N, 6)

        labels = torch.unique(true_event1)
        class_means = []
        for label in labels:
            mask = (true_event1 == label)
            class_mean = cos_sims[mask].mean(dim=0)
            class_means.append(class_mean)
        class_means = torch.stack(class_means, dim=0)  # (n_class, 6)

        sim_mean_global = cos_sims.mean(dim=0, keepdim=True)    # (1, 6)
        X_centered = cos_sims - sim_mean_global                 # (N, 6)
        sim_cov = (X_centered.T @ X_centered) / cos_sims.size(0)   # (6, 6)
        eps = 1e-3
        sim_cov += torch.eye(sim_cov.size(0), device=sim_cov.device) * eps

        sim_precision = torch.inverse(sim_cov)  # (6, 6)

        torch.save({
            'class_means': class_means.cpu(),     # (n_class, 6)
            'cov': sim_cov.cpu(),                 # (6, 6)
            'precision': sim_precision.cpu(),     # (6, 6)
            'labels': labels.cpu(),
        }, 'train_cosine_stats_fullmaha.pt')
        
        # ===== End cosine similarity statistics =====

        log = {"train/OS_CI": ci_event1,
               "train/LFFS_CI": ci_event2,
               "train/RFFS_CI": ci_event3,
               "train/DFFS_CI": ci_event4,
               "train/AVG_CI": (ci_event1+ci_event2+ci_event3+ci_event4)/4,
               }  
        self.log_dict(log)
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        step_outputs = self.step(batch)
        loss, preds1, preds2, preds3, preds4, y1, y2, y3, y4, labels, feat_norm, features = step_outputs[:12]
        return {"loss": loss, "preds1": preds1, "preds2": preds2, "preds3": preds3, "preds4": preds4, "feat_norm": feat_norm, "labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        loss        = torch.stack([x["loss"] for x in outputs]).mean()

        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        pred_risk1 = mtlr_risk(pred_prob1).detach().numpy()  
        ci_event1  = concordance_index(true_time1, -pred_risk1, event_observed=true_event1)
        
        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu() 
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        pred_risk2 = mtlr_risk(pred_prob2).detach().numpy()  
        ci_event2  = concordance_index(true_time2, -pred_risk2, event_observed=true_event2)
        
        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu() 
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        pred_risk3 = mtlr_risk(pred_prob3).detach().numpy()  
        ci_event3  = concordance_index(true_time3, -pred_risk3, event_observed=true_event3)
        
        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu() 
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        pred_risk4 = mtlr_risk(pred_prob4).detach().numpy()  
        ci_event4  = concordance_index(true_time4, -pred_risk4, event_observed=true_event4)

        log = {"val/loss": loss,
               "val/OS_CI": ci_event1,
               "val/LFFS_CI": ci_event2,
               "val/RFFS_CI": ci_event3,
               "val/DFFS_CI": ci_event4,
               "val/AVG_CI": (ci_event1+ci_event2+ci_event3+ci_event4)/4,
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
        loss, preds1, preds2, preds3, preds4, y1,y2,y3,y4, labels, feat_norm, features, seprate_logits1, seprate_logits2, seprate_logits3, seprate_logits4 = self.step(batch)
        return {"loss": loss, "preds1": preds1, "preds2": preds2, "preds3": preds3, "preds4": preds4,"labels": labels, 'y1':y1, 'y2':y2, 'y3':y3, 'y4':y4, 'feat_norm':feat_norm, 'features':features, 'seprate_logits1':seprate_logits1, 'seprate_logits2':seprate_logits2, 'seprate_logits3':seprate_logits3, 'seprate_logits4':seprate_logits4}
    
    def compute_mcd_score_with_features(self, features, mtlr_layer):
        """
        Compute MCD scores from feature vectors and an MTLR layer.
        
        Args:
            features: Input feature vectors with shape (N, features).
            mtlr_layer: MTLR layer used to produce logits.
        
        Returns:
            mcd_scores: MCD scores for each sample, shape (N,)
        """
        eps = 1e-9
        N = features.shape[0]
        
        z = torch.matmul(features, mtlr_layer.mtlr_weight) + mtlr_layer.mtlr_bias  # (N, m)
        m = z.shape[1]
        mtlr_logits = torch.matmul(z, mtlr_layer.G)  # (N, m+1)
        p = torch.softmax(mtlr_logits, dim=1)  # (N, m+1)
        
        q = torch.sigmoid(z)
        
        num_intervals = m + 1
        ptilde_unnorm = torch.zeros(N, num_intervals, device=z.device)
        
        for k in range(num_intervals):
            # prefix: ∏_{i<=k}(1 - q[i])
            if k == 0:
                prefix = torch.ones(N, device=z.device)
            else:
                prefix = torch.prod(1 - q[:, :k], dim=1)
            
            # suffix: ∏_{i>k} q[i]  
            if k >= m:
                suffix = torch.ones(N, device=z.device)
            else:
                suffix = torch.prod(q[:, k:], dim=1)
            
            ptilde_unnorm[:, k] = prefix * suffix
        
        ptilde_sum = torch.sum(ptilde_unnorm, dim=1, keepdim=True)
        ptilde = ptilde_unnorm / (ptilde_sum + eps)  # (N, m+1)
        
        # JSD(p || ptilde) = 0.5 * (KL(p || M) + KL(ptilde || M))
        # where M = 0.5 * (p + ptilde)
        M = 0.5 * (p + ptilde)
        M = M + eps
        
        def kl_div(P, Q):
            ratio = (P + eps) / (Q + eps)
            return torch.sum(P * torch.log(ratio), dim=1)
        
        kl_p_M = kl_div(p, M)
        kl_ptilde_M = kl_div(ptilde, M)
        
        mcd_scores = 0.5 * (kl_p_M + kl_ptilde_M)
        
        return mcd_scores

    def test_epoch_end(self, outputs: List[Any]):
        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob1   = torch.cat([x["preds1"] for x in outputs]).cpu() 
        true_time1   = torch.cat([x["labels"]["time1"] for x in outputs]).cpu()
        true_event1  = torch.cat([x["labels"]["event1"] for x in outputs]).cpu()
        pred_risk1 = mtlr_risk(pred_prob1).detach().numpy()
        pred_hazard1 = mtlr_hazard(pred_prob1).detach().numpy()
        ci_event1  = concordance_index(true_time1, -pred_risk1, event_observed=true_event1)

        pred_prob2   = torch.cat([x["preds2"] for x in outputs]).cpu()
        true_time2   = torch.cat([x["labels"]["time2"] for x in outputs]).cpu()
        true_event2  = torch.cat([x["labels"]["event2"] for x in outputs]).cpu()
        pred_risk2 = mtlr_risk(pred_prob2).detach().numpy()
        pred_hazard2 = mtlr_hazard(pred_prob2).detach().numpy()
        ci_event2  = concordance_index(true_time2, -pred_risk2, event_observed=true_event2)

        pred_prob3   = torch.cat([x["preds3"] for x in outputs]).cpu()
        true_time3   = torch.cat([x["labels"]["time3"] for x in outputs]).cpu()
        true_event3  = torch.cat([x["labels"]["event3"] for x in outputs]).cpu()
        pred_risk3 = mtlr_risk(pred_prob3).detach().numpy()
        pred_hazard3 = mtlr_hazard(pred_prob3).detach().numpy()
        ci_event3  = concordance_index(true_time3, -pred_risk3, event_observed=true_event3)

        pred_prob4   = torch.cat([x["preds4"] for x in outputs]).cpu()
        true_time4   = torch.cat([x["labels"]["time4"] for x in outputs]).cpu()
        true_event4  = torch.cat([x["labels"]["event4"] for x in outputs]).cpu()
        pred_risk4 = mtlr_risk(pred_prob4).detach().numpy()
        pred_hazard4 = mtlr_hazard(pred_prob4).detach().numpy()
        ci_event4  = concordance_index(true_time4, -pred_risk4, event_observed=true_event4)

        domain_tensor = None
        mini_batch_tensor = None
        if outputs and isinstance(outputs[0]["labels"], dict):
            if "domain" in outputs[0]["labels"]:
                try:
                    domain_tensor = torch.cat([x["labels"]["domain"] for x in outputs]).cpu()
                except Exception as exc:
                    log.warning("Failed to aggregate domain labels: %s", exc)
                    domain_tensor = None
            if "mini_batch_id" in outputs[0]["labels"]:
                try:
                    mini_batch_tensor = torch.cat([x["labels"]["mini_batch_id"] for x in outputs]).cpu()
                except Exception as exc:
                    log.warning("Failed to aggregate mini-batch identifiers: %s", exc)
                    mini_batch_tensor = None

        features = torch.cat([x["features"] for x in outputs])  # (N, feature_dim)
        
        mcd_scores1 = self.compute_mcd_score_with_features(features, self.model.mtlr1)
        mcd_scores2 = self.compute_mcd_score_with_features(features, self.model.mtlr2)
        mcd_scores3 = self.compute_mcd_score_with_features(features, self.model.mtlr3)
        mcd_scores4 = self.compute_mcd_score_with_features(features, self.model.mtlr4)
        
        probs = torch.nn.functional.softmax(pred_prob1, dim=-1)  # (num_samples, num_time_bins)
        entropy1 = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (num_samples,)
        energy1 = -(torch.logsumexp(pred_prob1, dim=-1)) 

        probs = torch.nn.functional.softmax(pred_prob2, dim=-1)  # (num_samples, num_time_bins)
        entropy2 = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (num_samples,)
        energy2 = -(torch.logsumexp(pred_prob2, dim=-1)) 

        probs = torch.nn.functional.softmax(pred_prob3, dim=-1)  # (num_samples, num_time_bins)
        entropy3 = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (num_samples,)
        energy3 = -(torch.logsumexp(pred_prob3, dim=-1)) 

        probs = torch.nn.functional.softmax(pred_prob4, dim=-1)  # (num_samples, num_time_bins)
        entropy4 = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # (num_samples,)
        energy4 = -(torch.logsumexp(pred_prob4, dim=-1))

        separate_logits1 = torch.cat([x["seprate_logits1"] for x in outputs]).cpu()
        separate_logits2 = torch.cat([x["seprate_logits2"] for x in outputs]).cpu()
        separate_logits3 = torch.cat([x["seprate_logits3"] for x in outputs]).cpu()
        separate_logits4 = torch.cat([x["seprate_logits4"] for x in outputs]).cpu()

        probs_sep1 = torch.nn.functional.softmax(separate_logits1, dim=-1)
        entropy_sep1 = -torch.sum(probs_sep1 * torch.log(probs_sep1 + 1e-9), dim=-1)
        energy_sep1 = -(torch.logsumexp(separate_logits1, dim=-1))

        probs_sep2 = torch.nn.functional.softmax(separate_logits2, dim=-1)
        entropy_sep2 = -torch.sum(probs_sep2 * torch.log(probs_sep2 + 1e-9), dim=-1)
        energy_sep2 = -(torch.logsumexp(separate_logits2, dim=-1))

        probs_sep3 = torch.nn.functional.softmax(separate_logits3, dim=-1)
        entropy_sep3 = -torch.sum(probs_sep3 * torch.log(probs_sep3 + 1e-9), dim=-1)
        energy_sep3 = -(torch.logsumexp(separate_logits3, dim=-1))

        probs_sep4 = torch.nn.functional.softmax(separate_logits4, dim=-1)
        entropy_sep4 = -torch.sum(probs_sep4 * torch.log(probs_sep4 + 1e-9), dim=-1)
        energy_sep4 = -(torch.logsumexp(separate_logits4, dim=-1))

        metrics_log = {"test/loss": loss,
                       "test/OS_CI": ci_event1,
                       "test/LFFS_CI": ci_event2,
                       "test/RFFS_CI": ci_event3,
                       "test/DFFS_CI": ci_event4,
                       "test/AVG_CI": (ci_event1+ci_event2+ci_event3+ci_event4)/4,
                       }
        self.log_dict(metrics_log)
        ## save entropy results
        entropy1_np = entropy1.cpu().numpy()
        entropy2_np = entropy2.cpu().numpy()
        entropy3_np = entropy3.cpu().numpy()
        entropy4_np = entropy4.cpu().numpy()

        energy1_np = energy1.cpu().numpy()
        energy2_np = energy2.cpu().numpy()
        energy3_np = energy3.cpu().numpy()
        energy4_np = energy4.cpu().numpy()

        mcd_scores1_np = mcd_scores1.cpu().numpy()
        mcd_scores2_np = mcd_scores2.cpu().numpy()
        mcd_scores3_np = mcd_scores3.cpu().numpy()
        mcd_scores4_np = mcd_scores4.cpu().numpy()

        entropy_sep1_np = entropy_sep1.cpu().numpy()
        entropy_sep2_np = entropy_sep2.cpu().numpy()
        entropy_sep3_np = entropy_sep3.cpu().numpy()
        entropy_sep4_np = entropy_sep4.cpu().numpy()

        energy_sep1_np = energy_sep1.cpu().numpy()
        energy_sep2_np = energy_sep2.cpu().numpy()
        energy_sep3_np = energy_sep3.cpu().numpy()
        energy_sep4_np = energy_sep4.cpu().numpy()

        extended_scores_enabled = bool(getattr(self.hparams, "extended_scores_enabled", False))
        metric_bootstrap_n = int(getattr(self.hparams, "metric_bootstrap_n", 0) or 0)
        metric_bootstrap_seed = int(getattr(self.hparams, "metric_bootstrap_seed", 786) or 786)
        extended_score_entries = []
        if extended_scores_enabled:
            extended_score_entries = extended_score_family_from_logits(
                {
                    "OS": pred_prob1.detach().numpy(),
                    "LFFS": pred_prob2.detach().numpy(),
                    "RFFS": pred_prob3.detach().numpy(),
                    "DFFS": pred_prob4.detach().numpy(),
                }
            )

        hazard_reference_mode = getattr(self.hparams, "hazard_reference_mode", "none")
        hazard_reference_dir = getattr(self.hparams, "hazard_reference_dir", None)
        if hazard_reference_dir:
            hazard_reference_dir = to_absolute_path(str(hazard_reference_dir))

        hazard_dev_entries = []

        if hazard_reference_mode == "compute":
            if hazard_reference_dir is None:
                log.warning("hazard_reference_mode is 'compute' but hazard_reference_dir is not set; skipping reference export.")
            else:
                os.makedirs(hazard_reference_dir, exist_ok=True)
                np.save(Path(hazard_reference_dir) / "hazard_mean_OS.npy", pred_hazard1.mean(axis=0))
                np.save(Path(hazard_reference_dir) / "hazard_mean_LFFS.npy", pred_hazard2.mean(axis=0))
                np.save(Path(hazard_reference_dir) / "hazard_mean_RFFS.npy", pred_hazard3.mean(axis=0))
                np.save(Path(hazard_reference_dir) / "hazard_mean_DFFS.npy", pred_hazard4.mean(axis=0))
                log.info("Hazard reference means saved to %s", hazard_reference_dir)

        elif hazard_reference_mode == "use":
            if hazard_reference_dir is None:
                log.warning("hazard_reference_mode is 'use' but hazard_reference_dir is not set; skipping hazard deviation metrics.")
            else:
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

                    hazard_dev_os = np.mean(diff_os, axis=1)
                    hazard_dev_lffs = np.mean(diff_lffs, axis=1)
                    hazard_dev_rffs = np.mean(diff_rffs, axis=1)
                    hazard_dev_dffs = np.mean(diff_dffs, axis=1)

                    pos_diff_os = np.maximum(diff_os, 0)
                    pos_diff_lffs = np.maximum(diff_lffs, 0)
                    pos_diff_rffs = np.maximum(diff_rffs, 0)
                    pos_diff_dffs = np.maximum(diff_dffs, 0)

                    hazard_dev_pos_os = np.mean(pos_diff_os, axis=1)
                    hazard_dev_pos_lffs = np.mean(pos_diff_lffs, axis=1)
                    hazard_dev_pos_rffs = np.mean(pos_diff_rffs, axis=1)
                    hazard_dev_pos_dffs = np.mean(pos_diff_dffs, axis=1)

                    # Max version: use maximum absolute difference across time bins, but keep the sign
                    # For each sample, find the difference with largest absolute value and keep its sign
                    abs_diff_os = np.abs(diff_os)
                    abs_diff_lffs = np.abs(diff_lffs)
                    abs_diff_rffs = np.abs(diff_rffs)
                    abs_diff_dffs = np.abs(diff_dffs)

                    max_idx_os = np.argmax(abs_diff_os, axis=1)
                    max_idx_lffs = np.argmax(abs_diff_lffs, axis=1)
                    max_idx_rffs = np.argmax(abs_diff_rffs, axis=1)
                    max_idx_dffs = np.argmax(abs_diff_dffs, axis=1)

                    hazard_dev_max_os = diff_os[np.arange(len(diff_os)), max_idx_os]
                    hazard_dev_max_lffs = diff_lffs[np.arange(len(diff_lffs)), max_idx_lffs]
                    hazard_dev_max_rffs = diff_rffs[np.arange(len(diff_rffs)), max_idx_rffs]
                    hazard_dev_max_dffs = diff_dffs[np.arange(len(diff_dffs)), max_idx_dffs]

                    hazard_dev_pos_max_os = np.max(pos_diff_os, axis=1)
                    hazard_dev_pos_max_lffs = np.max(pos_diff_lffs, axis=1)
                    hazard_dev_pos_max_rffs = np.max(pos_diff_rffs, axis=1)
                    hazard_dev_pos_max_dffs = np.max(pos_diff_dffs, axis=1)

                    np.save("Test_HazardDeviation_OS.npy", hazard_dev_os)
                    np.save("Test_HazardDeviation_LFFS.npy", hazard_dev_lffs)
                    np.save("Test_HazardDeviation_RFFS.npy", hazard_dev_rffs)
                    np.save("Test_HazardDeviation_DFFS.npy", hazard_dev_dffs)

                    np.save("Test_HazardDeviationPos_OS.npy", hazard_dev_pos_os)
                    np.save("Test_HazardDeviationPos_LFFS.npy", hazard_dev_pos_lffs)
                    np.save("Test_HazardDeviationPos_RFFS.npy", hazard_dev_pos_rffs)
                    np.save("Test_HazardDeviationPos_DFFS.npy", hazard_dev_pos_dffs)

                    np.save("Test_HazardDeviationMax_OS.npy", hazard_dev_max_os)
                    np.save("Test_HazardDeviationMax_LFFS.npy", hazard_dev_max_lffs)
                    np.save("Test_HazardDeviationMax_RFFS.npy", hazard_dev_max_rffs)
                    np.save("Test_HazardDeviationMax_DFFS.npy", hazard_dev_max_dffs)

                    np.save("Test_HazardDeviationPosMax_OS.npy", hazard_dev_pos_max_os)
                    np.save("Test_HazardDeviationPosMax_LFFS.npy", hazard_dev_pos_max_lffs)
                    np.save("Test_HazardDeviationPosMax_RFFS.npy", hazard_dev_pos_max_rffs)
                    np.save("Test_HazardDeviationPosMax_DFFS.npy", hazard_dev_pos_max_dffs)

                    # retain per-time-point differences for downstream inspection
                    np.save("Test_HazardDiff_OS.npy", diff_os)
                    np.save("Test_HazardDiff_LFFS.npy", diff_lffs)
                    np.save("Test_HazardDiff_RFFS.npy", diff_rffs)
                    np.save("Test_HazardDiff_DFFS.npy", diff_dffs)

                    extended_hazard_entries = []
                    if extended_scores_enabled:
                        extended_hazard_entries = hazard_l1_l2_entries(
                            {
                                "OS": diff_os,
                                "LFFS": diff_lffs,
                                "RFFS": diff_rffs,
                                "DFFS": diff_dffs,
                            }
                        )
                        for name, values in extended_hazard_entries:
                            export_name = name.replace("HazardDevL1_", "Test_HazardDeviationL1_")
                            export_name = export_name.replace("HazardDevL2_", "Test_HazardDeviationL2_")
                            np.save(f"{export_name}.npy", values)

                    hazard_dev_entries = [
                        ("HazardDev_OS", hazard_dev_os),
                        ("HazardDev_LFFS", hazard_dev_lffs),
                        ("HazardDev_RFFS", hazard_dev_rffs),
                        ("HazardDev_DFFS", hazard_dev_dffs),
                        ("HazardDevPos_OS", hazard_dev_pos_os),
                        ("HazardDevPos_LFFS", hazard_dev_pos_lffs),
                        ("HazardDevPos_RFFS", hazard_dev_pos_rffs),
                        ("HazardDevPos_DFFS", hazard_dev_pos_dffs),
                        ("HazardDevMax_OS", hazard_dev_max_os),
                        ("HazardDevMax_LFFS", hazard_dev_max_lffs),
                        ("HazardDevMax_RFFS", hazard_dev_max_rffs),
                        ("HazardDevMax_DFFS", hazard_dev_max_dffs),
                        ("HazardDevPosMax_OS", hazard_dev_pos_max_os),
                        ("HazardDevPosMax_LFFS", hazard_dev_pos_max_lffs),
                        ("HazardDevPosMax_RFFS", hazard_dev_pos_max_rffs),
                        ("HazardDevPosMax_DFFS", hazard_dev_pos_max_dffs),
                    ]
                    hazard_dev_entries.extend(extended_hazard_entries)

        entropy_results = pd.DataFrame({
            "Entropy_OS": entropy1_np,
            "Entropy_LFFS": entropy2_np,
            "Entropy_RFFS": entropy3_np,
            "Entropy_DFFS": entropy4_np,
            "Energy_OS": energy1_np,
            "Energy_LFFS": energy2_np,
            "Energy_RFFS": energy3_np,
            "Energy_DFFS": energy4_np,
            "MCD_OS": mcd_scores1_np,
            "MCD_LFFS": mcd_scores2_np,
            "MCD_RFFS": mcd_scores3_np,
            "MCD_DFFS": mcd_scores4_np,
            "Sep_Entropy_OS": entropy_sep1_np,
            "Sep_Entropy_LFFS": entropy_sep2_np,
            "Sep_Entropy_RFFS": entropy_sep3_np,
            "Sep_Entropy_DFFS": entropy_sep4_np,
            "Sep_Energy_OS": energy_sep1_np,
            "Sep_Energy_LFFS": energy_sep2_np,
            "Sep_Energy_RFFS": energy_sep3_np,
            "Sep_Energy_DFFS": energy_sep4_np,
        })

        for col_name, values in hazard_dev_entries:
            entropy_results[col_name] = values

        for col_name, values in extended_score_entries:
            entropy_results[col_name] = values

        hazard_dev_dict = dict(hazard_dev_entries) if hazard_dev_entries else {}
        extended_score_dict = dict(extended_score_entries) if extended_score_entries else {}

        pred_prob1_np = pred_prob1.detach().numpy()
        pred_prob2_np = pred_prob2.detach().numpy()
        pred_prob3_np = pred_prob3.detach().numpy()
        pred_prob4_np = pred_prob4.detach().numpy()

        separate_logits1_np = separate_logits1.detach().numpy()
        separate_logits2_np = separate_logits2.detach().numpy()
        separate_logits3_np = separate_logits3.detach().numpy()
        separate_logits4_np = separate_logits4.detach().numpy()

        domain_np = None
        id_label = None
        ood_label = None

        mini_batch_np = None
        if mini_batch_tensor is not None and mini_batch_tensor.numel() > 0:
            mini_batch_np = mini_batch_tensor.numpy().astype(int)
            entropy_results["mini_batch_id"] = mini_batch_np

        if domain_tensor is not None and domain_tensor.numel() > 0:
            domain_np = domain_tensor.numpy().astype(int)
            entropy_results["domain"] = domain_np

            unique_labels = np.unique(domain_np)
            label_counts = {label: int((domain_np == label).sum()) for label in unique_labels}

            if 0 in unique_labels and 1 in unique_labels:
                id_label, ood_label = 0, 1
            elif len(unique_labels) >= 2:
                sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
                id_label, ood_label = sorted_labels[0][0], sorted_labels[1][0]
            else:
                id_label = unique_labels[0]

            domain_name_map = {id_label: "ID"} if id_label is not None else {}
            if ood_label is not None:
                domain_name_map[ood_label] = "OOD"
            for label in unique_labels:
                domain_name_map.setdefault(label, f"class_{label}")

            entropy_results["domain_name"] = [domain_name_map.get(v, f"class_{v}") for v in domain_np]

            if ood_label is None:
                log.warning("Only one domain present in mixed dataloader; skipping ID/OOD metrics.")
            else:

                true_time1_np = true_time1.numpy()
                true_time2_np = true_time2.numpy()
                true_time3_np = true_time3.numpy()
                true_time4_np = true_time4.numpy()
                true_event1_np = true_event1.numpy()
                true_event2_np = true_event2.numpy()
                true_event3_np = true_event3.numpy()
                true_event4_np = true_event4.numpy()

                def _append_ci(logs, name, mask, times, risks, events, store):
                    if mask.sum() < 2:
                        return
                    try:
                        val = float(concordance_index(times[mask], -risks[mask], event_observed=events[mask]))
                    except Exception as exc:
                        log.warning("Failed to compute %s: %s", name, exc)
                        return
                    logs[name] = val
                    store.append(val)

                binary_targets = (domain_np == ood_label).astype(int)
                id_mask = domain_np == id_label
                ood_mask = domain_np == ood_label

                # Compute CI metrics for ID samples
                id_ci_logs = {}
                id_ci_values = []
                _append_ci(id_ci_logs, "test/id/OS_CI", id_mask, true_time1_np, pred_risk1, true_event1_np, id_ci_values)
                _append_ci(id_ci_logs, "test/id/LFFS_CI", id_mask, true_time2_np, pred_risk2, true_event2_np, id_ci_values)
                _append_ci(id_ci_logs, "test/id/RFFS_CI", id_mask, true_time3_np, pred_risk3, true_event3_np, id_ci_values)
                _append_ci(id_ci_logs, "test/id/DFFS_CI", id_mask, true_time4_np, pred_risk4, true_event4_np, id_ci_values)
                if id_ci_values:
                    id_ci_logs["test/id/AVG_CI"] = float(np.mean(id_ci_values))
                    self.log_dict(id_ci_logs, prog_bar=False, on_epoch=True, sync_dist=False)

                # Compute CI metrics for OOD samples
                ood_ci_logs = {}
                ood_ci_values = []
                _append_ci(ood_ci_logs, "test/ood/OS_CI", ood_mask, true_time1_np, pred_risk1, true_event1_np, ood_ci_values)
                _append_ci(ood_ci_logs, "test/ood/LFFS_CI", ood_mask, true_time2_np, pred_risk2, true_event2_np, ood_ci_values)
                _append_ci(ood_ci_logs, "test/ood/RFFS_CI", ood_mask, true_time3_np, pred_risk3, true_event3_np, ood_ci_values)
                _append_ci(ood_ci_logs, "test/ood/DFFS_CI", ood_mask, true_time4_np, pred_risk4, true_event4_np, ood_ci_values)
                if ood_ci_values:
                    ood_ci_logs["test/ood/AVG_CI"] = float(np.mean(ood_ci_values))
                    self.log_dict(ood_ci_logs, prog_bar=False, on_epoch=True, sync_dist=False)

                if metric_bootstrap_n > 0:
                    cindex_rows = cindex_metric_rows(
                        [
                            ("OS", true_time1_np, pred_risk1, true_event1_np),
                            ("LFFS", true_time2_np, pred_risk2, true_event2_np),
                            ("RFFS", true_time3_np, pred_risk3, true_event3_np),
                            ("DFFS", true_time4_np, pred_risk4, true_event4_np),
                        ],
                        domain_labels=domain_np,
                        id_label=id_label,
                        ood_label=ood_label,
                        n_bootstrap=metric_bootstrap_n,
                        seed=metric_bootstrap_seed,
                    )
                    pd.DataFrame(cindex_rows).to_csv("Test_CIndex_metrics.csv", index=False)

                score_arrays = [
                    ("Entropy_OS", entropy1_np),
                    ("Entropy_LFFS", entropy2_np),
                    ("Entropy_RFFS", entropy3_np),
                    ("Entropy_DFFS", entropy4_np),
                    ("Energy_OS", energy1_np),
                    ("Energy_LFFS", energy2_np),
                    ("Energy_RFFS", energy3_np),
                    ("Energy_DFFS", energy4_np),
                    ("MCD_OS", mcd_scores1_np),
                    ("MCD_LFFS", mcd_scores2_np),
                    ("MCD_RFFS", mcd_scores3_np),
                    ("MCD_DFFS", mcd_scores4_np),
                    ("Sep_Entropy_OS", entropy_sep1_np),
                    ("Sep_Entropy_LFFS", entropy_sep2_np),
                    ("Sep_Entropy_RFFS", entropy_sep3_np),
                    ("Sep_Entropy_DFFS", entropy_sep4_np),
                    ("Sep_Energy_OS", energy_sep1_np),
                    ("Sep_Energy_LFFS", energy_sep2_np),
                    ("Sep_Energy_RFFS", energy_sep3_np),
                    ("Sep_Energy_DFFS", energy_sep4_np),
                ]

                if hazard_dev_entries:
                    score_arrays.extend(hazard_dev_entries)
                if extended_score_entries:
                    score_arrays.extend(extended_score_entries)

                metrics_rows = []
                hazard_dev_scores = None
                hazard_dev_labels = None
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

                if hazard_dev_entries:
                    hazard_columns = [
                        "HazardDev_OS",
                        "HazardDev_LFFS",
                        "HazardDev_RFFS",
                        "HazardDev_DFFS",
                        "HazardDevPos_OS",
                        "HazardDevPos_LFFS",
                        "HazardDevPos_RFFS",
                        "HazardDevPos_DFFS",
                    ]

                    available_columns = [col for col in hazard_columns if col in entropy_results.columns]
                    if len(available_columns) >= 2:
                        hazard_features = entropy_results[available_columns].to_numpy()
                        id_feats = hazard_features[id_mask]
                        ood_feats = hazard_features[ood_mask]

                        if id_feats.size and ood_feats.size:
                            # compute covariance with small ridge for stability
                            diff = ood_feats.mean(axis=0) - id_feats.mean(axis=0)
                            cov = np.cov(hazard_features, rowvar=False)
                            cov += np.eye(cov.shape[0]) * 1e-6
                            inv_cov = np.linalg.pinv(cov)
                            lda_weights = inv_cov @ diff
                            hazard_dev_scores = hazard_features @ lda_weights
                            hazard_dev_labels = binary_targets
                            try:
                                lda_auroc = roc_auc_score(binary_targets, hazard_dev_scores)
                            except ValueError:
                                lda_auroc = float("nan")
                            try:
                                lda_aupr = average_precision_score(binary_targets, hazard_dev_scores)
                            except ValueError:
                                lda_aupr = float("nan")

                            # Calculate FPR95 for LDA
                            try:
                                fpr_lda, tpr_lda, thresholds_lda = roc_curve(binary_targets, hazard_dev_scores)
                                lda_fpr95 = float(np.interp(0.95, tpr_lda, fpr_lda))
                            except ValueError:
                                lda_fpr95 = float("nan")

                            entropy_results["HazardDev_LDA"] = hazard_dev_scores

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

                if id_mask.any() and ood_mask.any():
                    split_arrays = {
                        "Test_Entropy_OS": entropy1_np,
                        "Test_Entropy_LFFS": entropy2_np,
                        "Test_Entropy_RFFS": entropy3_np,
                        "Test_Entropy_DFFS": entropy4_np,
                        "Test_Energy_OS": energy1_np,
                        "Test_Energy_LFFS": energy2_np,
                        "Test_Energy_RFFS": energy3_np,
                        "Test_Energy_DFFS": energy4_np,
                        "Test_MCD_OS": mcd_scores1_np,
                        "Test_MCD_LFFS": mcd_scores2_np,
                        "Test_MCD_RFFS": mcd_scores3_np,
                        "Test_MCD_DFFS": mcd_scores4_np,
                        "Test_Sep_Entropy_OS": entropy_sep1_np,
                        "Test_Sep_Entropy_LFFS": entropy_sep2_np,
                        "Test_Sep_Entropy_RFFS": entropy_sep3_np,
                        "Test_Sep_Entropy_DFFS": entropy_sep4_np,
                        "Test_Sep_Energy_OS": energy_sep1_np,
                        "Test_Sep_Energy_LFFS": energy_sep2_np,
                        "Test_Sep_Energy_RFFS": energy_sep3_np,
                        "Test_Sep_Energy_DFFS": energy_sep4_np,
                        "Test_MTLR_Logits_OS": pred_prob1_np,
                        "Test_MTLR_Logits_LFFS": pred_prob2_np,
                        "Test_MTLR_Logits_RFFS": pred_prob3_np,
                        "Test_MTLR_Logits_DFFS": pred_prob4_np,
                        "Test_Separate_Logits_OS": separate_logits1_np,
                        "Test_Separate_Logits_LFFS": separate_logits2_np,
                        "Test_Separate_Logits_RFFS": separate_logits3_np,
                        "Test_Separate_Logits_DFFS": separate_logits4_np,
                        "Test_Risk_OS": pred_risk1.astype(np.float32),
                        "Test_Risk_LFFS": pred_risk2.astype(np.float32),
                        "Test_Risk_RFFS": pred_risk3.astype(np.float32),
                        "Test_Risk_DFFS": pred_risk4.astype(np.float32),
                        "Test_Hazard_OS": pred_hazard1.astype(np.float32),
                        "Test_Hazard_LFFS": pred_hazard2.astype(np.float32),
                        "Test_Hazard_RFFS": pred_hazard3.astype(np.float32),
                        "Test_Hazard_DFFS": pred_hazard4.astype(np.float32),
                    }

                    if hazard_dev_dict:
                        split_arrays.update({
                        "Test_HazardDeviation_OS": hazard_dev_dict.get("HazardDev_OS"),
                        "Test_HazardDeviation_LFFS": hazard_dev_dict.get("HazardDev_LFFS"),
                        "Test_HazardDeviation_RFFS": hazard_dev_dict.get("HazardDev_RFFS"),
                        "Test_HazardDeviation_DFFS": hazard_dev_dict.get("HazardDev_DFFS"),
                        "Test_HazardDeviationPos_OS": hazard_dev_dict.get("HazardDevPos_OS"),
                        "Test_HazardDeviationPos_LFFS": hazard_dev_dict.get("HazardDevPos_LFFS"),
                        "Test_HazardDeviationPos_RFFS": hazard_dev_dict.get("HazardDevPos_RFFS"),
                        "Test_HazardDeviationPos_DFFS": hazard_dev_dict.get("HazardDevPos_DFFS"),
                        })
                        if hazard_dev_scores is not None:
                            split_arrays["Test_HazardDeviation_LDA"] = hazard_dev_scores

                        extended_hazard_mappings = {
                            "HazardDevL1_OS": "Test_HazardDeviationL1_OS",
                            "HazardDevL1_LFFS": "Test_HazardDeviationL1_LFFS",
                            "HazardDevL1_RFFS": "Test_HazardDeviationL1_RFFS",
                            "HazardDevL1_DFFS": "Test_HazardDeviationL1_DFFS",
                            "HazardDevL2_OS": "Test_HazardDeviationL2_OS",
                            "HazardDevL2_LFFS": "Test_HazardDeviationL2_LFFS",
                            "HazardDevL2_RFFS": "Test_HazardDeviationL2_RFFS",
                            "HazardDevL2_DFFS": "Test_HazardDeviationL2_DFFS",
                        }
                        for col_name, export_name in extended_hazard_mappings.items():
                            if col_name in hazard_dev_dict:
                                split_arrays[export_name] = hazard_dev_dict[col_name]

                    for col_name, values in extended_score_dict.items():
                        split_arrays[f"Test_{col_name}"] = values

                    for prefix, values in split_arrays.items():
                        np.save(f"{prefix}_ID.npy", values[id_mask])
                        np.save(f"{prefix}_OOD.npy", values[ood_mask])
                else:
                    log.warning("Unable to create ID/OOD split files due to missing domain samples.")

                if mini_batch_np is not None:
                    valid_groups_mask = mini_batch_np >= 0
                    if valid_groups_mask.any():
                        grouped = entropy_results[valid_groups_mask].groupby("mini_batch_id")
                        aggregated_df = grouped.mean(numeric_only=True).reset_index()
                        aggregated_df["domain"] = grouped["domain"].first().values
                        aggregated_df["domain_name"] = grouped["domain_name"].first().values
                        aggregated_df["mini_batch_size"] = grouped.size().values

                        dispersion_inputs = {
                            "HazardDispersion_OS": pred_hazard1,
                            "HazardDispersion_LFFS": pred_hazard2,
                            "HazardDispersion_RFFS": pred_hazard3,
                            "HazardDispersion_DFFS": pred_hazard4,
                        }

                        mini_batch_ids = aggregated_df["mini_batch_id"].to_numpy().astype(int)
                        valid_mask = valid_groups_mask

                        for disp_name, hazard_matrix in dispersion_inputs.items():
                            dispersion_vals = np.full(mini_batch_ids.shape, np.nan, dtype=float)
                            for idx_batch, batch_id in enumerate(mini_batch_ids):
                                batch_mask = valid_mask & (mini_batch_np == batch_id)
                                batch_count = int(np.sum(batch_mask))
                                if batch_count >= 2:
                                    per_interval_var = np.var(hazard_matrix[batch_mask], axis=0, ddof=1)
                                    dispersion_vals[idx_batch] = float(np.mean(per_interval_var))
                            aggregated_df[disp_name] = dispersion_vals

                        aggregated_df.to_csv("Test_Entropy_MiniBatch.csv", index=False)

                        agg_domain = aggregated_df["domain"].astype(int).to_numpy()
                        binary_targets_mb = (agg_domain == ood_label).astype(int)
                        id_mask_mb = agg_domain == id_label
                        ood_mask_mb = agg_domain == ood_label

                        score_arrays_mb = [
                            ("Entropy_OS", aggregated_df["Entropy_OS"].to_numpy()),
                            ("Entropy_LFFS", aggregated_df["Entropy_LFFS"].to_numpy()),
                            ("Entropy_RFFS", aggregated_df["Entropy_RFFS"].to_numpy()),
                            ("Entropy_DFFS", aggregated_df["Entropy_DFFS"].to_numpy()),
                            ("Energy_OS", aggregated_df["Energy_OS"].to_numpy()),
                            ("Energy_LFFS", aggregated_df["Energy_LFFS"].to_numpy()),
                            ("Energy_RFFS", aggregated_df["Energy_RFFS"].to_numpy()),
                            ("Energy_DFFS", aggregated_df["Energy_DFFS"].to_numpy()),
                            ("MCD_OS", aggregated_df["MCD_OS"].to_numpy()),
                            ("MCD_LFFS", aggregated_df["MCD_LFFS"].to_numpy()),
                            ("MCD_RFFS", aggregated_df["MCD_RFFS"].to_numpy()),
                            ("MCD_DFFS", aggregated_df["MCD_DFFS"].to_numpy()),
                            ("Sep_Entropy_OS", aggregated_df["Sep_Entropy_OS"].to_numpy()),
                            ("Sep_Entropy_LFFS", aggregated_df["Sep_Entropy_LFFS"].to_numpy()),
                            ("Sep_Entropy_RFFS", aggregated_df["Sep_Entropy_RFFS"].to_numpy()),
                            ("Sep_Entropy_DFFS", aggregated_df["Sep_Entropy_DFFS"].to_numpy()),
                            ("Sep_Energy_OS", aggregated_df["Sep_Energy_OS"].to_numpy()),
                            ("Sep_Energy_LFFS", aggregated_df["Sep_Energy_LFFS"].to_numpy()),
                            ("Sep_Energy_RFFS", aggregated_df["Sep_Energy_RFFS"].to_numpy()),
                            ("Sep_Energy_DFFS", aggregated_df["Sep_Energy_DFFS"].to_numpy()),
                        ]

                        if hazard_dev_dict:
                            for key in [
                                "HazardDev_OS",
                                "HazardDev_LFFS",
                                "HazardDev_RFFS",
                                "HazardDev_DFFS",
                                "HazardDevPos_OS",
                                "HazardDevPos_LFFS",
                                "HazardDevPos_RFFS",
                                "HazardDevPos_DFFS",
                            ]:
                                if key in aggregated_df.columns:
                                    score_arrays_mb.append((key, aggregated_df[key].to_numpy()))

                        for disp_name in [
                            "HazardDispersion_OS",
                            "HazardDispersion_LFFS",
                            "HazardDispersion_RFFS",
                            "HazardDispersion_DFFS",
                        ]:
                            if disp_name in aggregated_df.columns:
                                score_arrays_mb.append((disp_name, aggregated_df[disp_name].to_numpy()))

                        mini_metrics_rows = []
                        hazard_dev_mb_scores = None
                        hazard_dev_mb_labels = None

                        for score_name, values in score_arrays_mb:
                            values = np.asarray(values)
                            valid_entries = ~np.isnan(values)

                            if not valid_entries.any():
                                mini_metrics_rows.append(
                                    {
                                        "score": score_name,
                                        "mean_id": float("nan"),
                                        "mean_ood": float("nan"),
                                        "auroc": float("nan"),
                                        "aupr": float("nan"),
                                    }
                                )
                                continue

                            values_valid = values[valid_entries]
                            targets_valid = binary_targets_mb[valid_entries]
                            id_mask_valid = id_mask_mb[valid_entries]
                            ood_mask_valid = ood_mask_mb[valid_entries]

                            id_mean = float(np.mean(values_valid[id_mask_valid])) if id_mask_valid.any() else float("nan")
                            ood_mean = float(np.mean(values_valid[ood_mask_valid])) if ood_mask_valid.any() else float("nan")
                            try:
                                auroc = roc_auc_score(targets_valid, values_valid)
                            except ValueError:
                                auroc = float("nan")
                            try:
                                aupr = average_precision_score(targets_valid, values_valid)
                            except ValueError:
                                aupr = float("nan")

                            mini_metrics_rows.append(
                                {
                                    "score": score_name,
                                    "mean_id": id_mean,
                                    "mean_ood": ood_mean,
                                    "auroc": float(auroc),
                                    "aupr": float(aupr),
                                }
                            )

                        hazard_columns_mb = [
                            "HazardDev_OS",
                            "HazardDev_LFFS",
                            "HazardDev_RFFS",
                            "HazardDev_DFFS",
                            "HazardDevPos_OS",
                            "HazardDevPos_LFFS",
                            "HazardDevPos_RFFS",
                            "HazardDevPos_DFFS",
                        ]

                        available_mb_cols = [col for col in hazard_columns_mb if col in aggregated_df.columns]
                        if len(available_mb_cols) >= 2 and id_mask_mb.any() and ood_mask_mb.any():
                            mb_features = aggregated_df[available_mb_cols].to_numpy()
                            id_mb_feats = mb_features[id_mask_mb]
                            ood_mb_feats = mb_features[ood_mask_mb]
                            if id_mb_feats.size and ood_mb_feats.size:
                                diff_mb = ood_mb_feats.mean(axis=0) - id_mb_feats.mean(axis=0)
                                cov_mb = np.cov(mb_features, rowvar=False)
                                cov_mb += np.eye(cov_mb.shape[0]) * 1e-6
                                inv_cov_mb = np.linalg.pinv(cov_mb)
                                lda_weights_mb = inv_cov_mb @ diff_mb
                                hazard_dev_mb_scores = mb_features @ lda_weights_mb
                                hazard_dev_mb_labels = binary_targets_mb
                                try:
                                    lda_auroc_mb = roc_auc_score(binary_targets_mb, hazard_dev_mb_scores)
                                except ValueError:
                                    lda_auroc_mb = float("nan")
                                try:
                                    lda_aupr_mb = average_precision_score(binary_targets_mb, hazard_dev_mb_scores)
                                except ValueError:
                                    lda_aupr_mb = float("nan")

                                aggregated_df["HazardDev_LDA"] = hazard_dev_mb_scores

                                mini_metrics_rows.append(
                                    {
                                        "score": "HazardDev_LDA",
                                        "mean_id": float(np.mean(hazard_dev_mb_scores[id_mask_mb])) if id_mask_mb.any() else float("nan"),
                                        "mean_ood": float(np.mean(hazard_dev_mb_scores[ood_mask_mb])) if ood_mask_mb.any() else float("nan"),
                                        "auroc": float(lda_auroc_mb),
                                        "aupr": float(lda_aupr_mb),
                                    }
                                )

                        mini_batch_metrics_df = pd.DataFrame(mini_metrics_rows)
                        mini_batch_metrics_df.to_csv("Test_Mixed_OOD_mini_batch_metrics.csv", index=False)

                        if id_mask_mb.any() and ood_mask_mb.any():
                            mini_split_arrays = {
                                "Test_Entropy_OS_MiniBatch": aggregated_df["Entropy_OS"].to_numpy(),
                                "Test_Entropy_LFFS_MiniBatch": aggregated_df["Entropy_LFFS"].to_numpy(),
                                "Test_Entropy_RFFS_MiniBatch": aggregated_df["Entropy_RFFS"].to_numpy(),
                                "Test_Entropy_DFFS_MiniBatch": aggregated_df["Entropy_DFFS"].to_numpy(),
                                "Test_Energy_OS_MiniBatch": aggregated_df["Energy_OS"].to_numpy(),
                                "Test_Energy_LFFS_MiniBatch": aggregated_df["Energy_LFFS"].to_numpy(),
                                "Test_Energy_RFFS_MiniBatch": aggregated_df["Energy_RFFS"].to_numpy(),
                                "Test_Energy_DFFS_MiniBatch": aggregated_df["Energy_DFFS"].to_numpy(),
                                "Test_MCD_OS_MiniBatch": aggregated_df["MCD_OS"].to_numpy(),
                                "Test_MCD_LFFS_MiniBatch": aggregated_df["MCD_LFFS"].to_numpy(),
                                "Test_MCD_RFFS_MiniBatch": aggregated_df["MCD_RFFS"].to_numpy(),
                                "Test_MCD_DFFS_MiniBatch": aggregated_df["MCD_DFFS"].to_numpy(),
                                "Test_Sep_Entropy_OS_MiniBatch": aggregated_df["Sep_Entropy_OS"].to_numpy(),
                                "Test_Sep_Entropy_LFFS_MiniBatch": aggregated_df["Sep_Entropy_LFFS"].to_numpy(),
                                "Test_Sep_Entropy_RFFS_MiniBatch": aggregated_df["Sep_Entropy_RFFS"].to_numpy(),
                                "Test_Sep_Entropy_DFFS_MiniBatch": aggregated_df["Sep_Entropy_DFFS"].to_numpy(),
                                "Test_Sep_Energy_OS_MiniBatch": aggregated_df["Sep_Energy_OS"].to_numpy(),
                                "Test_Sep_Energy_LFFS_MiniBatch": aggregated_df["Sep_Energy_LFFS"].to_numpy(),
                                "Test_Sep_Energy_RFFS_MiniBatch": aggregated_df["Sep_Energy_RFFS"].to_numpy(),
                                "Test_Sep_Energy_DFFS_MiniBatch": aggregated_df["Sep_Energy_DFFS"].to_numpy(),
                            }

                            if hazard_dev_dict:
                                hazard_mb_mappings = {
                                    "HazardDev_OS": "Test_HazardDeviation_OS_MiniBatch",
                                    "HazardDev_LFFS": "Test_HazardDeviation_LFFS_MiniBatch",
                                    "HazardDev_RFFS": "Test_HazardDeviation_RFFS_MiniBatch",
                                    "HazardDev_DFFS": "Test_HazardDeviation_DFFS_MiniBatch",
                                    "HazardDevPos_OS": "Test_HazardDeviationPos_OS_MiniBatch",
                                    "HazardDevPos_LFFS": "Test_HazardDeviationPos_LFFS_MiniBatch",
                                    "HazardDevPos_RFFS": "Test_HazardDeviationPos_RFFS_MiniBatch",
                                    "HazardDevPos_DFFS": "Test_HazardDeviationPos_DFFS_MiniBatch",
                                }
                                for col_name, export_name in hazard_mb_mappings.items():
                                    if col_name in aggregated_df.columns:
                                        mini_split_arrays[export_name] = aggregated_df[col_name].to_numpy()
                                if hazard_dev_mb_scores is not None:
                                    mini_split_arrays["Test_HazardDeviation_LDA_MiniBatch"] = hazard_dev_mb_scores

                            for disp_name in [
                                "HazardDispersion_OS",
                                "HazardDispersion_LFFS",
                                "HazardDispersion_RFFS",
                                "HazardDispersion_DFFS",
                            ]:
                                if disp_name in aggregated_df.columns:
                                    mini_split_arrays[f"Test_{disp_name}_MiniBatch"] = aggregated_df[disp_name].to_numpy()

                            for prefix, values in mini_split_arrays.items():
                                np.save(f"{prefix}_ID.npy", values[id_mask_mb])
                                np.save(f"{prefix}_OOD.npy", values[ood_mask_mb])
                        else:
                            log.warning("Unable to create mini-batch ID/OOD split files due to missing domain groups.")
                    else:
                        log.warning("Mini-batch identifiers present but none passed validation; skipping mini-batch metrics.")

        entropy_results.to_csv("Test_Entropy.csv", index=False)
        
        np.save("Test_MCD_OS.npy", mcd_scores1_np)
        np.save("Test_MCD_LFFS.npy", mcd_scores2_np)
        np.save("Test_MCD_RFFS.npy", mcd_scores3_np)
        np.save("Test_MCD_DFFS.npy", mcd_scores4_np)

        np.save("Test_Sep_Entropy_OS.npy", entropy_sep1_np)
        np.save("Test_Sep_Entropy_LFFS.npy", entropy_sep2_np)
        np.save("Test_Sep_Entropy_RFFS.npy", entropy_sep3_np)
        np.save("Test_Sep_Entropy_DFFS.npy", entropy_sep4_np)

        np.save("Test_Sep_Energy_OS.npy", energy_sep1_np)
        np.save("Test_Sep_Energy_LFFS.npy", energy_sep2_np)
        np.save("Test_Sep_Energy_RFFS.npy", energy_sep3_np)
        np.save("Test_Sep_Energy_DFFS.npy", energy_sep4_np)

        np.save("Test_MTLR_Logits_OS.npy", pred_prob1_np)
        np.save("Test_MTLR_Logits_LFFS.npy", pred_prob2_np)
        np.save("Test_MTLR_Logits_RFFS.npy", pred_prob3_np)
        np.save("Test_MTLR_Logits_DFFS.npy", pred_prob4_np)

        np.save("Test_Separate_Logits_OS.npy", separate_logits1_np)
        np.save("Test_Separate_Logits_LFFS.npy", separate_logits2_np)
        np.save("Test_Separate_Logits_RFFS.npy", separate_logits3_np)
        np.save("Test_Separate_Logits_DFFS.npy", separate_logits4_np)

        np.save("Test_Risk_OS.npy", pred_risk1.astype(np.float32))
        np.save("Test_Risk_LFFS.npy", pred_risk2.astype(np.float32))
        np.save("Test_Risk_RFFS.npy", pred_risk3.astype(np.float32))
        np.save("Test_Risk_DFFS.npy", pred_risk4.astype(np.float32))

        np.save("Test_Hazard_OS.npy", pred_hazard1.astype(np.float32))
        np.save("Test_Hazard_LFFS.npy", pred_hazard2.astype(np.float32))
        np.save("Test_Hazard_RFFS.npy", pred_hazard3.astype(np.float32))
        np.save("Test_Hazard_DFFS.npy", pred_hazard4.astype(np.float32))

        PatientID  = [x['labels']['ID'] for x in outputs] 
        PatientID = sum(PatientID, []) #inefficient way to flatten a list


        results = pd.DataFrame({'ID':PatientID, 'OS_risk':pred_risk1, 'OS':true_time1,'Death':true_event1,\
                                'LFFS_risk':pred_risk2, 'LFFS':true_time2,'LF':true_event2,\
                                'RFFS_risk':pred_risk3, 'RFFS':true_time3,'RF':true_event3,\
                                'DFFS_risk':pred_risk4, 'DFFS':true_time4,'DF':true_event4})
        results.to_csv('Test_Predictions.csv')

        return {"loss": loss, "OS-CI": ci_event1, "LFFS-CI": ci_event2, "RFFS-CI": ci_event3, "DFFS-CI": ci_event4,
                "entropy1": entropy1, "entropy2": entropy2, "entropy3": entropy3, "entropy4": entropy4,
                "mcd1": mcd_scores1, "mcd2": mcd_scores2, "mcd3": mcd_scores3, "mcd4": mcd_scores4,
                "sep_entropy1": entropy_sep1, "sep_entropy2": entropy_sep2, "sep_entropy3": entropy_sep3, "sep_entropy4": entropy_sep4,
                "sep_energy1": energy_sep1, "sep_energy2": energy_sep2, "sep_energy3": energy_sep3, "sep_energy4": energy_sep4}
   

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

def pysurvival_mtlr_loss(model, X_cens, X_uncens, Y_cens, Y_uncens, 
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
