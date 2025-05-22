# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
from sklearn.metrics import top_k_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import scipy.stats as stats
from itertools import combinations

logger = logging.getLogger(__name__)

def log_once(msg):
    if not hasattr(log_once, "_has_logged"):
        logger.info(msg)
        log_once._has_logged = True


def calculate_bedroc(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """
    
        # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    #print(scores.shape, y_true.shape)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:,0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    return bedroc

@register_loss("decoder_loss")
class DecoderLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=None,
            fix_encoder=fix_encoder
        )
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        targets = sample["net_input"]["selfie_tokens"]
        sample_size = targets.size(0)
        
        lprobs = net_output[:,:,:targets.shape[-1]]
        if not self.training:
            logging_output = {
                "loss": loss.data,
                "prob": lprobs.data,
                "target": targets.data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = net_output
        targets = sample["net_input"]["selfie_tokens"]
        lprobs = lprobs[:,:,:targets.shape[-1]]
        # print("111", lprobs.shape, targets.shape, sample["target"]["finetune_target"].shape)
        nll_loss = F.nll_loss(
            lprobs,
            targets,
            reduction="sum" if reduce else "none",
        ) / lprobs.shape[-1]

        loss =  nll_loss 
        #print(loss.data, nll_loss.data, kld_loss.data)
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        # if split == "valid":
        #     print("hi1")
        loss = sum(log.get("loss", 0).float() for log in logging_outputs)
        # if split == "valid":
        #     print("hi2")
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # if split == "valid":
        #     print("hi3")
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss_all", loss / sample_size / math.log(2), sample_size, round=3
        )

        if "valid" in split or "test" in split:
            
            prob_list = []
            pred_list = []
            target_list = []
            for log in logging_outputs:
                prob = log.get("prob")
                prob = torch.transpose(prob, 1, 2)
                prob = prob.reshape((-1, prob.shape[-1]))
                prob_list.append(prob)
                pred = log.get("prob").argmax(dim=1)
                pred = pred.flatten()
                pred_list.append(pred)
                target = log.get("target")
                target = target.flatten()
                target_list.append(target)

            preds = torch.cat(pred_list, dim=0)
            targets = torch.cat(target_list, dim=0)
            #print(preds.shape, targets.shape)
            acc = (preds == targets).float().mean(dim=-1)
            #print(acc.shape)
            metrics.log_scalar(
                f"{split}_acc", acc , sample_size, round=3
            )
            
        
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("decoder_vae_loss")
class DecoderVAELoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, candidate_reps, candidate_embs, candidate_smiles, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            candidate_reps=candidate_reps,
            candidate_embs=candidate_embs,
            candidate_smiles=candidate_smiles,
            features_only=True,
            classification_head_name=None,
            fix_encoder=fix_encoder
        )
        loss, nll_loss, kld_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        targets = sample["net_input"]["selfie_tokens"]
        sample_size = targets.size(0)
        
        lprobs = net_output[0][:,:,:targets.shape[-1]]
        if not self.training:
            logging_output = {
                "loss": loss.data,
                "kld_loss": kld_loss.data,
                "nll_loss": nll_loss.data,
                "prob": lprobs.data,
                "target": targets.data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "kld_loss": kld_loss.data,
                "nll_loss": nll_loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        out, z, mu, log_var = net_output
        lprobs = out
        targets = sample["net_input"]["selfie_tokens"]
        lprobs = lprobs[:,:,:targets.shape[-1]]
        # print("111", lprobs.shape, targets.shape, sample["target"]["finetune_target"].shape)
        nll_loss = F.nll_loss(
            lprobs,
            targets,
            reduction="sum" if reduce else "none",
        ) / lprobs.shape[-1]
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / lprobs.shape[1]
        p=0.2
        loss = p * kld_loss + (1-p) * nll_loss 
        #print(loss.data, nll_loss.data, kld_loss.data)
        return loss, nll_loss, kld_loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        # if split == "valid":
        #     print("hi1")
        loss_all = sum(log.get("loss", 0).float() for log in logging_outputs)
        loss_kld = sum(log.get("kld_loss", 0).float() for log in logging_outputs)
        loss_nll = sum(log.get("nll_loss", 0).float() for log in logging_outputs)
        # if split == "valid":
        #     print("hi2")
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # if split == "valid":
        #     print("hi3")
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss_all", loss_all / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_kld", loss_kld / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "loss_nll", loss_nll / sample_size, sample_size, round=3
        )
        if "valid" in split or "test" in split:
            
            prob_list = []
            pred_list = []
            target_list = []
            for log in logging_outputs:
                prob = log.get("prob")
                prob = torch.transpose(prob, 1, 2)
                prob = prob.reshape((-1, prob.shape[-1]))
                prob_list.append(prob)
                pred = log.get("prob").argmax(dim=1)
                pred = pred.flatten()
                pred_list.append(pred)
                target = log.get("target")
                target = target.flatten()
                target_list.append(target)

            probs = torch.cat(prob_list, dim=0)
            preds = torch.cat(pred_list, dim=0)
            targets = torch.cat(target_list, dim=0)
            #print(preds.shape, targets.shape)
            acc = (preds == targets).float().mean(dim=-1)
            #print(acc.shape)
            metrics.log_scalar(
                f"{split}_acc", acc , sample_size, round=3
            )
            '''
            # smi_list = [
            #     item for log in logging_outputs for item in log.get("smi_name")
            # ]
            probs = torch.exp(probs)
            #prob_flat = prob_flat.reshape((-1, prob_flat.shape[-1]))
            print(probs.shape)

            #targets = targets.squeeze(dim=-1)
            auc = roc_auc_score(targets.cpu(), probs.cpu(), multi_class="ovo", labels=torch.arange(probs.shape[-1]))
            #df = df.groupby("smi").mean()
            #agg_auc = roc_auc_score(df["targets"], df["probs"])
            agg_auc = auc
            
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)
            '''
        
    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("finetune_cross_entropy")
class FinetuneCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
            fix_encoder=fix_encoder
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = F.log_softmax(net_output.float(), dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        targets = sample["target"]["finetune_target"].view(-1)
        # print("111", lprobs.shape, targets.shape, sample["target"]["finetune_target"].shape)
        loss = F.nll_loss(
            lprobs,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            probs = torch.cat([log.get("prob") for log in logging_outputs], dim=0)
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            if probs.size(-1) == 2:
                # binary classification task, add auc score
                targets = torch.cat(
                    [log.get("target", 0) for log in logging_outputs], dim=0
                )
                smi_list = [
                    item for log in logging_outputs for item in log.get("smi_name")
                ]
                df = pd.DataFrame(
                    {
                        "probs": probs[:, 1].cpu(),
                        "targets": targets.cpu(),
                        "smi": smi_list,
                    }
                )
                auc = roc_auc_score(df["targets"], df["probs"])
                df = df.groupby("smi").mean()
                agg_auc = roc_auc_score(df["targets"], df["probs"])
                metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
                metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("ce")
class CEntropyLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            smi_list = sample["smi_name"],
            pocket_list = sample["pocket_name"],
            features_only=True,
            fix_encoder=fix_encoder
        )
        logit_output = net_output
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        #print(sample["target"]["finetune_target"])
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = torch.sigmoid(logit_output.float())
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):

        targets = sample["target"]["finetune_target"].view(-1)
        # print("111", lprobs.shape, targets.shape, sample["target"]["finetune_target"].shape)
        #print(net_output.shape, targets.shape)
        loss = F.binary_cross_entropy_with_logits(
            net_output.float(),
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:

            # get acc
            acc_sum = 0
            for log in logging_outputs:
                probs = torch.sigmoid(log.get("prob"))
                targets = log.get("target")
                probs = probs > 0.5
                # convert to int
                probs = probs.long()
            
            
            logs = [log.get("prob") for log in logging_outputs]
            targets = torch.cat(
                [log.get("target", 0) for log in logging_outputs], dim=0
            )
            probs = torch.cat([log.get("prob") for log in logging_outputs], dim=0)
            #probs = torch.sigmoid(probs)
            print(probs.shape, targets.shape)
            print(probs[:10], targets[:10])
            preds = probs > 0.5
            # convert to int
            preds = preds.long()
            acc_sum = (preds == targets).sum()
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            # binary classification task, add auc score
            
            
            smi_list = [
                item for log in logging_outputs for item in log.get("smi_name")
            ]
            df = pd.DataFrame(
                {
                    "probs": probs.cpu(),
                    "targets": targets.cpu(),
                    "smi": smi_list,
                }
            )
            # get values of df["targets"]
            # 
            
            auc = roc_auc_score(df["targets"].values, df["probs"].values)
            df = df.groupby("smi").mean()
            agg_auc = roc_auc_score(df["targets"], df["probs"])
            #print(df["targets"])
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("in_batch_softmax")
class IBSLoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        inputs = [
            (sample['net_input']['mol_src_tokens'], sample['net_input']['mol_src_distance'], sample['net_input']['mol_src_edge_type']),
            (sample['net_input']['pocket_src_tokens'], sample['net_input']['pocket_src_distance'], sample['net_input']['pocket_src_edge_type']),
        ]
        components = ["mol", "pocket"]
        if "fpocket_src_tokens" in sample["net_input"].keys():
            inputs.append((sample["net_input"]["fpocket_src_tokens"], sample["net_input"]["fpocket_src_distance"], sample["net_input"]["fpocket_src_edge_type"]))
            components.append("pocket")
        outputs, logit_param = model(inputs, components)
        if "fpocket_src_tokens" in sample["net_input"].keys():
            mol_emb, pocket_emb, fpocket_emb = outputs
        else:
            mol_emb, pocket_emb = outputs
        sample_size = mol_emb.size(0)
        loss = self.compute_loss(model, mol_emb, pocket_emb, reduce=reduce)
        if "fpocket_src_tokens" in sample["net_input"].keys():
            fpocket_padding_mask = sample["net_input"]["fpocket_src_tokens"].ne(model.pocket_model.padding_idx).sum(-1).eq(2)
            fpocket_neg_mask = ~ sample["net_input"]["fpocket_mask"]
            loss = loss + self.compute_loss(model, fpocket_emb, mol_emb, pocket_padding_mask=fpocket_padding_mask, pocket_neg_mask=fpocket_neg_mask, reduce=reduce,)
            loss = loss + self.compute_loss(model, fpocket_emb, pocket_emb, pocket_padding_mask=fpocket_padding_mask, pocket_neg_mask=fpocket_neg_mask, reduce=reduce,)
        targets = torch.arange(sample_size, dtype=torch.long).cuda()
        # affinities = sample["target"]["finetune_target"].view(-1)
        if not self.training:
            logit_output = (torch.matmul(pocket_emb, mol_emb.transpose(0, 1))) * model.logit_scale.exp().detach()
            logit_output = logit_output[:,:sample_size]
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": targets,
                # "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": targets.size(0),
                # "scale": net_output[1].data,
                # "affinity": affinities,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": targets.size(0),
                # "scale": net_output[1].data
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, pocket_emb, mol_emb, pocket_padding_mask=None, mol_padding_mask=None, pocket_neg_mask=None, mol_neg_mask=None, reduce=True):
        mol_padding_mask = mol_padding_mask if mol_padding_mask is not None else torch.zeros(mol_emb.size(0)).bool().cuda()
        pocket_padding_mask = pocket_padding_mask if pocket_padding_mask is not None else torch.zeros(pocket_emb.size(0)).bool().cuda()
        net_output = torch.matmul(pocket_emb, mol_emb.transpose(0, 1)) * model.logit_scale.exp().detach()
        net_output = net_output.masked_fill(pocket_padding_mask.unsqueeze(1), -6e4)
        net_output = net_output.masked_fill(mol_padding_mask.unsqueeze(0), -6e4)
        lprobs_pocket = F.log_softmax(net_output.float(), dim=-1)
        lprobs_pocket = lprobs_pocket.view(-1, lprobs_pocket.size(-1))
        sample_size = lprobs_pocket.size(0)
        targets= torch.arange(sample_size, dtype=torch.long).view(-1).cuda()
        pocket_neg_mask = pocket_neg_mask if pocket_neg_mask is not None else torch.zeros(pocket_emb.size(0)).bool().cuda()
        mol_neg_mask = mol_neg_mask if mol_neg_mask is not None else torch.zeros(mol_emb.size(0)).bool().cuda()
        valid_mask = pocket_padding_mask | mol_padding_mask | pocket_neg_mask | mol_neg_mask

        # pocket retrieve mol
        loss_pocket = F.nll_loss(
            lprobs_pocket,
            targets,
            reduction="none",
        )
        loss_pocket = loss_pocket.masked_fill(valid_mask, 0)
        if reduce:
            loss_pocket = loss_pocket.sum()
        
        lprobs_mol = F.log_softmax(torch.transpose(net_output.float(), 0, 1), dim=-1)
        lprobs_mol = lprobs_mol.view(-1, lprobs_mol.size(-1))
        lprobs_mol = lprobs_mol[:sample_size]

        # mol retrieve pocket
        loss_mol = F.nll_loss(
            lprobs_mol,
            targets,
            reduction="none",
        )
        if pocket_padding_mask is not None:  # don't retrieve padding pocket
            loss_mol = loss_mol.masked_fill(valid_mask, 0)
        if reduce:
            loss_mol = loss_mol.sum()

        loss = 0.5 * loss_pocket + 0.5 * loss_mol
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        # metrics.log_scalar("scale", logging_outputs[0].get("scale"), round=3)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            
            prob_list = []
            if len(logging_outputs) == 1:
                prob_list.append(logging_outputs[0].get("prob"))
            else:
                for i in range(len(logging_outputs)-1):
                    prob_list.append(logging_outputs[i].get("prob"))
            probs = torch.cat(prob_list, dim=0)
            
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )

            metrics.log_scalar(
                "valid_neg_loss", -loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            
            
            targets = torch.cat(
                [log.get("target", 0) for log in logging_outputs], dim=0
            )

            targets = targets[:len(probs)]
            bedroc_list = []
            auc_list = []
            for i in range(len(probs)):
                prob = probs[i]
                target = targets[i]
                label = torch.zeros_like(prob)
                label[target]=1.0
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
                bedroc = calculate_bedroc(label.cpu(), prob.cpu(), 80.5)
                bedroc_list.append(bedroc)
            bedroc = np.mean(bedroc_list)
            auc = np.mean(auc_list)
            
            top_k_acc = top_k_accuracy_score(targets.cpu(), probs.cpu(), k=3, normalize=True)
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_bedroc", bedroc, sample_size, round=3)
            metrics.log_scalar(f"{split}_top3_acc", top_k_acc, sample_size, round=3)

            

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("siglip")
class SigmoidLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    @staticmethod
    def get_logits(pocket_emb, mol_emb, logit_scale, logit_bias=None):
        if pocket_emb.squeeze().dim() == 3:
            logits = logit_scale * (pocket_emb * mol_emb.unsqueeze(0)).sum(dim=-1)
        else:
            logits = logit_scale * (pocket_emb @ mol_emb.T)
        if logit_bias is not None:
            logits += logit_bias
        return logits

    @staticmethod
    def get_labels(device, dtype, num_logits, pocket_padding_mask=None, mol_padding_mask=None, pocket_neg_mask=None, mol_neg_mask=None, graph_mask=None, pocket_id=None, mol_id=None,):
        labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) - torch.ones(num_logits, device=device, dtype=dtype)
        if pocket_padding_mask is not None:
            labels = labels.masked_fill_(pocket_padding_mask.unsqueeze(1), 0)
        if mol_padding_mask is not None:
            labels = labels.masked_fill_(mol_padding_mask.unsqueeze(0), 0)
        if pocket_neg_mask is not None:
            labels = labels.masked_fill_(pocket_neg_mask.unsqueeze(1), -1)
        if mol_neg_mask is not None:
            labels = labels.masked_fill_(mol_neg_mask.unsqueeze(0), -1)
        if graph_mask is not None:
            labels = labels.masked_fill_(graph_mask, 0)
        if pocket_id:
            pocket_id = np.array(pocket_id, dtype=str)
            pocket_id = np.expand_dims(pocket_id, 1)
            matrix1 = np.repeat(pocket_id, len(pocket_id), 1)
            matrix2 = np.repeat(np.transpose(pocket_id), len(pocket_id), 0)
            pocket_duplicate_matrix = matrix1==matrix2
            # replace diagonal
            pocket_duplicate_matrix = pocket_duplicate_matrix - np.eye(len(pocket_id))
            pocket_duplicate_matrix = torch.tensor(pocket_duplicate_matrix, device=labels.device, dtype=torch.bool)
            labels.masked_fill_(pocket_duplicate_matrix, 0)
        if mol_id:
            mol_id = np.array(mol_id, dtype=str)
            mol_id = np.expand_dims(mol_id, 1)
            matrix1 = np.repeat(mol_id, len(mol_id), 1)
            matrix2 = np.repeat(np.transpose(mol_id), len(mol_id), 0)
            mol_duplicate_matrix = matrix1==matrix2
            # replace diagonal
            mol_duplicate_matrix = mol_duplicate_matrix - np.eye(len(mol_id))
            mol_duplicate_matrix = torch.tensor(mol_duplicate_matrix, device=labels.device, dtype=torch.bool)
            labels.masked_fill_(mol_duplicate_matrix, 0)
        return labels

    def compute_loss(self, logit_param, pocket_emb, mol_emb, pocket_padding_mask=None, mol_padding_mask=None, pocket_neg_mask=None, mol_neg_mask=None, graph_mask=None, pocket_id=None, mol_id=None, reduce=True):
        logits = self.get_logits(pocket_emb, mol_emb, logit_param["logit_scale"], logit_param["logit_bias"])
        labels = self.get_labels(pocket_emb.device, pocket_emb.dtype, pocket_emb.shape[0], pocket_padding_mask, mol_padding_mask, pocket_neg_mask, mol_neg_mask, graph_mask=graph_mask, pocket_id=pocket_id, mol_id=mol_id,)
        loss = -F.logsigmoid(labels * logits)
        if reduce:
            # loss = loss.sum() / ((~ pocket_padding_mask).sum() if pocket_padding_mask is not None else pocket_emb.shape[0])
            loss = loss.sum()
        return loss

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if "mol_id" in sample.keys():
            mol_id = sample["mol_id"]
            protein_id = sample["protein_id"]
        else:
            mol_id, protein_id = None, None

        inputs = [
            (sample['net_input']['mol_src_tokens'], sample['net_input']['mol_src_distance'], sample['net_input']['mol_src_edge_type']),
            (sample['net_input']['pocket_src_tokens'], sample['net_input']['pocket_src_distance'], sample['net_input']['pocket_src_edge_type']),
        ]
        components = ["mol", "pocket"]
        if "fpocket_src_tokens" in sample["net_input"].keys():
            inputs.append((sample["net_input"]["fpocket_src_tokens"], sample["net_input"]["fpocket_src_distance"], sample["net_input"]["fpocket_src_edge_type"]))
            components.append("pocket")
        outputs, logit_param = model(inputs, components)
        if "fpocket_src_tokens" in sample["net_input"].keys():
            mol_emb, pocket_emb, fpocket_emb = outputs
        else:
            mol_emb, pocket_emb = outputs
        sample_size = mol_emb.size(0)
        loss = self.compute_loss(logit_param, pocket_emb, mol_emb, reduce=reduce, pocket_id=protein_id, mol_id=mol_id)
        if "fpocket_src_tokens" in sample["net_input"].keys():
            fpocket_padding_mask = sample["net_input"]["fpocket_src_tokens"].ne(model.pocket_model.padding_idx).sum(-1).eq(2)
            fpocket_neg_mask = ~ sample["net_input"]["fpocket_mask"]
            loss = loss + self.compute_loss(logit_param, fpocket_emb, mol_emb, pocket_padding_mask=fpocket_padding_mask, pocket_neg_mask=fpocket_neg_mask, reduce=reduce, pocket_id=protein_id, mol_id=mol_id)
            loss = loss + self.compute_loss(logit_param, fpocket_emb, pocket_emb, pocket_padding_mask=fpocket_padding_mask, pocket_neg_mask=fpocket_neg_mask, reduce=reduce, pocket_id=protein_id, mol_id=protein_id)
        targets = torch.arange(sample_size, dtype=torch.long).cuda()
        # affinities = sample["target"]["finetune_target"].view(-1)
        if not self.training:
            logit_output = self.get_logits(pocket_emb, mol_emb, logit_param["logit_scale"], logit_param["logit_bias"])
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": targets,
                # "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data,
                # "affinity": affinities,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data
            }
        logging_output["logit_bias"] = logit_param["logit_bias"].detach() if "logit_bias" in logit_param.keys() else None
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        # metrics.log_scalar("scale", logging_outputs[0].get("scale"), round=3)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            
            prob_list = []
            if len(logging_outputs) == 1:
                prob_list.append(logging_outputs[0].get("prob"))
            else:
                for i in range(len(logging_outputs)-1):
                    prob_list.append(logging_outputs[i].get("prob"))
            probs = torch.cat(prob_list, dim=0)
            
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )

            metrics.log_scalar(
                f"{split}_neg_loss", -loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            
            
            targets = torch.cat(
                [log.get("target", 0) for log in logging_outputs], dim=0
            )

            targets = targets[:len(probs)]
            bedroc_list = []
            auc_list = []
            for i in range(len(probs)):
                prob = probs[i]
                target = targets[i]
                label = torch.zeros_like(prob)
                label[target]=1.0
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
                bedroc = calculate_bedroc(label.cpu(), prob.cpu(), 80.5)
                bedroc_list.append(bedroc)
            bedroc = np.mean(bedroc_list)
            auc = np.mean(auc_list)
            
            top_k_acc = top_k_accuracy_score(targets.cpu(), probs.cpu(), k=3, normalize=True)
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_bedroc", bedroc, sample_size, round=3)
            metrics.log_scalar(f"{split}_top3_acc", top_k_acc, sample_size, round=3)
            metrics.log_scalar(f"{split}_logit_scale", logging_outputs[0].get("logit_scale"), round=3)
            if "logit_bias" in logging_outputs[0]:
                metrics.log_scalar(f"{split}_logit_bias", logging_outputs[0].get("logit_bias"), round=3)


@register_loss("siglip_ft")
class SigmoidLossFT(SigmoidLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if "mol_id" in sample.keys():
            mol_id = sample["mol_id"]
            protein_id = sample["protein_id"]
        else:
            mol_id, protein_id = None, None
        if "mol_embs" not in sample.keys():
            mol_emb, pocket_emb, logit_param = model(
                mol_inputs=(sample['net_input']['mol_src_tokens'], sample['net_input']['mol_src_distance'], sample['net_input']['mol_src_edge_type']),
                pocket_rep=sample.get("pocket_rep"), pocket_embs=sample.get("pocket_embs"), return_attn=False
            )
        else:
            mol_emb, pocket_emb, logit_param = model(
                mol_embs=sample["mol_embs"],
                pocket_rep=sample.get("pocket_rep"), pocket_embs=sample.get("pocket_embs"), return_attn=False
            )
        sample_size = mol_emb.size(0)
        if "pocket_edge" in sample.keys():
            pocket_edge = sample["pocket_edge"]  # list of set
            graph_mask = torch.zeros((sample_size, sample_size), dtype=torch.bool).cuda()
            for i, edge in enumerate(pocket_edge):
                for j, mid in enumerate(mol_id):
                    if i != j and mid in edge:
                        graph_mask[i, j] = True
            if graph_mask.to(torch.int).sum() == 0:
                graph_mask = None
            else:
                logging.debug(f"graph mask: {graph_mask.to(torch.int).sum()}")
        else:
            graph_mask = None
        loss = self.compute_loss(logit_param, pocket_emb, mol_emb, reduce=reduce, graph_mask=graph_mask, pocket_id=protein_id, mol_id=mol_id)
        targets = torch.arange(sample_size, dtype=torch.long).cuda()
        # affinities = sample["target"]["finetune_target"].view(-1)
        if not self.training:
            logit_output = self.get_logits(pocket_emb, mol_emb, logit_param["logit_scale"], logit_param["logit_bias"])
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": targets,
                # "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data,
                # "affinity": affinities,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data
            }
        logging_output["logit_bias"] = logit_param["logit_bias"].detach() if "logit_bias" in logit_param.keys() else None
        return loss, sample_size, logging_output


@register_loss("siglip_ft_attn_kl")
class SigmoidLossFT_AttnKL(SigmoidLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if "mol_id" in sample.keys():
            mol_id = sample["mol_id"]
            protein_id = sample["protein_id"]
        else:
            mol_id, protein_id = None, None
        if "mol_embs" not in sample.keys():
            mol_emb, pocket_emb, logit_param = model(
                mol_inputs=(sample['net_input']['mol_src_tokens'], sample['net_input']['mol_src_distance'], sample['net_input']['mol_src_edge_type']),
                pocket_rep=sample.get("pocket_rep"), pocket_embs=sample.get("pocket_embs"), return_attn=True
            )
        else:
            mol_emb, pocket_emb, logit_param = model(
                mol_embs=sample["mol_embs"],
                pocket_rep=sample.get("pocket_rep"), pocket_embs=sample.get("pocket_embs"), return_attn=True
            )
        pocket_emb, attn = pocket_emb
        attn_diag = attn.squeeze()[torch.arange(mol_emb.size(0)), torch.arange(mol_emb.size(0))]
        # one hot for pdbbind sample, prior embedding for other samples
        attn_label = torch.zeros_like(attn_diag)
        attn_label[sample["lig_source"] == 0, 0] = 1
        ori_pocket_rep = sample["pocket_embs"][sample["lig_source"]]
        ori_mol_rep = sample["mol_embs"][sample["lig_source"]]
        log_once(f"ori_pocket_rep:{ori_pocket_rep.shape}, ori_mol_rep:{ori_mol_rep.shape}")
        attn_score = torch.einsum('bld,bd->bl', ori_pocket_rep, ori_mol_rep) * 5
        attn_score.masked_fill_(ori_pocket_rep.eq(0).all(dim=-1), -1e4)
        attn_score = F.softmax(attn_score, dim=-1)
        attn_label[sample["lig_source"] == 1] = attn_score.detach()
        # logger.info(attn_label)
        kl_loss = F.kl_div(attn_diag.clamp(min=1e-4).log(), attn_label.clamp(min=1e-4).log(), reduction="batchmean", log_target=True)
        sample_size = mol_emb.size(0)
        if "pocket_edge" in sample.keys():
            pocket_edge = sample["pocket_edge"]  # list of set
            graph_mask = torch.zeros((sample_size, sample_size), dtype=torch.bool).cuda()
            for i, edge in enumerate(pocket_edge):
                for j, mid in enumerate(mol_id):
                    if i != j and mid in edge:
                        graph_mask[i, j] = True
            if graph_mask.to(torch.int).sum() == 0:
                graph_mask = None
            else:
                logging.debug(f"graph mask: {graph_mask.to(torch.int).sum()}")
        else:
            graph_mask = None
        cl_loss = self.compute_loss(logit_param, pocket_emb, mol_emb, reduce=reduce, graph_mask=graph_mask, pocket_id=protein_id, mol_id=mol_id)
        loss = cl_loss + kl_loss * sample_size**2
        logger.info(f"cl loss: {loss}, kl loss: {kl_loss * sample_size}")
        targets = torch.arange(sample_size, dtype=torch.long).cuda()
        # affinities = sample["target"]["finetune_target"].view(-1)
        if not self.training:
            logit_output = self.get_logits(pocket_emb, mol_emb, logit_param["logit_scale"], logit_param["logit_bias"])
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "attn_loss": kl_loss.data,
                "prob": probs.data,
                "target": targets,
                # "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data,
                # "affinity": affinities,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "attn_loss": kl_loss.data,
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data
            }
        logging_output["logit_bias"] = logit_param["logit_bias"].detach() if "logit_bias" in logit_param.keys() else None
        return loss, sample_size, logging_output


@register_loss("siglip_ft_rmuncer")
class SigmoidLossFTRMUncer(SigmoidLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False, attn_threshold=0.5):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if "mol_id" in sample.keys():
            mol_id = sample["mol_id"]
            protein_id = sample["protein_id"]
        else:
            mol_id, protein_id = None, None

        mol_emb, pocket_emb, logit_param = model(mol_inputs=(sample['net_input']['mol_src_tokens'], sample['net_input']['mol_src_distance'], sample['net_input']['mol_src_edge_type']), pocket_rep=sample.get("pocket_rep"), pocket_embs=sample.get("pocket_embs"), return_attn=True)
        pocket_emb, attn = pocket_emb
        sample_size = mol_emb.size(0)
        # certain_mask = torch.zeros((sample_size, sample_size), dtype=torch.bool).cuda()
        # # print(f"attn: {attn}, attn.shape: {attn.shape}")
        # for i in range(sample_size):
        #     log_once(f"attn shape for one sample: {attn.squeeze()[i, i].shape}")
        #     if attn.squeeze()[i, i].max() < attn_threshold:
        #         certain_mask[i, i] = True
        attn_diag = attn.squeeze()[torch.arange(sample_size), torch.arange(sample_size)]
        max_attn_vals = attn_diag.max(dim=-1).values
        certain_topk_vals, certain_topk_indices = max_attn_vals.topk(int(sample_size/5), largest=True)
        logger.info(f"least confident samples: {certain_topk_vals.min()}")
        certain_mask = torch.zeros((sample_size, sample_size), dtype=torch.bool).cuda()
        for i in range(sample_size):
            if i in certain_topk_indices:
                certain_mask[i] = True
                certain_mask[:, i] = True
        # logger.info(f"certain_mask: {certain_mask.sum()}")
        if "pocket_edge" in sample.keys():
            pocket_edge = sample["pocket_edge"]  # list of set
            graph_mask = torch.zeros((sample_size, sample_size), dtype=torch.bool).cuda()
            for i, edge in enumerate(pocket_edge):
                for j, mid in enumerate(mol_id):
                    if i != j and mid in edge:
                        graph_mask[i, j] = True
            if graph_mask.to(torch.int).sum() == 0:
                graph_mask = None
            else:
                logging.debug(f"graph mask: {graph_mask.to(torch.int).sum()}")
        else:
            graph_mask = None
        graph_mask = certain_mask if graph_mask is None else graph_mask | certain_mask
        loss = self.compute_loss(logit_param, pocket_emb, mol_emb, reduce=reduce, graph_mask=graph_mask, pocket_id=protein_id, mol_id=mol_id)
        targets = torch.arange(sample_size, dtype=torch.long).cuda()
        # affinities = sample["target"]["finetune_target"].view(-1)
        if not self.training:
            logit_output = self.get_logits(pocket_emb, mol_emb, logit_param["logit_scale"], logit_param["logit_bias"])
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": targets,
                # "smi_name": sample["smi_name"],
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data,
                # "affinity": affinities,
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": targets.size(0),
                "logit_scale": logit_param["logit_scale"].detach(),
                # "scale": net_output[1].data
            }
        logging_output["logit_bias"] = logit_param["logit_bias"].detach() if "logit_bias" in logit_param.keys() else None
        return loss, sample_size, logging_output


@register_loss("multi_task_BCE")
class MultiTaskBCELoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            masked_tokens=None,
            features_only=True,
            classification_head_name=self.args.classification_head_name,
            fix_encoder=fix_encoder
        )
        logit_output = net_output[0]
        is_valid = sample["target"]["finetune_target"] > -0.5
        loss = self.compute_loss(
            model, logit_output, sample, reduce=reduce, is_valid=is_valid
        )
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = torch.sigmoid(logit_output.float()).view(-1, logit_output.size(-1))
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "num_task": self.args.num_classes,
                "sample_size": sample_size,
                "conf_size": self.args.conf_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, is_valid=None):
        pred = net_output[is_valid].float()
        targets = sample["target"]["finetune_target"][is_valid].float()
        loss = F.binary_cross_entropy_with_logits(
            pred,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            agg_auc_list = []
            num_task = logging_outputs[0].get("num_task", 0)
            conf_size = logging_outputs[0].get("conf_size", 0)
            y_true = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            y_pred = (
                torch.cat([log.get("prob") for log in logging_outputs], dim=0)
                .view(-1, conf_size, num_task)
                .cpu()
                .numpy()
                .mean(axis=1)
            )
            # [test_size, num_classes] = [test_size * conf_size, num_classes].mean(axis=1)
            for i in range(y_true.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                    # ignore nan values
                    is_labeled = y_true[:, i] > -0.5
                    agg_auc_list.append(
                        roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                    )
            if len(agg_auc_list) < y_true.shape[1]:
                warnings.warn("Some target is missing!")
            if len(agg_auc_list) == 0:
                raise RuntimeError(
                    "No positively labeled data available. Cannot compute Average Precision."
                )
            agg_auc = sum(agg_auc_list) / len(agg_auc_list)
            metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("BCE")
class BCELoss(CrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            smi_list = sample["smi_name"],
            pocket_list = sample["pocket_name"],
            features_only=True,
            fix_encoder=fix_encoder
        )
        logit_output = net_output
        loss = self.compute_loss(
            model, logit_output, sample, reduce=reduce
        )
        sample_size = sample["target"]["finetune_target"].size(0)

        if not self.training:
            probs = torch.sigmoid(logit_output.float())
            #print(probs.size())
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, is_valid=None):
        pred = net_output.float()
        targets = sample["target"]["finetune_target"].float()
        loss = F.binary_cross_entropy_with_logits(
            pred,
            targets,
            reduction="sum" if reduce else "none",
        )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            y_true_list = []
            y_pred_list = []
            y_true_list = [log.get("target", 0) for log in logging_outputs]
            y_pred_list = [log.get("prob") for log in logging_outputs]
            y_true = (
                torch.cat(y_true_list, dim=0)
                .cpu()
                .numpy()
            )
            y_pred = (
                torch.cat(y_pred_list, dim=0)
                .cpu()
                .numpy()
            )
            # [test_size, num_classes] = [test_size * conf_size, num_classes].mean(axis=1)

            auc = roc_auc_score(y_true , y_pred)
                    

            agg_auc = auc
            metrics.log_scalar(f"{split}_agg_auc", agg_auc, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

@register_loss("finetune_cross_entropy_pocket")
class FinetuneCrossEntropyPocketLoss(FinetuneCrossEntropyLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
        )
        logit_output = net_output[0]
        loss = self.compute_loss(model, logit_output, sample, reduce=reduce)
        sample_size = sample["target"]["finetune_target"].size(0)
        if not self.training:
            probs = F.softmax(logit_output.float(), dim=-1).view(
                -1, logit_output.size(-1)
            )
            logging_output = {
                "loss": loss.data,
                "prob": probs.data,
                "target": sample["target"]["finetune_target"].view(-1).data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        else:
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample["target"]["finetune_target"].size(0),
            }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if "valid" in split or "test" in split:
            acc_sum = sum(
                sum(log.get("prob").argmax(dim=-1) == log.get("target"))
                for log in logging_outputs
            )
            metrics.log_scalar(
                f"{split}_acc", acc_sum / sample_size, sample_size, round=3
            )
            preds = (
                torch.cat(
                    [log.get("prob").argmax(dim=-1) for log in logging_outputs], dim=0
                )
                .cpu()
                .numpy()
            )
            targets = (
                torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
                .cpu()
                .numpy()
            )
            metrics.log_scalar(f"{split}_pre", precision_score(targets, preds), round=3)
            metrics.log_scalar(f"{split}_rec", recall_score(targets, preds), round=3)
            metrics.log_scalar(
                f"{split}_f1", f1_score(targets, preds), sample_size, round=3
            )
