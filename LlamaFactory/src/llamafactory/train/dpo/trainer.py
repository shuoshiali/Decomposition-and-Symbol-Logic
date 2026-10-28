# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
from trl import DPOTrainer
from trl.models.utils import prepare_deepspeed, prepare_fsdp
from trl.trainer import disable_dropout_in_model
from typing_extensions import override

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler, get_batch_logps, nested_detach


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


def get_per_token_logps(
    logits: "torch.Tensor", labels: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    r"""Compute per-token log probabilities: logit[label] - logsumexp(logits).

    使用 gather + logsumexp，不产生 vocab 级中间张量（显存开销仅 [B, L-1]）。
    返回 (已掩码的逐 token logp [B, L-1], 掩码 [B, L-1])。
    """
    shifted = labels[:, 1:]
    mask = shifted != IGNORE_INDEX
    safe_labels = shifted.clamp(min=0)
    target = logits[:, :-1, :].gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    lse = torch.logsumexp(logits[:, :-1, :], dim=-1)
    per_token_logps = (target - lse) * mask
    return per_token_logps, mask


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.f_divergence_type = "reverse_kl"
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.bco_gemma = finetuning_args.pref_bco_weight
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma
        self.ld_alpha = finetuning_args.ld_alpha

        self._last_ref_output = None

        Trainer.__init__(self, model=model, **kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if self.bco_gemma >= 1e-6:
            from trl.trainer import RunningMoments

            self.running = RunningMoments(self.accelerator)

        # ===== DiReDPO：维度加权 DPO（从 finetuning_args 读取，默认关闭）=====
        self.use_dim_weighted = getattr(finetuning_args, "use_dim_weighted_dpo", False)
        self.dim_temperature = getattr(finetuning_args, "dim_weight_temperature", 2.0)
        self.dim_weight_normalize = getattr(finetuning_args, "dim_weight_normalize", True)
        if self.use_dim_weighted:
            print(
                f"[DiReDPO] Dimension-Weighted DPO enabled: "
                f"temperature={self.dim_temperature}, normalize={self.dim_weight_normalize}"
            )
        # ===== DiReDPO-Token：token 级加权 =====
        self.use_token_weighted = getattr(finetuning_args, "use_token_weighted_dpo", False)
        self.token_weight_temperature = getattr(finetuning_args, "token_weight_temperature", 1.0)
        self.token_weight_normalize = getattr(finetuning_args, "token_weight_normalize", True)
        if self.use_token_weighted:
            print(
                f"[DiReDPO-Token] Token-Weighted DPO enabled: "
                f"temperature={self.token_weight_temperature}, normalize={self.token_weight_normalize}"
            )
        if self.use_token_weighted and self.loss_type in ["orpo", "simpo"]:
            print("[DiReDPO-Token] orpo/simpo 不兼容 token 级加权，自动回退到 sample 级（如已开启）。")

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def get_batch_samples(self, *args, **kwargs):
        r"""Replace the method of DPO Trainer with the one of the standard Trainer."""
        return Trainer.get_batch_samples(self, *args, **kwargs)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute ORPO's odds ratio (OR) loss for batched log probabilities of the policy model."""
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""Compute SimPO loss for batched log probabilities of the policy model."""
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def bco_loss(
        self,
        chosen_logps: "torch.Tensor",
        rejected_logps: "torch.Tensor",
        reference_chosen_logps: "torch.Tensor",
        reference_rejected_logps: "torch.Tensor",
    ) -> "torch.Tensor":
        chosen_logratios = chosen_logps - reference_chosen_logps
        rejected_logratios = rejected_logps - reference_rejected_logps
        chosen_rewards = self.beta * chosen_logratios
        rejected_rewards = self.beta * rejected_logratios
        rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
        self.running.update(rewards)  # update baseline
        delta = self.running.mean
        bco_loss = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
            -(self.beta * rejected_logratios - delta)
        )
        return bco_loss
    
    @staticmethod
    def compute_dimension_weights(
        dim_weights: "torch.Tensor",
        dim_gap: "torch.Tensor",
        temperature: float = 2.0,
    ) -> "torch.Tensor":
        r"""Compute sample-level dimension weighting factor for DiReDPO.

        Formula: w_sample = sum( softmax(w_i * (1 + g_i) * T) * w_i * (1 + g_i) )
        """
        effective = dim_weights * (1.0 + dim_gap)  # [n_pairs, n_dims]
        normalized = F.softmax(effective * temperature, dim=-1)
        sample_weights = (normalized * effective).sum(dim=-1)  # [n_pairs]
        return sample_weights

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""Compute loss for preference learning."""
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
            else:
                raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

            if self.bco_gemma > 1e-6:
                bco_losses = self.bco_loss(
                    policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
                )
                losses = (losses + bco_losses * self.bco_gemma) / (1.0 + self.bco_gemma)  # re-weight W_p and W_q

        return losses, chosen_rewards, rejected_rewards

    @override
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"], is_ref_model: bool = False
    ) -> dict[str, "torch.Tensor"]:
        r"""Compute the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = nested_detach(batch, clone=True)  # avoid error

        labels = batch.pop("labels")  # dpo do not need compute loss in forward
        all_logits: torch.Tensor = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)
        all_logps, valid_length = get_batch_logps(
            logits=all_logits, labels=labels, ld_alpha=(self.ld_alpha if not is_ref_model else None)
        )
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            all_logps = all_logps / valid_length

        batch_size = batch["input_ids"].size(0) // 2
        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
        chosen_length, _ = valid_length.split(batch_size, dim=0)
        if self.loss_type in ["ipo", "orpo", "simpo"]:
            chosen_logps_avg = chosen_logps
        else:
            chosen_logps_avg = chosen_logps / chosen_length

        # ===== DiReDPO-Token：仅在启用时计算逐 token logp（节省显存）=====
        per_token_logps = None
        if self.use_token_weighted:
            per_token_logps, _ = get_per_token_logps(all_logits, labels)

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_logits": chosen_logits,
            "rejected_logits": rejected_logits,
            "chosen_logps_avg": chosen_logps_avg,
            "per_token_logps": per_token_logps,  # [2n, L-1] 或 None
            "valid_length": valid_length,        # 供 ipo 加权路径使用
        }

    @override
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: dict[str, "torch.Tensor"]
    ) -> tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Compute log probabilities of the reference model."""
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            ref_output = self.concatenated_forward(ref_model, batch, is_ref_model=True)
            reference_chosen_logps = ref_output["chosen_logps"]
            reference_rejected_logps = ref_output["rejected_logps"]
            self._last_ref_output = ref_output  # ===== DiReDPO-Token：缓存供加权路径使用 =====

        return reference_chosen_logps, reference_rejected_logps

    @override
    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
        r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        # ===== DiReDPO：取出维度引导字段（不进入模型 forward）=====
        dim_weights = batch.pop("dim_weights", None)
        dim_gap = batch.pop("dim_gap", None)
        # ===== DiReDPO-Token =====
        token_weights = batch.pop("token_weights", None)

        model_output = self.concatenated_forward(model, batch)
        policy_chosen_logps = model_output["chosen_logps"]
        policy_rejected_logps = model_output["rejected_logps"]
        policy_chosen_logits = model_output["chosen_logits"]
        policy_rejected_logits = model_output["rejected_logits"]
        policy_chosen_logps_avg = model_output["chosen_logps_avg"]

        reference_chosen_logps, reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        prefix = "eval_" if train_eval == "eval" else ""

        # ===== DiReDPO-Token：用逐 token 加权隐式 reward 重算 losses =====
        # rewards / metrics 仍用未加权标准值，保证日志可比；只有 losses 被替换。
        # 组合顺序：token 级替换（此处）→ +ftx → ×sample 级权重（下方 DiReDPO 块），两级可叠加。
        token_mode_active = (
            self.use_token_weighted
            and token_weights is not None
            and self.loss_type not in ["orpo", "simpo"]  # orpo/simpo 概率语义不兼容，自动回退
            and model_output.get("per_token_logps") is not None
            and self._last_ref_output is not None
        )
        if token_mode_active:
            n_pairs = policy_chosen_logps.shape[0]
            tw = token_weights[:, 1:].to(dtype=torch.float32)  # 与 shift 后的 [2n, L-1] 对齐
            w_sum = (model_output["per_token_logps"] * tw).sum(-1)          # [2n]
            chosen_logps_w, rejected_logps_w = w_sum.split(n_pairs, dim=0)

            ref_output = self._last_ref_output  # T5e：compute_reference_log_probs 中缓存
            ref_w_sum = (ref_output["per_token_logps"] * tw).sum(-1)
            ref_chosen_w, ref_rejected_w = ref_w_sum.split(n_pairs, dim=0)

            if self.loss_type == "ipo":  # 与标准路径一致：平均化
                valid_length = model_output["valid_length"]
                chosen_len, rejected_len = valid_length.split(n_pairs, dim=0)
                chosen_logps_w = chosen_logps_w / chosen_len
                rejected_logps_w = rejected_logps_w / rejected_len
                ref_chosen_w = ref_chosen_w / chosen_len
                ref_rejected_w = ref_rejected_w / rejected_len

            logits_w = (chosen_logps_w - ref_chosen_w) - (rejected_logps_w - ref_rejected_w)
            if self.loss_type == "sigmoid":
                losses = -F.logsigmoid(self.beta * logits_w)
            elif self.loss_type == "hinge":
                losses = torch.relu(1.0 - self.beta * logits_w)
            elif self.loss_type == "ipo":
                losses = (logits_w - 1.0 / (2.0 * self.beta)) ** 2

            resp_w = tw[tw > 0]  # 只看 response token（prompt/pad 位置为 0）
            if resp_w.numel() > 0:
                metrics[f"{prefix}token_weight/mean"] = resp_w.mean().item()  # 归一化后恒 ≈1，仅供完整性
                metrics[f"{prefix}token_weight/std"] = resp_w.std().item()    # 关键：>0 才说明 token 间有分化
                metrics[f"{prefix}token_weight/max"] = resp_w.max().item()    # 峰值权重（错误行）
                metrics[f"{prefix}token_weight/min"] = resp_w.min().item()
            else:
                metrics[f"{prefix}token_weight/mean"] = 1.0

        sft_loss = -policy_chosen_logps_avg
        if self.ftx_gamma > 1e-6:
            losses += self.ftx_gamma * sft_loss

        # ===== DiReDPO：维度加权损失（sample 级，乘在 token 加权后的 loss 上）=====
        if self.use_dim_weighted and dim_weights is not None and dim_gap is not None:
            n_pairs = losses.shape[0]
            # 分组布局：batch 前 n 行为 chosen、后 n 行为 rejected；逐对维度引导取前 n 行
            dim_weights = dim_weights[:n_pairs].to(device=losses.device, dtype=losses.dtype)
            dim_gap = dim_gap[:n_pairs].to(device=losses.device, dtype=losses.dtype)

            sample_weights = self.compute_dimension_weights(dim_weights, dim_gap, self.dim_temperature)
            if self.dim_weight_normalize:
                # 均值归一化到 1：保持整体 loss 尺度与标准 DPO 一致；
                # 字段全为默认值（普通 DPO 数据）时严格等效不加权
                sample_weights = sample_weights * (sample_weights.numel() / (sample_weights.sum() + 1e-8))

            losses = losses * sample_weights

            metrics[f"{prefix}dim_weight/mean"] = sample_weights.mean().item()
            metrics[f"{prefix}dim_weight/max"] = sample_weights.max().item()
            metrics[f"{prefix}dim_weight/min"] = sample_weights.min().item()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
        metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
        if self.loss_type == "orpo":
            metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
            metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()

        return losses.mean(), metrics

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        r"""Subclass and override to accept extra kwargs."""
        return super().compute_loss(model, inputs, return_outputs)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        r"""Log `logs` on the various objects watching training, including stored metrics."""
        # logs either has "loss" or "eval_loss"
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        key_list, metric_list = [], []
        for key, metrics in self._stored_metrics[train_eval].items():
            key_list.append(key)
            metric_list.append(torch.tensor(metrics, dtype=torch.float).to(self.accelerator.device).mean().item())

        del self._stored_metrics[train_eval]
        if len(metric_list) < 10:  # pad to for all reduce
            for i in range(10 - len(metric_list)):
                key_list.append(f"dummy_{i}")
                metric_list.append(0.0)

        metric_list = torch.tensor(metric_list, dtype=torch.float).to(self.accelerator.device)
        metric_list = self.accelerator.reduce(metric_list, "mean").tolist()
        for key, metric in zip(key_list, metric_list):  # add remaining items
            if not key.startswith("dummy_"):
                logs[key] = metric

        return Trainer.log(self, logs, *args, **kwargs)
