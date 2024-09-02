import io
import os
import random
import logging
import traceback
import gc
from typing import Union

import logging
import numpy as np
import PIL
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm.auto import tqdm
import torch.nn.functional as F

from core.data import AGTPretrainingDataset
from core.criterions import Wav2vec2Loss
from core.trainer import Trainer
from utils.util import min_max_normalize, segment_feature

logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)
import matplotlib

matplotlib.use("Agg")


class ContrastivePretrainTrainer(Trainer):

    class History:
        def __init__(self, num_strings):
            self.num_strings = num_strings
            self.reset()

        def reset(self):
            self.loss = []
            self.contrastive_loss = []
            self.diversity_loss = []
            self.accuracy = []
            self.perplexity = []
            self.entropy = []

        def remove_inf_nan(self):
            self.loss = [l for l in self.loss if not np.isnan(l) and not np.isinf(l)]
            self.contrastive_loss = [
                c for c in self.contrastive_loss if not np.isnan(c) and not np.isinf(c)
            ]
            self.diversity_loss = [
                d for d in self.diversity_loss if not np.isnan(d) and not np.isinf(d)
            ]
            self.accuracy = [
                a for a in self.accuracy if not np.isnan(a) and not np.isinf(a)
            ]
            self.perplexity = [
                p for p in self.perplexity if not np.isnan(p) and not np.isinf(p)
            ]
            self.entropy = [
                e for e in self.entropy if not np.isnan(e) and not np.isinf(e)
            ]

        def average(self):
            self.remove_inf_nan()
            return {
                "loss": np.mean(self.loss),
                "contrastive_loss": np.mean(self.contrastive_loss),
                "diversity_loss": np.mean(self.diversity_loss),
                "accuracy": np.mean(self.accuracy),
                "perplexity": np.mean(self.perplexity),
                "entropy": np.mean(self.entropy),
            }

        def average_last_n(self, n=10):
            self.remove_inf_nan()
            return {
                "loss": np.mean(self.loss[-n:]),
                "contrastive_loss": np.mean(self.contrastive_loss[-n:]),
                "diversity_loss": np.mean(self.diversity_loss[-n:]),
                "accuracy": np.mean(self.accuracy[-n:]),
                "perplexity": np.mean(self.perplexity[-n:]),
                "entropy": np.mean(self.entropy[-n:]),
            }

    def __init__(self, cfg: DictConfig, run_dir: Union[os.PathLike, str]):
        super().__init__(cfg, run_dir)

        if str(cfg.criterions.pretrain.name) == "Wav2vec2Loss":
            self.criterion = Wav2vec2Loss(cfg.criterions.pretrain.config)
        else:
            raise ValueError(
                f"A valid criterion for pretraining must be specified, got {cfg.criterions.pretrain.name}"
            )

        self.valid_data_loader = AGTPretrainingDataset.get_valid_dataloader(
            cfg.data, self.audio_processor
        )
        self.train_data_loader = AGTPretrainingDataset.get_train_dataloader(
            cfg.data, self.audio_processor
        )
        if cfg.data.test_csv_file is not None:
            self.test_data_loader = AGTPretrainingDataset.get_test_dataloader(
                cfg.data, self.audio_processor
            )

        if (
            cfg.checkpoint.finetune_from_model is not None
            or cfg.evaluate_before_training
        ):
            self.evaluate()

    def _get_data_items(self, batch):
        batch_size = batch["x"].shape[0]
        batch_x = batch["x"].to(self.device)
        batch_lengths = batch["lengths"].to(self.device)
        if "ffm" in batch:
            batch_ffm = batch["ffm"].to(self.device)
        else:
            batch_ffm = torch.tensor([]).to(self.device)
        return batch_size, batch_ffm, batch_x, batch_lengths

    def _compute_losses(
        self,
        encoder_out,
        quantized_features,
        perplexity,
        time_mask_indices,
        training=True,
    ):
        step_loss = 0

        step_loss, contrastive_loss, diversity_loss = self.criterion(
            encoder_out, quantized_features, perplexity, time_mask_indices
        )
        step_loss = step_loss.to(self.device)

        if training:
            if self.accumulation_steps > 1 and self.accumulation_reduction == "mean":
                step_loss /= self.accumulation_steps
            if self.use_amp:
                self.scaler.scale(step_loss).backward()
            else:
                step_loss.backward()
                if not self.check_valid_gradients():
                    logger.warn(
                        f"Detected inf or nan values in gradients. Not updating model parameters."
                    )

        return step_loss, contrastive_loss, diversity_loss

    def _update_model_weights(self):
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.train_tensorboard.log_step(
                "scale", float(self.scaler.get_scale()), self.current_update
            )
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        self.current_update += 1

    def _update_lr_scheduler_updates(self):
        if self.lr_scheduler_updates is not None:
            if self.lr_scheduler_updates_interval is None:
                self.lr_scheduler_updates_interval = 1
            if self.current_update % self.lr_scheduler_updates_interval == 0:
                if self.lr_scheduler_warmup is not None:
                    self.lr_scheduler_warmup.step()
                else:
                    self.lr_scheduler_updates.step()
                logger.info(
                    f"Learning rate updated to {float(self.optimizer.param_groups[0]['lr']):.4e}"
                )
                self.valid_tensorboard.log_step(
                    "lr",
                    float(self.optimizer.param_groups[0]["lr"]),
                    self.current_update,
                )

    def _update_lr_scheduler_epochs(self):
        if (
            self.cfg.lr_scheduler.epoch is not None
            and self.cfg.lr_scheduler.epoch.name == "StepLR"
        ):
            self.lr_scheduler_epoch.step()
            logger.info(
                f"Learning rate updated to {float(self.optimizer.param_groups[0]['lr']):.2e}"
            )
            self.valid_tensorboard.log_step(
                "lr", float(self.optimizer.param_groups[0]["lr"]), self.current_update
            )
        elif (
            self.cfg.lr_scheduler.epoch is not None
            and self.cfg.lr_scheduler.epoch.name == "ReduceLROnPlateau"
        ):
            self.lr_scheduler_epoch.step(self.last_loss)
            logger.info(
                f"Learning rate updated to {float(self.optimizer.param_groups[0]['lr']):.2e}"
            )
            self.valid_tensorboard.log_step(
                "lr", float(self.optimizer.param_groups[0]["lr"]), self.current_update
            )

    def _log_tensorboard_step(self, tb, step_loss, step_contrastive_loss, step_diversity_loss, step_accuracy, step_ppl, step_entropy):
        tb.log_step("loss", float(step_loss.item()), self.current_update)
        tb.log_step("contrastive_loss", float(step_contrastive_loss.item()), self.current_update)
        tb.log_step("diversity_loss", float(step_diversity_loss.item()), self.current_update)
        tb.log_step("accuracy", float(step_accuracy.item()), self.current_update)
        tb.log_step("perplexity", float(step_ppl.item()), self.current_update)
        tb.log_step("entropy", float(step_entropy.item()), self.current_update)

    def _log_tensorboard_epoch(self, tb, hist):
        tb.log_epoch(
            "lr", float(self.optimizer.param_groups[0]["lr"]), self.current_epoch
        )
        tb.log_epoch("loss", float(np.mean(hist.loss)), self.current_epoch)
        tb.log_epoch("contrastive_loss", float(np.mean(hist.contrastive_loss)), self.current_epoch)
        tb.log_epoch("diversity_loss", float(np.mean(hist.diversity_loss)), self.current_epoch)
        tb.log_epoch("accuracy", float(np.mean(hist.accuracy)), self.current_epoch)
        tb.log_epoch("perplexity", float(np.mean(hist.perplexity)), self.current_epoch)
        tb.log_epoch("entropy", float(np.mean(hist.entropy)), self.current_epoch)

    def train_one_epoch(self):
        self.model.train()

        train_step_it = (
            tqdm(
                enumerate(self.train_data_loader),
                unit="step",
                total=len(self.train_data_loader),
                dynamic_ncols=True,
                colour="#00aaff",
                leave=False,
            )
            if self.use_tqdm
            else enumerate(self.train_data_loader)
        )

        train_hist = self.History(self.num_strings)

        assert (
            self.accumulation_steps > 0
        ), "self.accumulation_steps must be greater than 0"

        for idx, batch in train_step_it:
            try:
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()

                batch_size, batch_ffm, batch_x, batch_lengths = self._get_data_items(
                    batch
                )

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    output = self.model(
                        batch_x.to(self.device),
                        ffm=(
                            batch_ffm.to(self.device)
                            if batch_ffm.nelement() != 0
                            else None
                        ),
                        return_features=True,
                        # lengths=batch_lengths, # Not used
                        freeze_feature_extractor=self.freeze_finetune_updates
                        is not None
                        and self.current_update < self.freeze_finetune_updates,
                        freeze_encoder=self.freeze_finetune_updates is not None
                        and self.current_update < self.freeze_finetune_updates
                        and not self.freeze_only_feature_extractor,
                        **self.model_extra_forward_args,
                    )

                    batch_output, batch_target, batch_mask_indices, batch_ppl = (
                        output["o"],
                        output["q"],
                        output["mask_indices"],
                        output["ppl"],
                    )

                    step_loss, step_contrastive_loss, step_diversity_loss = self._compute_losses(
                        encoder_out=batch_output,
                        quantized_features=batch_target,
                        perplexity=batch_ppl,
                        time_mask_indices=batch_mask_indices,
                    )

                    step_ppl = batch_ppl.mean()
                    log_perplexity = torch.log(batch_ppl)
                    step_entropy = torch.sum(torch.sum(batch_ppl*log_perplexity, dim=-1))
                    batch_target_argmax = torch.nn.functional.softmax(
                        batch_target, dim=-1
                    ).argmax(dim=-1)
                    batch_output_argmax = torch.nn.functional.softmax(
                        batch_output, dim=-1
                    ).argmax(dim=-1)
                    step_accuracy = (
                        torch.sum(batch_target_argmax == batch_output_argmax).float()
                        / batch_target_argmax.nelement()
                    )

                if self.grad_norm_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm_clip
                    )

                if (
                    idx % self.accumulation_steps == 0
                    or idx == len(self.train_data_loader) - 1
                    and self.current_update > 0
                ):
                    train_step_it.set_description(f"Epoch {self.current_epoch} [-]")
                    self._update_model_weights()
                    self._update_lr_scheduler_updates()

                    if (
                        self.save_interval_updates is not None
                        and self.current_update % self.save_interval_updates == 0
                    ):
                        self.save_checkpoint(step=self.current_update)

                    # ============================================================

                    step_loss_log = float(step_loss.detach().cpu().item())
                    train_hist.loss.append(step_loss_log)
                    train_hist.accuracy.append(
                        float(step_accuracy.detach().cpu().item())
                    )
                    train_hist.perplexity.append(float(step_ppl.detach().cpu().item()))
                    train_hist.entropy.append(float(step_entropy.detach().cpu().item()))
                    train_hist.contrastive_loss.append(
                        float(step_contrastive_loss.detach().cpu().item())
                    )
                    train_hist.diversity_loss.append(
                        float(step_diversity_loss.detach().cpu().item())
                    )

                    self._log_tensorboard_step(
                        self.train_tensorboard, step_loss, step_contrastive_loss, step_diversity_loss, step_accuracy, step_ppl, step_entropy
                    )

                log_prog = [
                    f"update: {self.current_update}",
                    f"lr: {float(self.optimizer.param_groups[0]['lr']):.2e}",
                    f"loss: {step_loss_log:.3f}↓",
                    f"c_loss: {float(step_contrastive_loss):.3f}↓",
                    f"d_loss: {float(step_diversity_loss):.3f}",
                    f"accuracy: {float(step_accuracy):.3f}↑",
                    f"ppl: {float(step_ppl):.3f}",
                    f"entropy: {float(step_entropy):.3f}",
                ]
                if self.accumulation_steps > 1:
                    log_prog.append(
                        f"i/ac: {(idx % self.accumulation_steps)+1}/{self.accumulation_steps}"
                    )
                if self.use_amp:
                    log_prog.append(
                        f"scale: {self.scaler.get_scale() if self.use_amp else 1.0:.3f}"
                    )
                if self.use_tqdm:
                    train_step_it.set_description(f"Epoch {self.current_epoch} [#]")
                    train_step_it.set_postfix_str(", ".join(log_prog))

                valid_hist_average = None
                if (
                    self.validate_interval_updates is not None
                    and self.current_update % self.validate_interval_updates == 0
                ):
                    valid_hist_average = self.evaluate()

                if self.log_interval_steps > 0 and idx % self.log_interval_steps == 0:
                    avg_metrics = train_hist.average_last_n(self.log_interval_steps)
                    log_interval_steps = {
                        "epoch": f"{self.current_epoch}",
                        "updates": f"{self.current_update}",
                        "lr": f"{float(self.optimizer.param_groups[0]['lr']):.2e}",
                        "loss_avg": f"{avg_metrics.get('loss'):.3f}",
                        "contrastive_loss_avg": f"{avg_metrics.get('contrastive_loss'):.3f}",
                        "diversity_loss_avg": f"{avg_metrics.get('diversity_loss'):.3f}",
                        "accuracy_avg": f"{avg_metrics.get('accuracy'):.3f}",
                        "perplexity_avg": f"{avg_metrics.get('perplexity'):.3f}",
                        "entropy_avg": f"{avg_metrics.get('entropy'):.3f}",
                    }
                    if valid_hist_average is not None:
                        log_interval_steps["valid_loss"] = (
                            f"{valid_hist_average.get('loss'):.3f}"
                        )
                        log_interval_steps["valid_accuracy"] = (
                            f"{valid_hist_average.get('accuracy'):.3f}"
                        )
                        log_interval_steps["valid_perplexity"] = (
                            f"{valid_hist_average.get('perplexity'):.3f}"
                        )

                    if self.use_tqdm:
                        train_step_it.clear()
                    logger.info(log_interval_steps)

                if (
                    self.save_interval_updates is not None
                    and self.current_update % self.save_interval_updates == 0
                ):
                    self.save_checkpoint(step=self.current_update)

                if (
                    self.max_updates is not None
                    and self.current_update > self.max_updates
                ):
                    logger.info(
                        f"Stopping training due to max_updates={self.max_updates}"
                    )
                    self.save_checkpoint()
                    self.evaluate()
                    break

            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())

                if self.raise_error:
                    raise e

                torch.cuda.empty_cache()
                continue

        assert self.current_update > 0, "No updates were made during training"

        self.save_checkpoint(step=self.current_update)

        self._log_tensorboard_epoch(self.train_tensorboard, train_hist)

        return train_hist.average()

    def _log_heatmap(self, tb_writer, tag, matrix, global_step):
        # Normalize the matrix
        normalized_matrix = min_max_normalize(matrix, 1, 0)

        if len(normalized_matrix.shape) == 3:
            normalized_matrix = normalized_matrix.squeeze()

        # Convert to a colormap (heatmap) in matplotlib
        fig, ax = plt.subplots(dpi=100)
        cax = ax.matshow(normalized_matrix.detach().cpu().numpy(), cmap="viridis")
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        # Save the plot to a bufcer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Convert the bufcer to a PIL image
        image = np.array(PIL.Image.open(buf).convert("RGB"))
        # H W C -> C H W
        image = image.transpose(2, 0, 1)
        # Log the image to TensorBoard
        tb_writer.log_image(tag, image, global_step)
        plt.close(fig)
        buf.close()

    def _log_images(
        self,
        tb,
        batch_x,
        batch_ffm,
        batch_audio,
        output,
        ex_idx,
        heatmap=True,
        max_width=1000,
    ):
        # TODO limit size
        if len(batch_x[ex_idx].shape) == 3:
            tb.log_image(f"features/x", batch_x[ex_idx].transpose(1, 2), self.current_update)
            self._log_heatmap(
                tb, "heatmaps/features/x", batch_x[ex_idx], self.current_update
            )
        if batch_ffm.nelement() != 0:
            tb.log_image(
                f"features/ffm",
                batch_ffm[ex_idx].unsqueeze(0).transpose(1, 2),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb, "heatmaps/features/ffm", batch_ffm[ex_idx].transpose(1, 2), self.current_update
                )
        if batch_audio is not None:
            tb.log_audio(
                f"audio",
                batch_audio[ex_idx],
                sample_rate=self.cfg.audio.sr,
                step=self.current_update,
            )
        if output.get("ffm_emb", None) is not None:
            tb.log_image(
                f"ffm_emb",
                output["ffm_emb"][ex_idx, :, :].unsqueeze(0).transpose(1, 2),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    "heatmaps/ffm_emb",
                    output["ffm_emb"][ex_idx, :, :].transpose(1, 2),
                    self.current_update,
                )
        if output.get("attention_map", None) is not None:
            tb.log_image(
                f"attention_map",
                min_max_normalize(output["attention_map"][ex_idx, :, :], 1, 0)
                .unsqueeze(0)
                .transpose(1, 2)
                * 255,
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    "heatmaps/attention_map",
                    output["attention_map"][ex_idx, :, :],
                    self.current_update,
                )
        if output.get("z", None) is not None:
            tb.log_image(
                f"z",
                output["z"][ex_idx, :, :].squeeze().unsqueeze(0).transpose(1, 2),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    "heatmaps/z",
                    output["z"][ex_idx, :, :].squeeze().transpose(0, 1),
                    self.current_update,
                )
        if output.get("masked_z", None) is not None:
            tb.log_image(
                f"masked_z",
                output["masked_z"][ex_idx, :, :].squeeze().unsqueeze(0).transpose(1, 2),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    "heatmaps/masked_z",
                    output["masked_z"][ex_idx, :, :].squeeze().transpose(0, 1),
                    self.current_update,
                )
        if output.get("q", None) is not None:
            tb.log_image(
                f"q",
                output["q"][ex_idx, :, :].squeeze().unsqueeze(0).transpose(1, 2),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    "heatmaps/q",
                    output["q"][ex_idx, :, :].squeeze().transpose(0, 1),
                    self.current_update,
                )
        if output.get("c", None) is not None:
            tb.log_image(
                f"c",
                output["c"][ex_idx, :, :].squeeze().unsqueeze(0).transpose(1, 2),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    "heatmaps/c",
                    output["c"][ex_idx, :, :].squeeze().transpose(0, 1),
                    self.current_update,
                )
        if output.get("o", None) is not None:
            tb.log_image(
                f"o",
                output["o"][ex_idx, :, :].squeeze().unsqueeze(0).transpose(1, 2),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    "heatmaps/o",
                    output["o"][ex_idx, :, :].squeeze().transpose(0, 1),
                    self.current_update,
                )

    def evaluate(self, data_loader=None, log_to_tensorboard=True):
        logger.info(f"Starting evaluation")
        self.model.eval()

        if data_loader is None:
            data_loader = self.valid_data_loader

        evaluation_step_it = (
            tqdm(
                enumerate(data_loader),
                unit="step",
                total=len(data_loader),
                dynamic_ncols=True,
                colour="#23ffaa",
                leave=False,
            )
            if self.use_tqdm
            else enumerate(data_loader)
        )

        valid_hist = self.History(self.num_strings)

        for idx, batch in evaluation_step_it:

            batch_size, batch_ffm, batch_x, batch_lengths = self._get_data_items(batch)

            with torch.no_grad():
                output = self.model(
                    batch_x.to(self.device),
                    ffm=(
                        batch_ffm.to(self.device) if batch_ffm.nelement() != 0 else None
                    ),
                    # lengths=batch_lengths, # Not used
                    return_features=True,
                    **self.model_extra_forward_args,
                )

                batch_output, batch_target, batch_mask_indices, batch_ppl = (
                    output["o"],
                    output["q"],
                    output["mask_indices"],
                    output["ppl"],
                )

                step_ppl = batch_ppl.mean()
                log_perplexity = torch.log(batch_ppl)
                step_entropy = torch.sum(torch.sum(batch_ppl*log_perplexity, dim=-1))
                step_loss, step_contrastive_loss, step_diversity_loss = self._compute_losses(
                    encoder_out=batch_output,
                    quantized_features=batch_target,
                    perplexity=batch_ppl,
                    time_mask_indices=batch_mask_indices,
                    training=False,
                )

                batch_target_argmax = torch.nn.functional.softmax(
                    batch_target, dim=-1
                ).argmax(dim=-1)
                batch_output_argmax = torch.nn.functional.softmax(
                    batch_output, dim=-1
                ).argmax(dim=-1)
                step_accuracy = (
                    torch.sum(batch_target_argmax == batch_output_argmax).float()
                    / batch_target_argmax.nelement()
                )

            if output.get("features", None) is not None:
                batch_audio = batch_x
                batch_x = output["features"]
            else:
                batch_audio = None

            if log_to_tensorboard and idx == 0:
                self._log_images(
                    self.valid_tensorboard,
                    batch_x,
                    batch_ffm,
                    batch_audio,
                    output,
                    0,
                )

            step_loss_log = float(step_loss.detach().cpu().item())
            valid_hist.loss.append(step_loss_log)
            valid_hist.accuracy.append(float(step_accuracy.detach().cpu().item()))
            valid_hist.perplexity.append(float(step_ppl.detach().cpu().item()))
            valid_hist.entropy.append(float(step_entropy.detach().cpu().item()))
            valid_hist.contrastive_loss.append(
                float(step_contrastive_loss.detach().cpu().item())
            )
            valid_hist.diversity_loss.append(
                float(step_diversity_loss.detach().cpu().item())
            )

            log_prog = [
                f"epoch {self.current_epoch}",
                f"update: {self.current_update}",
                f"loss: {step_loss_log:.3f}↓",
                f"c_loss: {float(step_contrastive_loss):.3f}↓",
                f"d_loss: {float(step_diversity_loss):.3f}",
                f"accuracy: {float(step_accuracy):.3f}↑",
                f"ppl: {float(step_ppl):.3f}",
                f"entropy: {float(step_entropy):.3f}",
            ]
            if self.use_tqdm:
                evaluation_step_it.set_description(log_prog[0])
                evaluation_step_it.set_postfix_str(", ".join(log_prog[1:]))

            torch.cuda.empty_cache()

            if self.log_interval_steps > 0 and idx % self.log_interval_steps == 0:
                avg_metrics = valid_hist.average_last_n(self.log_interval_steps)
                log_interval_steps = {
                    "epoch": f"{self.current_epoch}",
                    "updates": f"{self.current_update}",
                    "loss_avg": f"{avg_metrics.get('loss'):.3f}",
                    "contrastive_loss_avg": f"{avg_metrics.get('contrastive_loss'):.3f}",
                    "diversity_loss_avg": f"{avg_metrics.get('diversity_loss'):.3f}",
                    "accuracy_avg": f"{avg_metrics.get('accuracy'):.3f}",
                    "perplexity_avg": f"{avg_metrics.get('perplexity'):.3f}",
                    "entropy_avg": f"{avg_metrics.get('entropy'):.3f}",
                }
                if self.use_tqdm:
                    evaluation_step_it.clear()
                logger.info(log_interval_steps)

        if log_to_tensorboard:
            self._log_tensorboard_epoch(self.valid_tensorboard, valid_hist)
            self._log_tensorboard_step(
                self.valid_tensorboard,
                step_loss,
                step_contrastive_loss,
                step_diversity_loss,
                step_accuracy,
                step_ppl,
                step_entropy,
            )

        if self.best_loss is None:
            self.best_loss = valid_hist.average().get("loss")
        if valid_hist.average().get("loss") < self.best_loss:
            self.best_loss = valid_hist.average().get("loss")
            if self.use_tqdm:
                evaluation_step_it.clear()
            logger.info(f"Best loss reached: {self.best_loss}")
            self.save_checkpoint(is_best=True)

        return valid_hist.average()

    def train(self):
        logger.info(f"Start training")
        self.model.train()
        self.is_training = True

        self.cer_tabs_total_avg = 0
        assert self.accumulation_steps > 0, "accumulation_steps must be greater than 0"

        for _ in range(1, self.max_epochs + 1):

            if self.max_updates is not None and self.current_update > self.max_updates:
                break

            self.current_epoch += 1

            if self.current_epoch > self.max_epochs:
                logger.info(f"Stopping training due to max_epochs={self.max_epochs}")
                self.save_checkpoint()
                valid_hist_avg = self.evaluate()
                epoch_info = {
                    "valid_loss": f"{valid_hist_avg.get('loss'):.3f}",
                    "valid_contrastive_loss": f"{valid_hist_avg.get('contrastive_loss'):.3f}",
                    "valid_diversity_loss": f"{valid_hist_avg.get('diversity_loss'):.3f}",
                    "valid_accuracy": f"{valid_hist_avg.get('accuracy'):.3f}",
                    "valid_perplexity": f"{valid_hist_avg.get('perplexity'):.3f}",
                }
                logger.info(f"Training done: {epoch_info}")
                break

            logger.info(f"Training epoch {self.current_epoch}")
            train_hist_avg = self.train_one_epoch()

            epoch_info = {
                "loss": f"{train_hist_avg.get('loss', 0):.3f}",
                "contrastive_loss": f"{train_hist_avg.get('contrastive_loss', 0):.3f}",
                "diversity_loss": f"{train_hist_avg.get('diversity_loss', 0):.3f}",
                "accuracy": f"{train_hist_avg.get('accuracy', 0):.3f}",
                "perplexity": f"{train_hist_avg.get('perplexity', 0):.3f}",
            }
            if (
                self.validate_interval_epochs is not None
                and self.current_epoch % self.validate_interval_epochs == 0
            ):
                valid_hist_avg = self.evaluate()
                self.last_loss = valid_hist_avg.get("loss")
                self._update_lr_scheduler_epochs()
                epoch_info.update(
                    {
                        "valid_loss": f"{valid_hist_avg.get('loss', 0):.3f}",
                        "valid_contrastive_loss": f"{valid_hist_avg.get('contrastive_loss', 0):.3f}",
                        "valid_diversity_loss": f"{valid_hist_avg.get('diversity_loss', 0):.3f}",
                        "valid_accuracy": f"{valid_hist_avg.get('accuracy', 0):.3f}",
                        "valid_perplexity": f"{valid_hist_avg.get('perplexity', 0):.3f}",
                    }
                )
            logger.info(f"Epoch {self.current_epoch} done: {epoch_info}")

        print("=" * (os.get_terminal_size().columns))

        logger.info("Training done.")
        self.is_training = False

        if self.run_test_at_end:
            logger.info("Running evaluation on test set")
            test_hist_avg = self.evaluate(
                data_loader=self.test_data_loader, log_to_tensorboard=False
            )
            info = {
                "test_loss": f"{test_hist_avg.get('loss', 0):.3f}",
                "test_contrastive_loss": f"{test_hist_avg.get('contrastive_loss', 0):.3f}",
                "test_diversity_loss": f"{test_hist_avg.get('diversity_loss', 0):.3f}",
                "test_accuracy": f"{test_hist_avg.get('accuracy', 0):.3f}",
                "test_perplexity": f"{test_hist_avg.get('perplexity', 0):.3f}",
            }
            logger.info(f"Evaluation on test set: {info}")
