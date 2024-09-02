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

from core.data import AGTSequenceDataset
from core.criterions import MCTC, CTCLossByString, CTCLossByStringAlignment, MultiResolutionCTCLossByString
from core.trainer import Trainer
from core.decoder import GreedyCTCDecoder
from utils.metrics import cer
from utils.util import min_max_normalize, segment_feature

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')

class CTCTrainer(Trainer):

    class History:
        # TODO: some metrics are not implemented. Note Error Rate (NER) needs a proper decoding method for notes. Tab Error Rate (TER) and Fret Error Rate (FER) requires a ground truth tablature to compare with the predicted tablature, which might not be available since this training code is designed to work with CTC loss.
        def __init__(self, num_strings):
            self.num_strings = num_strings
            self.reset()

        def reset(self):
            self.strings_losses = [[] for _ in range(self.num_strings)]
            self.strings_cers = [[] for _ in range(self.num_strings)]
            self.strings_onsets_ratio = [[] for _ in range(self.num_strings)]
            self.loss = []
            self.loss_tabs = []
            self.loss_notes = []
            self.mean_ctc_loss = []
            self.cer = []
            # self.ter = []  # TODO: not implemented
            # self.fer = []  # TODO: not implemented
            self.ner = [] # TODO: not implemented
            self.onsets_ratio = []
            
        def remove_inf_nan(self):
            self.loss = [l for l in self.loss if not np.isnan(l) and not np.isinf(l)]
            self.loss_tabs = [l for l in self.loss_tabs if not np.isnan(l) and not np.isinf(l)]
            self.loss_notes = [l for l in self.loss_notes if not np.isnan(l) and not np.isinf(l)]
            self.mean_ctc_loss = [l for l in self.mean_ctc_loss if not np.isnan(l) and not np.isinf(l)]
            self.cer = [l for l in self.cer if not np.isnan(l) and not np.isinf(l)]
            self.ner = [l for l in self.ner if not np.isnan(l) and not np.isinf(l)]
            self.onsets_ratio = [l for l in self.onsets_ratio if not np.isnan(l) and not np.isinf(l)]
            for s in range(self.num_strings):
                self.strings_losses[s] = [l for l in self.strings_losses[s] if not np.isnan(l) and not np.isinf(l)]
                self.strings_cers[s] = [l for l in self.strings_cers[s] if not np.isnan(l) and not np.isinf(l)]
                self.strings_onsets_ratio[s] = [l for l in self.strings_onsets_ratio[s] if not np.isnan(l) and not np.isinf(l)]

        def average(self):
            self.remove_inf_nan()
            return {
                "loss": np.mean(self.loss),
                "loss_tabs": np.mean(self.loss_tabs),
                "loss_notes": np.mean(self.loss_notes),
                "mean_ctc_loss": np.mean(self.mean_ctc_loss),
                "cer": np.mean(self.cer),
                "ner": np.mean(self.ner),
                "onsets_ratio": np.mean(self.onsets_ratio),
                "strings_losses": [np.mean(s) for s in self.strings_losses],
                "strings_cers": [np.mean(s) for s in self.strings_cers],
                "strings_onsets_ratio": [np.mean(s) for s in self.strings_onsets_ratio],
            }

        def average_last_n(self, n=10):
            self.remove_inf_nan()
            return {
                "loss": np.mean(self.loss[-n:]),
                "loss_tabs": np.mean(self.loss_tabs[-n:]),
                "loss_notes": np.mean(self.loss_notes[-n:]),
                "mean_ctc_loss": np.mean(self.mean_ctc_loss[-n:]),
                "cer": np.mean(self.cer[-n:]),
                "ner": np.mean(self.ner[-n:]),
                "onsets_ratio": np.mean(self.onsets_ratio[-n:]),
                "strings_losses": [np.mean(s[-n:]) for s in self.strings_losses],
                "strings_cers": [np.mean(s[-n:]) for s in self.strings_cers],
                "strings_onsets_ratio": [
                    np.mean(s[-n:]) for s in self.strings_onsets_ratio
                ],
            }

    def __init__(self, cfg: DictConfig, run_dir: Union[os.PathLike, str]):
        super().__init__(cfg, run_dir)

        self.class_probabilities = cfg.class_probabilities
        # self._labels = ["-", *[str(x) for x in list(range(cfg.model.fret_size + 1))]]
        # TODO: get labels from config
        self._labels = ["-", *[str(x) for x in list(range(cfg.model.fret_size + 1))], '*']
        self.decoder = GreedyCTCDecoder(labels=self._labels)

        self.predict_notes = cfg.predict_notes
        
        self.tab_by_string_criterion = self.note_criterion = None
        if str(cfg.criterions.tab.name) == "CTCLossByString":
            self.tab_by_string_criterion = CTCLossByString(**cfg.criterions.tab.config)
        elif str(cfg.criterions.tab.name) == "CTCLossByStringAlignment":
            self.tab_by_string_criterion = CTCLossByStringAlignment(**cfg.criterions.tab.config)
        elif str(cfg.criterions.tab.name) == "MultiResolutionCTCLossByString":
            self.tab_by_string_criterion = MultiResolutionCTCLossByString(**cfg.criterions.tab.config)
        else:
            raise ValueError(
                f"A valid criterion for tab prediction must be specified, got {cfg.criterions.tab}"
            )
        if self.predict_notes and cfg.criterions.notes is not None:
            if str(cfg.criterions.notes.name) == "MCTC":
                self.note_criterion = MCTC(**cfg.criterions.notes.config)
        self.criterion_weights = (
            cfg.criterions.criterion_weights
            if "criterion_weights" in cfg.criterions
            else [1] * len(cfg.criterions) / len(cfg.criterions)  # BUG
        )

        self.valid_data_loader = AGTSequenceDataset.get_valid_dataloader(
            cfg.data, self.audio_processor
        )
        self.train_data_loader = AGTSequenceDataset.get_train_dataloader(
            cfg.data, self.audio_processor
        )
        if cfg.data.test_csv_file is not None:
            self.test_data_loader = AGTSequenceDataset.get_test_dataloader(
                cfg.data, self.audio_processor
            )

        if (
            cfg.checkpoint.finetune_from_model is not None
            or cfg.evaluate_before_training
        ):
            self.evaluate()

    def labels(self, i: int):
        if 0 <= int(i) < len(self._labels):
            return self._labels[i]
        else:
            return "?"

    def _get_data_items(self, batch_features, batch_targets):
        batch_size = batch_features["x"].shape[0]
        batch_x = batch_features["x"].to(self.device)
        if "ffm" in batch_features:
            batch_ffm = batch_features["ffm"].to(self.device)
        else:
            batch_ffm = torch.tensor([]).to(self.device)
        batch_frets_seq = [
            tgt_string.to(self.device) for tgt_string in batch_targets["frets_seq"]
        ]
        if "notes_seq" in batch_targets:
            batch_notes_seq = batch_targets["notes_seq"].to(self.device)
        else:
            batch_notes_seq = None
        batch_fret_target_lengths = batch_targets["fret_target_lengths"].to(self.device)
        return (
            batch_size,
            batch_x,
            batch_ffm,
            batch_frets_seq,
            batch_notes_seq,
            batch_fret_target_lengths,
        )

    def _compute_losses(
        self,
        batch_tabs_logits,
        batch_tabs_log_probs,
        batch_frets_seq,
        tab_input_lengths,
        batch_fret_target_lengths,
        batch_notes_log_probs,
        batch_notes_seq,
        training=True,
    ):
        step_loss = 0

        step_tab_loss, step_string_losses = self.tab_by_string_criterion(
            logits=batch_tabs_logits,  # S x T x B x C
            log_probs=batch_tabs_log_probs,  # S x T x B x C
            targets=batch_frets_seq,  # S x B x V
            input_lengths=tab_input_lengths,  # S x B
            target_lengths=batch_fret_target_lengths,  # S x B
        )
        step_tab_loss = step_tab_loss.to(self.device)
        step_loss += step_tab_loss * self.criterion_weights[0]

        step_note_loss = None
        if self.note_criterion is not None:
            # B x 2 x T x (C+1) -> B x 2 x (C+1) x T
            batch_notes_log_probs = batch_notes_log_probs.permute(0, 1, 3, 2)
            step_note_loss = self.note_criterion(batch_notes_log_probs, batch_notes_seq)
            step_note_loss = step_note_loss.to(self.device)
            step_loss += step_note_loss * self.criterion_weights[1]

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

        return step_loss, step_tab_loss, step_note_loss, step_string_losses

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
        if self.cfg.lr_scheduler.epoch is not None and self.cfg.lr_scheduler.epoch.name == "StepLR":
            self.lr_scheduler_epoch.step()
            logger.info(
                f"Learning rate updated to {float(self.optimizer.param_groups[0]['lr']):.2e}"
            )
            self.valid_tensorboard.log_step(
                "lr", float(self.optimizer.param_groups[0]["lr"]), self.current_update
            )
        elif self.cfg.lr_scheduler.epoch is not None and self.cfg.lr_scheduler.epoch.name == "ReduceLROnPlateau":
            self.lr_scheduler_epoch.step(self.last_loss)
            logger.info(
                f"Learning rate updated to {float(self.optimizer.param_groups[0]['lr']):.2e}"
            )
            self.valid_tensorboard.log_step(
                "lr", float(self.optimizer.param_groups[0]["lr"]), self.current_update
            )

    def _generate_tab_transcriptions(
        self, batch_tabs_log_probs, batch_frets_seq, batch_fret_target_lengths
    ):
        actual_transcriptions = [[] for _ in range(self.num_strings)]
        for s in range(self.num_strings):
            for b in range(batch_fret_target_lengths[s].shape[0]):
                indices = batch_frets_seq[s][b][: int(batch_fret_target_lengths[s][b])]
                actual_transcriptions[s].append(
                    [self.labels(i.item()) for i in indices]
                )

        pred_transcriptions = [[] for _ in range(self.num_strings)]
        for s in range(self.num_strings):
            for p in batch_tabs_log_probs[s].transpose(1, 0):  # T x B x C -> B x T x C
                pred_transcriptions[s].append(self.decoder(p))  # T x C -> L
        return actual_transcriptions, pred_transcriptions

    def _compute_cer_onsets_ratio(
        self, actual_transcriptions, pred_transcriptions, batch_size, hist
    ):
        strings_cers_step = []
        predicted_onsets = []
        total_onsets = []
        for s in range(self.num_strings):
            string_step_cer = 0
            string_predicted_onsets = []
            string_total_onsets = []
            for at, pt in zip(actual_transcriptions[s], pred_transcriptions[s]):
                at = [x for x in at if x != "*"]
                pt = [x for x in pt if x != "*"]
                string_total_onsets.append(len(at))
                string_predicted_onsets.append(len(pt))
                total_onsets.append(len(at))
                predicted_onsets.append(len(pt))
                if at == []:
                    at = ["-"]
                if pt == []:
                    pt = ["-"]
                c = cer(" ".join(at), " ".join(pt))
                string_step_cer += c
            string_step_cer /= batch_size
            strings_cers_step.append(string_step_cer)
            hist.strings_cers[s].append(string_step_cer)
            hist.strings_onsets_ratio[s].append(
                (sum(string_predicted_onsets) / sum(string_total_onsets))
                if sum(string_total_onsets) > 0
                else 0
            )
        step_cer = sum(strings_cers_step) / self.num_strings
        onsets_ratio = sum(predicted_onsets) / sum(total_onsets)
        hist.cer.append(step_cer)
        hist.onsets_ratio.append(onsets_ratio)
        return step_cer, onsets_ratio

    def _log_tensorboard_step(
        self,
        tb,
        step_loss,
        step_tab_loss,
        step_note_loss,
        step_mean_ctc_loss,
        step_cer,
        step_ner,
        onsets_ratio,
    ):
        tb.log_step("loss", float(step_loss.item()), self.current_update)
        tb.log_step("loss/tabs", float(step_tab_loss.item()), self.current_update)
        tb.log_step("loss/mean_ctc", float(step_mean_ctc_loss.item()), self.current_update)
        if self.note_criterion is not None:
            tb.log_step("loss/notes", float(step_note_loss.item()), self.current_update)
            # tb.log_step("ner", float(step_ner), self.current_update) # TODO: not implemented
        tb.log_step("cer", float(step_cer), self.current_update)
        tb.log_step("onset_ratio", float(onsets_ratio), self.current_update)

    def _log_tensorboard_epoch(self, tb, hist):
        tb.log_epoch(
            "lr", float(self.optimizer.param_groups[0]["lr"]), self.current_epoch
        )
        for s in range(self.num_strings):
            tb.log_epoch(
                f"loss_string/string-{s+1}",
                float(np.mean(hist.strings_losses[s])),
                self.current_epoch,
            )
            tb.log_epoch(
                f"cer_string/string-{s+1}",
                float(np.mean(hist.strings_cers[s])),
                self.current_epoch,
            )
            tb.log_epoch(
                f"onset_ratio_string/string-{s+1}",
                float(np.mean(hist.strings_onsets_ratio[s])),
                self.current_epoch,
            )
        tb.log_epoch("loss", float(np.mean(hist.loss)), self.current_epoch)
        tb.log_epoch("cer", float(np.mean(hist.cer)), self.current_epoch)
        tb.log_epoch(
            "onset_ratio", float(np.mean(hist.onsets_ratio)), self.current_epoch
        )
        tb.log_epoch("loss/tabs", float(np.mean(hist.loss_tabs)), self.current_epoch)
        tb.log_epoch("loss/mean_ctc", float(np.mean(hist.mean_ctc_loss)), self.current_epoch)
        if self.note_criterion is not None:
            tb.log_epoch(
                "loss/notes", float(np.mean(hist.loss_notes)), self.current_epoch
            )
            # tb.log_epoch("ner", float(np.mean(hist.ner)), self.current_epoch) # TODO: not implemented

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

        for idx, (batch_features, batch_targets) in train_step_it:
            try:
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                
                (
                    batch_size,
                    batch_x,
                    batch_ffm,
                    batch_frets_seq,
                    batch_notes_seq,
                    batch_fret_target_lengths,
                ) = self._get_data_items(batch_features, batch_targets)           

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.segment_audio_frames is not None:
                        batch_tabs_logits = []
                        batch_notes_log_probs = []
                        batch_x = segment_feature(batch_x, segment_length=self.segment_audio_frames)
                        batch_ffm = segment_feature(batch_ffm, segment_length=self.segment_audio_frames)
                        if self.use_tqdm:
                            batch_features = tqdm(
                                zip(batch_x, batch_ffm),
                                unit="segment",
                                total=len(batch_x),
                                dynamic_ncols=True,
                                colour="#00aaff",
                                leave=False,
                            )
                        for batch_x_segment, batch_ffm_segment in batch_features:
                            if not self.use_tqdm:
                                print('.', end='', flush=True)
                            output_segment = self.model(
                                batch_x_segment.to(self.device),
                                ffm=(
                                    batch_ffm_segment.to(self.device)
                                    if batch_ffm_segment.nelement() != 0
                                    else None
                                ),
                                return_logits=True,
                                freeze_feature_extractor=self.freeze_finetune_updates is not None
                                and self.current_update < self.freeze_finetune_updates, 
                                freeze_encoder=self.freeze_finetune_updates is not None
                                and self.current_update < self.freeze_finetune_updates and not self.freeze_only_feature_extractor,
                                **self.model_extra_forward_args,
                            )
                            batch_tabs_logits_segment, batch_notes_log_probs_segment = (
                                output_segment["tab"],
                                output_segment["notes"] if self.predict_notes else None,
                            )
                            batch_tabs_logits.append(batch_tabs_logits_segment)
                            if self.predict_notes:
                                batch_notes_log_probs.append(batch_notes_log_probs_segment)
                            torch.cuda.empty_cache()
                                
                        batch_tabs_logits = torch.cat(batch_tabs_logits, dim=2)
                        batch_tabs_log_probs = F.log_softmax(batch_tabs_logits, dim=-1)
                        
                        if self.predict_notes:
                            batch_notes_log_probs = torch.cat(batch_notes_log_probs, dim=2)
                    else:
                        output = self.model(
                            batch_x.to(self.device),
                            ffm=(
                                batch_ffm.to(self.device)
                                if batch_ffm.nelement() != 0
                                else None
                            ),
                            return_logits=True,
                            freeze_feature_extractor=self.freeze_finetune_updates is not None
                            and self.current_update < self.freeze_finetune_updates, 
                            freeze_encoder=self.freeze_finetune_updates is not None
                            and self.current_update < self.freeze_finetune_updates and not self.freeze_only_feature_extractor,
                            **self.model_extra_forward_args,
                        )

                        batch_tabs_logits, batch_notes_log_probs = (
                            output["tab"],
                            output["notes"],
                        )
                        batch_tabs_log_probs = F.log_softmax(batch_tabs_logits, dim=-1)
                        
                    # B x S x T x C -> S x T x B x C
                    batch_tabs_log_probs = batch_tabs_log_probs.permute(1, 2, 0, 3)
                    batch_tabs_logits = batch_tabs_logits.permute(1, 2, 0, 3)
                    tab_input_lengths = torch.stack(
                        [
                            torch.full(
                                (batch_tabs_log_probs.shape[2],),
                                batch_tabs_log_probs.shape[1],
                                dtype=torch.long,
                            )
                            for _ in range(self.num_strings)
                        ]
                    )  # (S, B)
                            
                    step_loss, step_tab_loss, step_note_loss, step_string_losses = (
                        self._compute_losses(
                            batch_tabs_logits,
                            batch_tabs_log_probs,
                            batch_frets_seq,
                            tab_input_lengths,
                            batch_fret_target_lengths,
                            batch_notes_log_probs,
                            batch_notes_seq,
                        )
                    )
                    
                with torch.no_grad():
                    step_mean_ctc_loss = torch.stack([
                        F.ctc_loss(batch_tabs_log_probs[s], batch_frets_seq[s], tab_input_lengths[s], batch_fret_target_lengths[s]) 
                        for s in range(self.num_strings)
                    ]).mean()
                    train_hist.mean_ctc_loss.append(float(step_mean_ctc_loss.detach().cpu().item()))
                
                if self.grad_norm_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm_clip
                    )

                if (
                    idx % self.accumulation_steps == 0
                    or idx == len(self.train_data_loader) - 1
                    and self.current_update > 0
                ):
                    train_step_it.set_description(f"Epoch {self.current_epoch} [#]")
                    self._update_model_weights()
                    self._update_lr_scheduler_updates()

                    if (
                        self.save_interval_updates is not None
                        and self.current_update % self.save_interval_updates == 0
                    ):
                        self.save_checkpoint(step=self.current_update)

                    # ============================================================

                    step_loss_log = float(step_loss.detach().cpu().item())
                    step_tab_loss_log = float(step_tab_loss.detach().cpu().item())
                    train_hist.loss.append(step_loss_log)
                    train_hist.loss_tabs.append(step_tab_loss_log)
                    if self.note_criterion is not None:
                        step_note_loss_log = float(step_note_loss.detach().cpu().item())
                        train_hist.loss_notes.append(step_note_loss_log)
                    for s in range(self.num_strings):
                        train_hist.strings_losses[s].append(
                            step_string_losses[s].detach().cpu().item()
                        )

                    if self.note_criterion is not None:
                        batch_notes_log_probs = batch_notes_log_probs.detach().cpu()
                        batch_notes_seq = batch_notes_seq.detach().cpu()
                    batch_tabs_log_probs = batch_tabs_log_probs.detach().cpu()
                    
                    batch_frets_seq = [
                        fret_seq_string.detach().cpu().long()
                        for fret_seq_string in batch_frets_seq
                    ]
                    actual_transcriptions, pred_transcriptions = (
                        self._generate_tab_transcriptions(
                            batch_tabs_log_probs,
                            batch_frets_seq,
                            batch_fret_target_lengths,
                        )
                    )
                    step_cer, onsets_ratio = self._compute_cer_onsets_ratio(
                        actual_transcriptions,
                        pred_transcriptions,
                        batch_size,
                        train_hist,
                    )

                    step_ner = 0  # TODO: implement note error rate

                    self._log_tensorboard_step(
                        self.train_tensorboard,
                        step_loss,
                        step_tab_loss,
                        step_note_loss,
                        step_mean_ctc_loss,
                        step_cer,
                        step_ner,
                        onsets_ratio,
                    )

                log_prog = [
                    f"update: {self.current_update}",
                    f"lr: {float(self.optimizer.param_groups[0]['lr']):.2e}",
                    f"loss: {step_loss_log:.2f}↓",
                    f"loss_tab: {step_tab_loss_log:.2f}↓",
                    f"mean_ctc: {step_mean_ctc_loss:.2f}↓",
                    f"cer: {step_cer:.2f}↓",
                    f"onset_r: {onsets_ratio:.2f}→1",
                ]
                if self.note_criterion is not None:
                    log_prog.append(f"loss_note: {step_note_loss_log:.2f}↓")
                    # log_prog.append(f"ner: {step_ner:.2f}↓")  # TODO: not implemented
                if self.accumulation_steps > 1:
                    log_prog.append(
                        f"i/ac: {(idx % self.accumulation_steps)+1}/{self.accumulation_steps}"
                    )
                if self.use_amp:
                    log_prog.append(
                        f"scale: {self.scaler.get_scale() if self.use_amp else 1.0:.2f}"
                    )
                if self.use_tqdm:
                    train_step_it.set_description(f"Epoch {self.current_epoch} [>]")
                    train_step_it.set_postfix_str(", ".join(log_prog))

                valid_hist_average = None
                if (
                    self.validate_interval_updates is not None
                    and self.current_update % self.validate_interval_updates == 0
                ):
                    valid_hist_average = self.evaluate()

                if (
                    self.log_interval_steps > 0
                    and idx % self.log_interval_steps == 0
                ):
                    avg_metrics = train_hist.average_last_n(self.log_interval_steps)
                    log_interval_steps = {
                        "epoch": f"{self.current_epoch}",
                        "updates": f"{self.current_update}",
                        "lr": f"{float(self.optimizer.param_groups[0]['lr']):.2e}",
                        "loss_avg": f"{avg_metrics.get('loss'):.2f}",
                        "loss_tabs_avg": f"{avg_metrics.get('loss_tabs', 0):.2f}",
                        "mean_ctc_loss_avg": f"{avg_metrics.get('mean_ctc_loss', 0):.2f}",
                        "onset_ratio_avg": f"{avg_metrics.get('onsets_ratio', 0):.2f}",
                        "cer_avg": f"{avg_metrics.get('cer', 0):.2f}",
                    }
                    if self.note_criterion is not None:
                        log_interval_steps["loss_notes_avg"] = (
                            f"{avg_metrics.get('loss_notes', 0):.2f}"
                        )
                        # log_interval_steps["ner_avg"] = f"{avg_metrics.get('ner', 0):.2f}"  # TODO: not implemented
                    if valid_hist_average is not None:
                        log_interval_steps["valid_loss"] = f"{valid_hist_average.get('loss', 0):.2f}"
                        log_interval_steps["valid_loss_tabs"] = (
                            f"{valid_hist_average.get('loss_tabs', 0):.2f}"
                        )
                        log_interval_steps["valid_cer"] = (
                            f"{valid_hist_average.get('cer', 0):.2f}"
                        )
                        log_interval_steps["valid_cer_tabs_strings"] = (
                            f"{valid_hist_average.get('cer_tabs_strings', 0):.2f}"
                        )
                        if self.note_criterion is not None:
                            log_interval_steps["valid_loss_notes"] = (
                                f"{valid_hist_average.get('loss_notes', 0):.2f}"
                            )
                            # log_interval_steps["valid_ner"] = (
                            #     f"{valid_hist_average.get('ner', 0):.2f}"
                            # ) # TODO: not implemented
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

        # self.save_checkpoint(step=self.current_update)

        self._log_tensorboard_epoch(self.train_tensorboard, train_hist)

        return train_hist.average()

    def _log_heatmap(
        self,
        tb_writer,
        tag,
        matrix,
        global_step
    ):
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
        batch_notes_seq,
        batch_tabs_log_probs,
        batch_notes_log_probs,
        ex_idx,
        heatmap=True,
        max_width=1000
    ):
        # TODO limit size
        if len(batch_x[ex_idx].shape) == 3:
            tb.log_image(f"features/x", batch_x[ex_idx],
                self.current_update)
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
                    tb, "heatmaps/features/ffm", batch_ffm[ex_idx], self.current_update
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
                    tb, "heatmaps/ffm_emb", output["ffm_emb"][ex_idx, :, :], self.current_update
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
                output["z"][ex_idx, :, :].squeeze().unsqueeze(0),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb, "heatmaps/z", output["z"][ex_idx, :, :].squeeze(), self.current_update
                )
        if output.get("masked_z", None) is not None:
            tb.log_image(
                f"masked_z",
                output["masked_z"][ex_idx, :, :].squeeze().unsqueeze(0),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb, "heatmaps/masked_z", output["masked_z"][ex_idx, :, :].squeeze(), self.current_update
                )
        if output.get("q", None) is not None:
            tb.log_image(
                f"q",
                output["q"][ex_idx, :, :].squeeze().unsqueeze(0),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb, "heatmaps/q", output["q"][ex_idx, :, :].squeeze(), self.current_update
                )
        if output.get("c", None) is not None:
            tb.log_image(
                f"c",
                output["c"][ex_idx, :, :].squeeze().unsqueeze(0),
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb, "heatmaps/c", output["c"][ex_idx, :, :].squeeze(), self.current_update
                )
        if self.predict_notes:
            tb.log_image(
                f"notes/target",
                min_max_normalize(batch_notes_seq[ex_idx, :, :], 1, 0).unsqueeze(0) * 255,
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(tb, "heatmaps/notes/target", batch_notes_seq[ex_idx, :, :], self.current_update)
            if batch_notes_log_probs is not None and len(batch_notes_log_probs.shape) == 4:
                exp_output_non_blank = torch.exp(batch_notes_log_probs[ex_idx, 1, :, :])
                tb.log_image(
                    f"notes/exp_output_non_blank",
                    min_max_normalize(
                        exp_output_non_blank, 1, 0
                    )
                    .unsqueeze(0)
                    .transpose(1, 2)
                    * 255,
                    self.current_update,
                )
                tb.log_image(
                    f"notes/exp_output_thresh",
                    min_max_normalize(
                        (exp_output_non_blank > 0.5).float(), 1, 0
                    )
                    .unsqueeze(0)
                    .transpose(1, 2)
                    * 255,
                    self.current_update,
                )
                tb.log_image(
                    f"notes/exp_output_blank",
                    min_max_normalize(
                        torch.exp(batch_notes_log_probs[ex_idx, 0, :, :]), 1, 0
                    )
                    .unsqueeze(0)
                    .transpose(1, 2)
                    * 255,
                    self.current_update,
                )
                if heatmap:
                    self._log_heatmap(
                        tb,
                        "heatmaps/notes/output_non_blank",
                        batch_notes_log_probs[ex_idx, 1, :, :].unsqueeze(0).transpose(1, 2),
                        self.current_update,
                    )
                    self._log_heatmap(
                        tb,
                        "heatmaps/notes/output_blank",
                        batch_notes_log_probs[ex_idx, 0, :, :].unsqueeze(0).transpose(1, 2),
                        self.current_update,
                    )

        for s in range(self.num_strings):
            tb.log_image(
                f"output/string-{s+1}",
                min_max_normalize(
                    torch.exp(batch_tabs_log_probs[s, :, ex_idx, :]), 1, 0
                )
                .unsqueeze(0)
                .transpose(1, 2)
                * 255,
                self.current_update,
            )
            if heatmap:
                self._log_heatmap(
                    tb,
                    f"heatmaps/output/string-{s+1}",
                    batch_tabs_log_probs[s, :, ex_idx, :]
                    .transpose(0, 1),
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
                colour="#22ffaa",
                leave=False,
            )
            if self.use_tqdm
            else enumerate(data_loader)
        )

        valid_hist = self.History(self.num_strings)

        for idx, (batch_features, batch_targets) in evaluation_step_it:

            (
                batch_size,
                batch_x,
                batch_ffm,
                batch_frets_seq,
                batch_notes_seq,
                batch_fret_target_lengths,
            ) = self._get_data_items(batch_features, batch_targets)

            with torch.no_grad():
                output = self.model(
                    batch_x.to(self.device),
                    ffm=(
                        batch_ffm.to(self.device) if batch_ffm.nelement() != 0 else None
                    ),
                    return_logits=True,
                    **self.model_extra_forward_args,
                )

                batch_tabs_logits, batch_notes_log_probs = (
                    output["tab"],
                    output["notes"],
                )
                batch_tabs_log_probs = F.log_softmax(batch_tabs_logits, dim=-1)
                # B x S x T x C -> S x T x B x C
                batch_tabs_log_probs = batch_tabs_log_probs.permute(1, 2, 0, 3)
                batch_tabs_logits = batch_tabs_logits.permute(1, 2, 0, 3)
                tab_input_lengths = torch.stack(
                    [
                        torch.full(
                            (batch_tabs_log_probs.shape[2],),
                            batch_tabs_log_probs.shape[1],
                            dtype=torch.long,
                        )
                        for _ in range(self.num_strings)
                    ]
                )

                step_loss, step_tab_loss, step_note_loss, step_string_losses = (
                    self._compute_losses(
                        batch_tabs_logits,
                        batch_tabs_log_probs,
                        batch_frets_seq,
                        tab_input_lengths,
                        batch_fret_target_lengths,
                        batch_notes_log_probs,
                        batch_notes_seq,
                        training=False,
                    )
                )
                with torch.no_grad():
                    step_mean_ctc_loss = torch.stack([
                        F.ctc_loss(batch_tabs_log_probs[s], batch_frets_seq[s], tab_input_lengths[s], batch_fret_target_lengths[s]) 
                        for s in range(self.num_strings)
                    ]).mean()
                    valid_hist.mean_ctc_loss.append(float(step_mean_ctc_loss.detach().cpu().item()))

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
                    batch_notes_seq,
                    batch_tabs_log_probs,
                    batch_notes_log_probs,
                    0,
                    # random.choice(range(batch_size)),
                )

            step_loss_log = float(step_loss.detach().cpu().item())
            step_tab_loss_log = float(step_tab_loss.detach().cpu().item())
            valid_hist.loss.append(step_loss_log)
            valid_hist.loss_tabs.append(step_tab_loss_log)
            if self.note_criterion is not None:
                step_note_loss_log = float(step_note_loss.detach().cpu().item())
                valid_hist.loss_notes.append(step_note_loss_log)
            for s in range(self.num_strings):
                valid_hist.strings_losses[s].append(
                    step_string_losses[s].detach().cpu().item()
                )

            if self.note_criterion is not None:
                batch_notes_log_probs = batch_notes_log_probs.detach().cpu()
                batch_notes_seq = batch_notes_seq.detach().cpu()
            batch_tabs_log_probs = batch_tabs_log_probs.detach().cpu()
            batch_frets_seq = [
                fret_seq_string.detach().cpu().long()
                for fret_seq_string in batch_frets_seq
            ]

            actual_transcriptions, pred_transcriptions = (
                self._generate_tab_transcriptions(
                    batch_tabs_log_probs,
                    batch_frets_seq,
                    batch_fret_target_lengths,
                )
            )

            step_cer, onsets_ratio = self._compute_cer_onsets_ratio(
                actual_transcriptions,
                pred_transcriptions,
                batch_size,
                valid_hist,
            )

            step_ner = 0  # TODO: implement note error rate

            log_prog = [
                f"epoch {self.current_epoch}",
                f"update: {self.current_update}",
                f"loss: {step_loss_log:.2f}↓",
                f"loss_tab: {step_tab_loss_log:.2f}↓",
                f"mean_ctc: {step_mean_ctc_loss:.2f}↓",
                f"cer: {step_cer:.2f}↓",
                f"onset_r: {onsets_ratio:.2f}→1",
            ]
            if self.note_criterion is not None:
                log_prog.append(f"loss_note: {step_note_loss_log:.2f}↓")
                # log_prog.append(f"ner: {step_ner:.2f}↓") # TODO: not implemented
            if self.use_tqdm:
                evaluation_step_it.set_description(log_prog[0])
                evaluation_step_it.set_postfix_str(", ".join(log_prog[1:]))

            if self.print_examples_validation:
                print(f"{'-'*(os.get_terminal_size().columns)}")
                for s in range(self.num_strings):
                    print("GT", s, actual_transcriptions[s][0])
                    print("PR", s, pred_transcriptions[s][0])
                    print()
                print(f"{'-'*(os.get_terminal_size().columns)}")

            torch.cuda.empty_cache()

            if self.log_interval_steps > 0 and idx % self.log_interval_steps == 0:
                avg_metrics = valid_hist.average_last_n(self.log_interval_steps)
                log_interval_steps = {
                    "epoch": f"{self.current_epoch}",
                    "updates": f"{self.current_update}",
                    "loss_avg": f"{avg_metrics.get('loss', 0):.2f}",
                    "loss_tabs_avg": f"{avg_metrics.get('loss_tabs', 0):.2f}",
                    "mean_ctc_avg": f"{avg_metrics.get('mean_ctc_loss', 0):.2f}",
                    "onset_ratio_avg": f"{avg_metrics.get('onsets_ratio', 0):.2f}",
                    "cer_avg": f"{avg_metrics.get('cer', 0):.2f}",
                }
                if self.note_criterion is not None:
                    log_interval_steps["loss_notes_avg"] = (
                        f"{avg_metrics.get('loss_notes', 0):.2f}"
                    )
                    # log_interval_steps["ner_avg"] = f"{avg_metrics.get('ner', 0):.2f}"  # TODO: not implemented
                if self.use_tqdm:
                    evaluation_step_it.clear()
                logger.info(log_interval_steps)

        if log_to_tensorboard:
            self._log_tensorboard_epoch(self.valid_tensorboard, valid_hist)
            self._log_tensorboard_step(
                self.valid_tensorboard,
                valid_hist.average().get("loss"),
                valid_hist.average().get("loss_tabs"),
                valid_hist.average().get("loss_notes"),
                valid_hist.average().get("mean_ctc_loss"),
                valid_hist.average().get("cer"),
                valid_hist.average().get("ner"), # TODO: not implemented
                valid_hist.average().get("onsets_ratio"),
            )

        if self.best_loss is None:
            self.best_loss = valid_hist.average().get("loss")
        if valid_hist.average().get("loss") < self.best_loss and self.is_training:
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
            
            if (
                self.max_updates is not None
                and self.current_update > self.max_updates
            ):
                break 
            
            self.current_epoch += 1

            if self.current_epoch > self.max_epochs:
                logger.info(f"Stopping training due to max_epochs={self.max_epochs}")
                self.save_checkpoint()
                valid_hist_avg = self.evaluate()
                epoch_info = {
                    "valid_loss": f"{valid_hist_avg.get('loss'):.3f}",
                    "valid_loss_tabs": f"{valid_hist_avg.get('loss_tabs'):.3f}",
                    "valid_loss_notes": f"{valid_hist_avg.get('loss_notes'):.3f}",
                    "valid_cer": f"{valid_hist_avg.get('cer'):.3f}",
                    "valid_strings_cers": [
                        f"{float(s):.2f}"
                        for s in valid_hist_avg.get("strings_cers", [0])
                    ],
                    "valid_onsets_ratio": f"{valid_hist_avg.get('onsets_ratio'):.3f}",
                    "valid_strings_onsets_ratio": [
                        f"{float(s):.2f}"
                        for s in valid_hist_avg.get("strings_onsets_ratio")
                    ],
                    # "valid_ner": f"{valid_hist_avg.get('ner'):.3f}", # TODO: not implemented
                }
                logger.info(f"Training done: {epoch_info}")
                break

            logger.info(f"Training epoch {self.current_epoch}")
            train_hist_avg = self.train_one_epoch()

            epoch_info = {
                "loss": f"{train_hist_avg.get('loss', 0):.3f}",
                "loss_tabs": f"{train_hist_avg.get('loss_tabs', 0):.3f}",
                "loss_notes": f"{train_hist_avg.get('loss_notes', 0):.3f}",
                "cer": f"{train_hist_avg.get('cer', 0):.3f}",
                "strings_cers": [
                    f"{float(s):.2f}" for s in train_hist_avg.get("strings_cers", [0])
                ],
                "onsets_ratio": f"{train_hist_avg.get('onsets_ratio', 0):.3f}",
                "strings_onsets_ratio": [
                    f"{float(s):.2f}"
                    for s in train_hist_avg.get("strings_onsets_ratio", 0)
                ],
                # "ner": f"{train_hist_avg.get('ner', 0):.3f}", # TODO: not implemented
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
                        "valid_loss_tabs": f"{valid_hist_avg.get('loss_tabs', 0):.3f}",
                        "valid_loss_notes": f"{valid_hist_avg.get('loss_notes', 0):.3f}",
                        "valid_cer": f"{valid_hist_avg.get('cer', 0):.3f}",
                        "valid_strings_cers": [
                            f"{float(s):.2f}"
                            for s in valid_hist_avg.get("strings_cers", [0])
                        ],
                        "valid_onsets_ratio": f"{valid_hist_avg.get('onsets_ratio', 0):.3f}",
                        "valid_strings_onsets_ratio": [
                            f"{float(s):.2f}"
                            for s in valid_hist_avg.get("strings_onsets_ratio", 0)
                        ],
                        # "valid_ner": f"{valid_hist_avg.get('ner', 0):.3f}", # TODO: not implemented
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
                "test_loss_tabs": f"{test_hist_avg.get('loss_tabs', 0):.3f}",
                "test_loss_notes": f"{test_hist_avg.get('loss_notes', 0):.3f}",
                "test_cer": f"{test_hist_avg.get('cer', 0):.3f}",
                "test_strings_cers": [
                    f"{float(s):.2f}" for s in test_hist_avg.get("strings_cers", [0])
                ],
                "test_onsets_ratio": f"{test_hist_avg.get('onsets_ratio', 0):.3f}",
                "test_strings_onsets_ratio": [
                    f"{float(s):.2f}"
                    for s in test_hist_avg.get("strings_onsets_ratio", 0)
                ],
                # "test_ner": f"{test_hist_avg.get('ner', 0):.3f}", # TODO: not implemented
            }
            logger.info(f"Evaluation on test set: {info}")
