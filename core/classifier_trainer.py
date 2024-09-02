import os
import io
import logging
from typing import Union

import logging
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from core.data import AGTFrameDataset
from core.criterions import TabByStringFrameWiseLoss, NoteFrameWiseLoss
from core.trainer import Trainer
from utils.util import argmax_map, min_max_normalize
from utils.metrics import f_measure
import matplotlib.pyplot as plt
import PIL


logger = logging.getLogger(__name__)

# TODO: this code needs to be refactored (see ctc_trainer.py)

class ClassifierTrainer(Trainer):

    def __init__(self, cfg: DictConfig, run_dir: Union[os.PathLike, str]):
        super().__init__(cfg, run_dir)

        # This trainer only supports the prediction of onsets and frets with the same shape 
        # as the tab. In practice, onsets == tab, and frets is the frame_level output of the tab.
        self.predict_onsets_and_frets = cfg.predict_onsets_and_frets
        self.predict_tab = cfg.predict_tab
        self.predict_notes = cfg.predict_notes
        if self.predict_onsets_and_frets:
            self.target_col_onsets = "tab"
            self.target_col_frets = "frets"
        elif self.predict_tab:
            self.target_col = "tab" 
            if cfg.prediction_type == "note_level":
                self.target_col = "tab"
            elif cfg.prediction_type == "frame_level":
                self.target_col = "frets"
        
        logger.info(f"Prediction type: {cfg.prediction_type}")

        self.apply_softmax = cfg.apply_softmax
        self.class_probabilities = cfg.class_probabilities
         
        self.tab_by_string_criterion = self.note_criterion = None
        if str(cfg.criterions.tab.name) == "TabByStringFrameWiseLoss":
            self.tab_by_string_criterion = TabByStringFrameWiseLoss(
                **cfg.criterions.tab.config
            )
        else:
            raise ValueError("A valid criterion for tab prediction must be specified")
        if cfg.predict_notes:
            if cfg.criterions.notes is None:
                raise ValueError("A valid criterion for note prediction must be specified")
            if str(cfg.criterions.notes.name) == "NoteFrameWiseLoss":
                self.note_criterion = NoteFrameWiseLoss(**cfg.criterions.notes.config)
        self.criterion_weights = (
            cfg.criterions.criterion_weights
            if "criterion_weights" in cfg.criterions
            else [1] * len(cfg.criterions) / len(cfg.criterions)  # BUG
        )

        self.valid_data_loader = AGTFrameDataset.get_valid_dataloader(
            cfg.data, self.audio_processor
        )
        self.train_data_loader = AGTFrameDataset.get_train_dataloader(
            cfg.data, self.audio_processor
        )
        if cfg.data.test_csv_file is not None:
            self.test_data_loader = AGTFrameDataset.get_test_dataloader(
                cfg.data, self.audio_processor
            )

        if (
            cfg.checkpoint.finetune_from_model is not None
            or cfg.evaluate_before_training
        ):
            self.evaluate()
            
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

        loss_epoch = []
        if self.predict_onsets_and_frets:
            loss_onsets_epoch = []
            loss_frets_epoch = []
            f_onsets_epoch = []
            f_frets_epoch = []
            p_onsets_epoch = []
            p_frets_epoch = []
            r_onsets_epoch = []
            r_frets_epoch = []
        elif self.predict_tab:
            loss_tabs_epoch = []
            f_tabs_epoch = []
            p_tabs_epoch = []
            r_tabs_epoch = []
        if self.predict_notes:
            loss_notes_epoch = []
            mse_notes_epoch = []
        strings_losses_epoch = [[] for _ in range(self.num_strings)]

        accumulation_steps = self.accumulation_steps

        for idx, (batch_features, batch_targets) in train_step_it:
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()

            batch_x = batch_features["x"].to(self.device)
            batch_ffm = batch_features["ffm"].to(self.device)
            if self.predict_onsets_and_frets:
                batch_onsets = batch_targets[self.target_col_onsets].to(self.device)
                batch_frets = batch_targets[self.target_col_frets].to(self.device)
            else:
                batch_tab = batch_targets[self.target_col].to(self.device)
            if self.predict_notes:
                batch_notes = batch_targets["notes"].to(self.device)

            freeze_encoder = (
                self.freeze_finetune_updates is not None
                and self.current_update < self.freeze_finetune_updates
            )

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(
                    batch_x.to(self.device),
                    ffm=(
                        batch_ffm.to(self.device) if batch_ffm.nelement() != 0 else None
                    ),
                    freeze_encoder=freeze_encoder,
                )
                step_loss = 0
                if self.predict_onsets_and_frets:
                    batch_predicted_frets, batch_predicted_onsets = (
                        output.frets,
                        output.onsets
                    )
                    step_onset_losses, step_onsets_string_losses = self.tab_by_string_criterion(
                        batch_predicted_onsets, batch_onsets
                    )
                    step_loss += step_onset_losses * self.criterion_weights[0]
                    step_frets_losses, step_frets_string_losses = self.tab_by_string_criterion(
                        batch_predicted_frets, batch_frets
                    )
                    step_string_losses = [
                        step_onsets_string_losses[s] + step_frets_string_losses[s]
                        for s in range(self.num_strings)
                    ]
                    step_loss += step_frets_losses * self.criterion_weights[0]
                else:
                    batch_predicted_tabs = output["tab"]
                    step_tab_losses, step_string_losses = self.tab_by_string_criterion(
                        batch_predicted_tabs, batch_tab
                    )
                    step_loss += step_tab_losses * self.criterion_weights[0]
                    
                step_note_loss = None
                if self.predict_notes:
                    batch_predicted_notes = output["notes"]
                    step_note_loss = self.note_criterion(
                        batch_predicted_notes, batch_notes
                    )
                    step_loss += step_note_loss * self.criterion_weights[1]

                if self.grad_norm_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_norm_clip
                    )
                step_loss /= accumulation_steps
                if self.use_amp:
                    self.scaler.scale(step_loss).backward()
                else:
                    step_loss.backward()
                    if not self.check_valid_gradients():
                        logger.warn(
                            f"Detected inf or nan values in gradients. Not updating model parameters."
                        )
                        self.optimizer.zero_grad()
                        step_loss = 0
                        accumulation_steps = self.accumulation_steps

            if self.predict_onsets_and_frets:
                batch_predicted_frets = batch_predicted_frets.detach().cpu()
                batch_predicted_onsets = batch_predicted_onsets.detach().cpu()
                batch_frets = batch_frets.detach().cpu()
                batch_onsets = batch_onsets.detach().cpu()
            elif self.predict_tab:
                batch_predicted_tabs = batch_predicted_tabs.detach().cpu()
                batch_tab = batch_tab.detach().cpu()
            if self.predict_notes:
                batch_predicted_notes = batch_predicted_notes.detach().cpu()
                batch_notes = batch_notes.detach().cpu()

            loss_epoch.append(step_loss.detach().cpu().item())
            
            if self.predict_onsets_and_frets:
                loss_onsets_epoch.append(step_onset_losses.detach().cpu().item())
                loss_frets_epoch.append(step_frets_losses.detach().cpu().item())
            elif self.predict_tab:
                loss_tabs_epoch.append(step_tab_losses.detach().cpu().item())
            if self.predict_notes:
                loss_notes_epoch.append(step_note_loss.detach().cpu().item())
            for s in range(self.num_strings):
                strings_losses_epoch[s].append(
                    step_string_losses[s].detach().cpu().item()
                )

            if idx % accumulation_steps == 0 or idx == len(self.train_data_loader) - 1:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.train_tensorboard.log_step(
                        "scale", float(self.scaler.get_scale()), self.current_update
                    )
                else:
                    self.optimizer.step()

                self.current_update += 1
                accumulation_steps = self.accumulation_steps

                if self.lr_scheduler_updates is not None:
                    if self.current_update % self.lr_scheduler_updates_interval == 0:
                        self.lr_scheduler_updates.step()
                        logger.info(
                            f"Learning rate updated to {float(self.optimizer.param_groups[0]['lr']):.2e}"
                        )
                        self.train_tensorboard.log_step(
                            "lr",
                            float(self.optimizer.param_groups[0]["lr"]),
                            self.current_update,
                        )

                self.train_tensorboard.log_step(
                    "loss", float(step_loss.item()), self.current_update
                )
                if self.predict_onsets_and_frets:
                    self.train_tensorboard.log_step(
                        "loss/onsets", float(step_onset_losses.item()), self.current_update
                    )
                    self.train_tensorboard.log_step(
                        "loss/frets", float(step_frets_losses.item()), self.current_update
                    )
                    loss_onsets_avg = sum(loss_onsets_epoch[-10:]) / (
                        10 if len(loss_onsets_epoch) > 10 else len(loss_onsets_epoch)
                    )
                    loss_frets_avg = sum(loss_frets_epoch[-10:]) / (
                        10 if len(loss_frets_epoch) > 10 else len(loss_frets_epoch)
                    )
                elif self.predict_tab:
                    self.train_tensorboard.log_step(
                        "loss/tabs", float(step_tab_losses.item()), self.current_update
                    )
                    loss_tabs_avg = sum(loss_tabs_epoch[-10:]) / (
                        10 if len(loss_tabs_epoch) > 10 else len(loss_tabs_epoch)
                    )
                if self.predict_notes:
                    self.train_tensorboard.log_step(
                        "loss/notes", float(step_note_loss.item()), self.current_update
                    )
                    loss_notes_avg = sum(loss_notes_epoch[-10:]) / (
                        10 if len(loss_notes_epoch) > 10 else len(loss_notes_epoch)
                    )

                loss_avg = sum(loss_epoch[-10:]) / (
                    10 if len(loss_epoch) > 10 else len(loss_epoch)
                )
                
                if (
                    self.validate_interval_updates is not None
                    and self.current_update % self.validate_interval_updates == 0
                ):
                    if self.predict_onsets_and_frets:
                        (
                            valid_loss,
                            valid_loss_onsets,
                            valid_loss_frets,
                            valid_f_onsets,
                            valid_f_frets,
                            valid_p_onsets,
                            valid_p_frets,
                            valid_r_onsets,
                            valid_r_frets,
                            valid_loss_notes,
                            valid_mse_notes,
                            _,
                            _,
                        ) = self.evaluate()
                    elif self.predict_tab:
                        (
                            valid_loss,
                            valid_loss_tabs,
                            valid_f_tabs,
                            valid_p_tabs,
                            valid_r_tabs,
                            valid_loss_notes,
                            valid_mse_notes,
                            _,
                            _,
                        ) = self.evaluate()
                        
                    info = {
                        "epoch": f"{self.current_epoch}",
                        "updates": f"{self.current_update}",
                        "lr": f"{float(self.optimizer.param_groups[0]['lr']):.2e}",
                        "valid_loss": f"{valid_loss:.3f}",
                    }
                    
                    if self.predict_onsets_and_frets:
                        info.update(
                            {
                                "valid_loss_onsets": f"{valid_loss_onsets:.3f}",
                                "valid_onsets_f": f"{valid_f_onsets:.3f}",
                                "valid_onsets_p": f"{valid_p_onsets:.3f}",
                                "valid_onsets_r": f"{valid_r_onsets:.3f}",
                                "valid_loss_frets": f"{valid_loss_frets:.3f}",
                                "valid_frets_f": f"{valid_f_frets:.3f}",
                                "valid_frets_p": f"{valid_p_frets:.3f}",
                                "valid_frets_r": f"{valid_r_frets:.3f}",
                            }
                        )
                    elif self.predict_tab:
                        info.update(
                            {
                                "valid_loss_tabs": f"{valid_loss_tabs:.3f}",
                                "valid_tabs_f": f"{valid_f_tabs:.3f}",
                                "valid_tabs_p": f"{valid_p_tabs:.3f}",
                                "valid_tabs_r": f"{valid_r_tabs:.3f}",
                            }
                        )
                    if self.predict_notes:
                        info.update(
                            {
                                "valid_loss_notes": f"{valid_loss_notes:.3f}",
                                "valid_notes_mse": f"{valid_mse_notes:.3f}",
                            }
                        )
                    logger.info(f"Validation done: {info}")

                if self.predict_onsets_and_frets:
                    batch_predicted_onsets = torch.nn.functional.softmax(
                        batch_predicted_onsets, dim=-1
                    )
                    batch_onsets_argmax = batch_onsets.argmax(dim=-1)
                    batch_predicted_onsets_argmax = torch.argmax(
                        batch_predicted_onsets, dim=-1
                    ).cpu()
                    batch_predicted_frets = torch.nn.functional.softmax(
                        batch_predicted_frets, dim=-1
                    )
                    batch_frets_argmax = batch_frets.argmax(dim=-1)
                    batch_predicted_frets_argmax = torch.argmax(
                        batch_predicted_frets, dim=-1
                    ).cpu()
                elif self.predict_tab:
                    batch_predicted_tabs = torch.nn.functional.softmax(
                        batch_predicted_tabs.float(), dim=-1
                    )
                    batch_tabs_argmax = batch_tab.argmax(dim=-1)
                    batch_predicted_tabs_argmax = torch.argmax(
                        batch_predicted_tabs, dim=-1
                    ).cpu()
                if self.predict_notes:
                    batch_predicted_notes = torch.nn.functional.softmax(
                        batch_predicted_notes, dim=-1
                    )
                    batch_notes_argmax = batch_notes.argmax(dim=-1)
                    batch_predicted_notes_argmax = torch.argmax(
                        batch_predicted_notes, dim=-1
                    ).cpu()

                if not self.class_probabilities:
                    if self.predict_onsets_and_frets:
                        batch_frets = batch_frets_argmax
                        batch_onsets = batch_onsets_argmax
                        batch_predicted_frets = batch_predicted_frets_argmax
                        batch_predicted_onsets = batch_predicted_onsets_argmax
                    elif self.predict_tab:
                        batch_tab = batch_tabs_argmax
                        batch_predicted_tabs = batch_predicted_tabs_argmax
                    if self.predict_notes:
                        batch_notes = batch_notes_argmax
                        batch_predicted_notes = batch_predicted_notes_argmax
                if idx % self.log_interval_steps == 0:

                    if self.predict_onsets_and_frets:
                        step_onsets_f, step_onsets_p, step_onsets_r = f_measure(
                            batch_predicted_onsets_argmax.numpy(),
                            batch_onsets_argmax.numpy(),
                            num_strings=self.num_strings,
                            offset_ratio=self.predict_tab,
                            ignore_empty=True  # Ignore empty strings because we are predicting small segments, not the whole tab, otherwise the metrics are not really meaningful (close to 0)
                        )
                        self.train_tensorboard.log_step(
                            "F/onsets", float(step_onsets_f), self.current_update
                        )
                        self.train_tensorboard.log_step(
                            "P/onsets", float(step_onsets_p), self.current_update
                        )
                        self.train_tensorboard.log_step(
                            "R/onsets", float(step_onsets_r), self.current_update
                        )
                        f_onsets_epoch.append(step_onsets_f)
                        p_onsets_epoch.append(step_onsets_p)
                        r_onsets_epoch.append(step_onsets_r)
                        step_frets_f, step_frets_p, step_frets_r = f_measure(
                            batch_predicted_frets_argmax.numpy(),
                            batch_frets_argmax.numpy(),
                            num_strings=self.num_strings,
                            offset_ratio=self.predict_tab,
                            ignore_empty=True
                        )
                        self.train_tensorboard.log_step(
                            "F/frets", float(step_frets_f), self.current_update
                        )
                        self.train_tensorboard.log_step(
                            "P/frets", float(step_frets_p), self.current_update
                        )
                        self.train_tensorboard.log_step(
                            "R/frets", float(step_frets_r), self.current_update
                        )
                        f_frets_epoch.append(step_frets_f)
                        p_frets_epoch.append(step_frets_p)
                        r_frets_epoch.append(step_frets_r)
                        
                        f_onsets_total_avg = sum(f_onsets_epoch) / len(f_onsets_epoch)
                        p_onsets_total_avg = sum(p_onsets_epoch) / len(p_onsets_epoch)
                        r_onsets_total_avg = sum(r_onsets_epoch) / len(r_onsets_epoch)
                        f_frets_total_avg = sum(f_frets_epoch) / len(f_frets_epoch)
                        p_frets_total_avg = sum(p_frets_epoch) / len(p_frets_epoch)
                        r_frets_total_avg = sum(r_frets_epoch) / len(r_frets_epoch)
                        
                    elif self.predict_tab:
                        step_tabs_f, step_tabs_p, step_tabs_r = f_measure(
                            batch_predicted_tabs_argmax.numpy(),
                            batch_tabs_argmax.numpy(),
                            num_strings=self.num_strings,
                            offset_ratio=self.predict_tab
                        )
                        self.train_tensorboard.log_step(
                            "F/tabs", float(step_tabs_f), self.current_update
                        )
                        self.train_tensorboard.log_step(
                            "P/tabs", float(step_tabs_p), self.current_update
                        )
                        self.train_tensorboard.log_step(
                            "R/tabs", float(step_tabs_r), self.current_update
                        )
                        f_tabs_epoch.append(step_tabs_f)
                        p_tabs_epoch.append(step_tabs_p)
                        r_tabs_epoch.append(step_tabs_r)
                        
                        f_tabs_total_avg = sum(f_tabs_epoch) / len(f_tabs_epoch)
                        p_tabs_total_avg = sum(p_tabs_epoch) / len(p_tabs_epoch)
                        r_tabs_total_avg = sum(r_tabs_epoch) / len(r_tabs_epoch)
                        
                    if self.predict_notes:
                        step_notes_mse = torch.nn.functional.mse_loss(
                            batch_predicted_notes, batch_notes.float()
                        )
                        self.train_tensorboard.log_step(
                            "mse/notes", float(step_notes_mse), self.current_update
                        )
                        mse_notes_epoch.append(step_notes_mse)

                        mse_notes_total_avg = sum(mse_notes_epoch) / len(mse_notes_epoch)

                    log_interval_steps = {
                        "epoch": f"{self.current_epoch}",
                        "updates": f"{self.current_update}",
                        "lr": f"{float(self.optimizer.param_groups[0]['lr']):.2e}",
                        "avg_loss": f"{loss_avg:.3f}",
                    }
                    if self.predict_onsets_and_frets:
                        log_interval_steps.update(
                            {
                                "avg_loss_onsets": f"{loss_onsets_avg:.3f}",
                                "avg_onsets_f": f"{f_onsets_total_avg:.3f}",
                                "avg_onsets_p": f"{p_onsets_total_avg:.3f}",
                                "avg_onsets_r": f"{r_onsets_total_avg:.3f}",
                                "avg_loss_frets": f"{loss_frets_avg:.3f}",
                                "avg_frets_f": f"{f_frets_total_avg:.3f}",
                                "avg_frets_p": f"{p_frets_total_avg:.3f}",
                                "avg_frets_r": f"{r_frets_total_avg:.3f}",
                            }
                        )
                    elif self.predict_tab:
                        log_interval_steps.update(
                            {
                                "avg_loss_tabs": f"{loss_tabs_avg:.3f}",
                                "avg_tabs_f": f"{f_tabs_total_avg:.3f}",
                                "avg_tabs_p": f"{p_tabs_total_avg:.3f}",
                                "avg_tabs_r": f"{r_tabs_total_avg:.3f}"
                            }
                        )
                    if self.predict_notes:
                        log_interval_steps.update(
                            {
                                "avg_loss_notes": f"{loss_notes_avg:.3f}",
                                "avg_notes_mse": f"{mse_notes_total_avg:.3f}"
                            }
                        )
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

            log = [
                f"lr: {float(self.optimizer.param_groups[0]['lr']):.2e}",
                f"updates: {self.current_update}",
                f"loss: {loss_avg:.3f}↓",
                # f"loss_tabs: {loss_tabs_avg:.3f}",
                # f"tabs_f: {f_tabs_total_avg:.3f}"
            ]
            if self.predict_onsets_and_frets:
                log.append(f"loss_onsets: {loss_onsets_avg:.3f}↓")
                log.append(f"onsets_f: {f_onsets_total_avg:.3f}→1")
                log.append(f"loss_frets: {loss_frets_avg:.3f}↓")
                log.append(f"frets_f: {f_frets_total_avg:.3f}→1")
            elif self.predict_tab:
                log.append(f"loss_tabs: {loss_tabs_avg:.3f}")
                log.append(f"tabs_f: {f_tabs_total_avg:.3f}→1")
            if self.predict_notes:
                log.append(f"loss_notes: {loss_notes_avg:.3f}↓")
                log.append(f"notes_mse: {mse_notes_total_avg:.3f}↓")
            if self.accumulation_steps > 1:
                log.append(f"i/ac: {(idx % accumulation_steps)+1}/{accumulation_steps}")
            if self.use_tqdm:
                train_step_it.set_postfix_str(", ".join(log))

        self.save_checkpoint(step=self.current_update)

        loss_epoch = sum(loss_epoch) / len(loss_epoch)
        epoch_string_losses = [
            sum(string_losses_epoch) / len(string_losses_epoch)
            for string_losses_epoch in strings_losses_epoch
        ]
        if self.predict_onsets_and_frets:
            loss_onsets_epoch = sum(loss_onsets_epoch) / len(loss_onsets_epoch)
            loss_frets_epoch = sum(loss_frets_epoch) / len(loss_frets_epoch)
            f_onsets_epoch = sum(f_onsets_epoch) / len(f_onsets_epoch)
            f_frets_epoch = sum(f_frets_epoch) / len(f_frets_epoch)
            p_onsets_epoch = sum(p_onsets_epoch) / len(p_onsets_epoch)
            p_frets_epoch = sum(p_frets_epoch) / len(p_frets_epoch)
            r_onsets_epoch = sum(r_onsets_epoch) / len(r_onsets_epoch)
            r_frets_epoch = sum(r_frets_epoch) / len(r_frets_epoch)
        elif self.predict_tab:    
            loss_tabs_epoch = sum(loss_tabs_epoch) / len(loss_tabs_epoch)
            f_tabs_epoch = sum(f_tabs_epoch) / len(f_tabs_epoch)
            p_tabs_epoch = sum(p_tabs_epoch) / len(p_tabs_epoch)
            r_tabs_epoch = sum(r_tabs_epoch) / len(r_tabs_epoch)
        
        if self.predict_notes:
            loss_notes_epoch = sum(loss_notes_epoch) / len(loss_notes_epoch)
            mse_notes_epoch = sum(mse_notes_epoch) / len(mse_notes_epoch)
        else:
            mse_notes_epoch = loss_notes_epoch = None

        self.train_tensorboard.log_epoch(
            "lr", float(self.optimizer.param_groups[0]["lr"]), self.current_epoch
        )

        for s in range(self.num_strings):
            self.train_tensorboard.log_epoch(
                f"string-{s+1}/loss", float(epoch_string_losses[s]), self.current_epoch
            )
        self.train_tensorboard.log_epoch("loss", loss_epoch, self.current_epoch)
        if self.predict_onsets_and_frets:
            self.train_tensorboard.log_epoch(
                "loss/onsets", loss_onsets_epoch, self.current_epoch
            )
            self.train_tensorboard.log_epoch(
                "loss/frets", loss_frets_epoch, self.current_epoch
            )
            self.train_tensorboard.log_epoch("F/onsets", f_onsets_epoch, self.current_epoch)
            self.train_tensorboard.log_epoch("P/onsets", p_onsets_epoch, self.current_epoch)
            self.train_tensorboard.log_epoch("R/onsets", r_onsets_epoch, self.current_epoch)
            self.train_tensorboard.log_epoch("F/frets", f_frets_epoch, self.current_epoch)
            self.train_tensorboard.log_epoch("P/frets", p_frets_epoch, self.current_epoch)
            self.train_tensorboard.log_epoch("R/frets", r_frets_epoch, self.current_epoch)
        elif self.predict_tab:
            self.train_tensorboard.log_epoch(
                "loss/tabs", loss_tabs_epoch, self.current_epoch
            )
            self.train_tensorboard.log_epoch("F/tabs", f_tabs_epoch, self.current_epoch)
            self.train_tensorboard.log_epoch("P/tabs", p_tabs_epoch, self.current_epoch)
            self.train_tensorboard.log_epoch("R/tabs", r_tabs_epoch, self.current_epoch)
        if self.predict_notes:
            self.train_tensorboard.log_epoch(
                "loss/notes", loss_notes_epoch, self.current_epoch
            )
            self.train_tensorboard.log_epoch("mse/notes", mse_notes_epoch, self.current_epoch)
        
        if self.predict_onsets_and_frets:
            return (
                loss_epoch,
                loss_onsets_epoch,
                loss_frets_epoch,
                f_onsets_epoch,
                f_frets_epoch,
                p_onsets_epoch,
                p_frets_epoch,
                r_onsets_epoch,
                r_frets_epoch,
                loss_notes_epoch,
                mse_notes_epoch,
            )
        elif self.predict_tab:
            return (
                loss_epoch,
                loss_tabs_epoch,
                loss_notes_epoch,
                f_tabs_epoch,
                p_tabs_epoch,
                r_tabs_epoch,
                loss_notes_epoch,
                mse_notes_epoch
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

        loss_epoch = []
        if self.predict_onsets_and_frets:
            loss_onsets_epoch = []
            loss_frets_epoch = []
            f_onsets_epoch = []
            f_frets_epoch = []
            p_onsets_epoch = []
            p_frets_epoch = []
            r_onsets_epoch = []
            r_frets_epoch = []
        elif self.predict_tab:
            loss_tabs_epoch = []
            f_tabs_epoch = []
            p_tabs_epoch = []
            r_tabs_epoch = []
        if self.predict_notes:
            loss_notes_epoch = []
            mse_notes_epoch = []
        strings_losses_epoch = [[] for _ in range(self.num_strings)]

        predicted_tabs = []
        target_tabs = []

        for idx, (batch_features, batch_targets) in evaluation_step_it:
            self.optimizer.zero_grad()

            batch_x = batch_features["x"].to(self.device)
            batch_ffm = batch_features["ffm"].to(self.device)
            if self.predict_onsets_and_frets:
                batch_onsets = batch_targets[self.target_col_onsets].to(self.device)
                batch_frets = batch_targets[self.target_col_frets].to(self.device)
            elif self.predict_tab:
                batch_tab = batch_targets[self.target_col].to(self.device)
            batch_notes = batch_targets["notes"].to(self.device)

            if idx == 0:
                ex_idx = 0  # random.randint(0, len(batch_features)-1)
                
                self.valid_tensorboard.log_image(
                    f"notes/target",
                    batch_notes[ex_idx, :, :].unsqueeze(0).transpose(1, 2) * 255,
                    self.current_epoch,
                )
                if self.predict_onsets_and_frets:
                    for s in range(self.num_strings):
                        self.valid_tensorboard.log_image(
                            f"target/onsets/string-{s+1}",
                            batch_onsets[ex_idx, s, :, :].unsqueeze(0).transpose(1, 2) * 255,
                            self.current_epoch,
                        )
                        self.valid_tensorboard.log_image(
                            f"target/frets/string-{s+1}",
                            batch_frets[ex_idx, s, :, :].unsqueeze(0).transpose(1, 2) * 255,
                            self.current_epoch,
                        )                    
                elif self.predict_tab:
                    for s in range(self.num_strings):
                        self.valid_tensorboard.log_image(
                            f"target/string-{s+1}",
                            batch_tab[ex_idx, s, :, :].unsqueeze(0).transpose(1, 2) * 255,
                            self.current_epoch,
                        )

            with torch.no_grad():
                output = self.model(
                    batch_x.to(self.device),
                    ffm=(
                        batch_ffm.to(self.device) if batch_ffm.nelement() != 0 else None
                    ),
                )
            if self.predict_onsets_and_frets:
                batch_predicted_frets, batch_predicted_onsets = (
                    output["frets"],
                    output["onsets"]
                )
            elif self.predict_tab:
                batch_predicted_tabs = output["tab"]
            if self.predict_notes:
                batch_predicted_notes = output["notes"]
                
            if getattr(output, "features", None) is not None:
                batch_audio = batch_x
                batch_x = output["features"]
            else:
                batch_audio = None

            if log_to_tensorboard:
                if not self.evaluated_once and idx == 0:
                    if len(batch_x[ex_idx].shape) == 3:
                        self.valid_tensorboard.log_image(f"features/x", batch_x[ex_idx])
                    if batch_ffm.nelement() != 0:
                        self.valid_tensorboard.log_image(
                            f"features/ffm",
                            batch_ffm[ex_idx].unsqueeze(0).transpose(1, 2),
                        )
                    if batch_audio is not None:
                        self.valid_tensorboard.log_audio(
                            f"audio",
                            batch_audio[ex_idx],
                            sample_rate=self.cfg.audio.sr,
                        )
                    self.evaluated_once = True
                if idx == 0:
                    if getattr(output, "ffm_emb", None) is not None:
                        self.valid_tensorboard.log_image(
                            f"ffm_emb",
                            output["ffm_emb"][ex_idx, :, :].unsqueeze(0).transpose(1, 2),
                            self.current_update,
                        )
                    if getattr(output, "attention_map", None) is not None:
                        self.valid_tensorboard.log_image(
                            f"attention_map",
                            output["attention_map"][ex_idx, :, :]
                            .unsqueeze(0)
                            .transpose(1, 2)
                            * 255,
                            self.current_update,
                        )
                    if self.predict_notes:
                        self.valid_tensorboard.log_image(
                            f"notes/output",
                            batch_predicted_notes[ex_idx, :, :].unsqueeze(0).transpose(1, 2)
                            * 255,
                            self.current_update,
                        )
                        self.valid_tensorboard.log_image(
                            f"notes/sigmoid",
                            torch.nn.functional.sigmoid(batch_predicted_notes[ex_idx, :, :])
                            .unsqueeze(0)
                            .transpose(1, 2)
                            * 255,
                            self.current_update,
                        )  
                    
                    if output.get("z", None) is not None:
                        self.valid_tensorboard.log_image(
                            f"z",
                            output["z"][ex_idx, :, :].squeeze().unsqueeze(0),
                            self.current_update,
                        )
                        self._log_heatmap(
                            self.valid_tensorboard, "heatmaps/z", output["z"][ex_idx, :, :].squeeze(), self.current_update
                        )
                        
                    if output.get("masked_z", None) is not None:
                        self.valid_tensorboard.log_image(
                            f"masked_z",
                            output["masked_z"][ex_idx, :, :].squeeze().unsqueeze(0),
                            self.current_update,
                        )
                        self._log_heatmap(
                            self.valid_tensorboard, "heatmaps/masked_z", output["masked_z"][ex_idx, :, :].squeeze(), self.current_update
                        )
                        
                    if output.get("c", None) is not None:
                        self.valid_tensorboard.log_image(
                            f"c",
                            output["c"][ex_idx, :, :].squeeze().unsqueeze(0),
                            self.current_update,
                        )
                        self._log_heatmap(
                            self.valid_tensorboard, "heatmaps/c", output["c"][ex_idx, :, :].squeeze(), self.current_update
                        )

                    for s in range(self.num_strings):
                        if self.apply_softmax:
                            if self.predict_onsets_and_frets:
                                self.valid_tensorboard.log_image(
                                    f"softmax/onsets/string-{s+1}",
                                    torch.nn.functional.softmax(
                                        batch_predicted_onsets[ex_idx, s, :, :], dim=-1
                                    )
                                    .unsqueeze(0)
                                    .transpose(1, 2)
                                    * 255,
                                    self.current_update,
                                )
                                self.valid_tensorboard.log_image(
                                    f"softmax/frets/string-{s+1}",
                                    torch.nn.functional.softmax(
                                        batch_predicted_frets[ex_idx, s, :, :], dim=-1
                                    )
                                    .unsqueeze(0)
                                    .transpose(1, 2)
                                    * 255,
                                    self.current_update,
                                )
                            elif self.predict_tab:
                                self.valid_tensorboard.log_image(
                                    f"softmax/string-{s+1}",
                                    torch.nn.functional.softmax(
                                        batch_predicted_tabs[ex_idx, s, :, :], dim=-1
                                    )
                                    .unsqueeze(0)
                                    .transpose(1, 2)
                                    * 255,
                                    self.current_update,
                                )
                        else:
                            if self.predict_onsets_and_frets:
                                self.valid_tensorboard.log_image(
                                    f"output/onsets/string-{s+1}",
                                    batch_predicted_onsets[ex_idx, s, :, :]
                                    .unsqueeze(0)
                                    .transpose(1, 2),
                                    self.current_update,
                                )
                                self.valid_tensorboard.log_image(
                                f"output/frets/string-{s+1}",
                                batch_predicted_frets[ex_idx, s, :, :]
                                .unsqueeze(0)
                                .transpose(1, 2),
                                self.current_update,
                            )
                            elif self.predict_tab:   
                                self.valid_tensorboard.log_image(
                                    f"output/string-{s+1}",
                                    batch_predicted_tabs[ex_idx, s, :, :]
                                    .unsqueeze(0)
                                    .transpose(1, 2),
                                    self.current_update,
                                )
                        if self.predict_onsets_and_frets:
                            self.valid_tensorboard.log_image(
                                f"argmax/onsets/string-{s+1}",
                                argmax_map(
                                    batch_predicted_onsets[ex_idx, s, :, :].detach().cpu()
                                )
                                .unsqueeze(0)
                                .transpose(1, 2),
                                self.current_update,
                            )
                            self.valid_tensorboard.log_image(
                                f"argmax/frets/string-{s+1}",
                                argmax_map(
                                    batch_predicted_frets[ex_idx, s, :, :].detach().cpu()
                                )
                                .unsqueeze(0)
                                .transpose(1, 2),
                                self.current_update,
                            )
                        elif self.predict_tab:
                            self.valid_tensorboard.log_image(
                                f"argmax/string-{s+1}",
                                argmax_map(
                                    batch_predicted_tabs[ex_idx, s, :, :].detach().cpu()
                                )
                                .unsqueeze(0)
                                .transpose(1, 2),
                                self.current_update,
                            )

            step_loss = 0
            if self.predict_onsets_and_frets:
                step_onset_losses, step_onsets_string_losses = self.tab_by_string_criterion(
                    batch_predicted_onsets, batch_onsets
                )
                step_loss += step_onset_losses * self.criterion_weights[0]
                step_frets_losses, step_frets_string_losses = self.tab_by_string_criterion(
                    batch_predicted_frets, batch_frets
                )
                step_string_losses = [
                    step_onsets_string_losses[s] + step_frets_string_losses[s]
                    for s in range(self.num_strings)
                ]
                step_loss += step_frets_losses * self.criterion_weights[0]
            elif self.predict_tab:
                step_tab_losses, step_string_losses = self.tab_by_string_criterion(
                    batch_predicted_tabs, batch_tab
                )
                step_loss += step_tab_losses * self.criterion_weights[0]
            step_note_loss = None
            if self.predict_notes:
                step_note_loss = self.note_criterion(
                    batch_predicted_notes, batch_notes
                )
                step_loss += step_note_loss * self.criterion_weights[1]
            
            if self.predict_onsets_and_frets:
                batch_predicted_frets = batch_predicted_frets.detach().cpu()
                batch_predicted_onsets = batch_predicted_onsets.detach().cpu()
                batch_frets = batch_frets.detach().cpu()
                batch_onsets = batch_onsets.detach().cpu()
            elif self.predict_tab:
                batch_predicted_tabs = batch_predicted_tabs.detach().cpu()
                batch_tab = batch_tab.detach().cpu()
            if self.predict_notes:
                batch_predicted_notes = batch_predicted_notes.detach().cpu()
                batch_notes = batch_notes.detach().cpu()

            loss_epoch.append(step_loss.detach().cpu().item())
            
            if self.predict_onsets_and_frets:
                loss_onsets_epoch.append(step_onset_losses.detach().cpu().item())
                loss_frets_epoch.append(step_frets_losses.detach().cpu().item())
            elif self.predict_tab:
                loss_tabs_epoch.append(step_tab_losses.detach().cpu().item())
            if self.predict_notes:
                loss_notes_epoch.append(step_note_loss.detach().cpu().item())
            for s in range(self.num_strings):
                strings_losses_epoch[s].append(
                    step_string_losses[s].detach().cpu().item()
                )
                
            self.valid_tensorboard.log_step("loss", float(step_loss.item()), self.current_update)
            if self.predict_onsets_and_frets:
                if log_to_tensorboard:
                    self.valid_tensorboard.log_step(
                        "loss/onsets", float(step_onset_losses.item()), self.current_update
                    )
                    self.valid_tensorboard.log_step(
                        "loss/frets", float(step_frets_losses.item()), self.current_update
                    )
                loss_onsets_avg = sum(loss_onsets_epoch[-10:]) / (
                    10 if len(loss_onsets_epoch) > 10 else len(loss_onsets_epoch)
                )
                loss_frets_avg = sum(loss_frets_epoch[-10:]) / (
                    10 if len(loss_frets_epoch) > 10 else len(loss_frets_epoch)
                )
            elif self.predict_tab:
                if log_to_tensorboard:
                    self.valid_tensorboard.log_step(
                        "loss/tabs", float(step_tab_losses.item()), self.current_update
                    )
                loss_tabs_avg = sum(loss_tabs_epoch[-10:]) / (
                    10 if len(loss_tabs_epoch) > 10 else len(loss_tabs_epoch)
                )
            if self.predict_notes:
                if log_to_tensorboard:
                    self.valid_tensorboard.log_step(
                        "loss/notes", float(step_note_loss.item()), self.current_update
                    )
                loss_notes_avg = sum(loss_notes_epoch[-10:]) / (
                    10 if len(loss_notes_epoch) > 10 else len(loss_notes_epoch)
                )
                
            loss_avg = sum(loss_epoch[-10:]) / (
                10 if len(loss_epoch) > 10 else len(loss_epoch)
            )
                
            if self.predict_onsets_and_frets:
                batch_predicted_onsets = torch.nn.functional.softmax(
                    batch_predicted_onsets, dim=-1
                )
                batch_onsets_argmax = batch_onsets.argmax(dim=-1)
                batch_predicted_onsets_argmax = torch.argmax(
                    batch_predicted_onsets, dim=-1
                ).cpu()
                batch_predicted_frets = torch.nn.functional.softmax(
                    batch_predicted_frets, dim=-1
                )
                batch_frets_argmax = batch_frets.argmax(dim=-1)
                batch_predicted_frets_argmax = torch.argmax(
                    batch_predicted_frets, dim=-1
                ).cpu()
            elif self.predict_tab:
                batch_predicted_tabs = torch.nn.functional.softmax(
                    batch_predicted_tabs, dim=-1
                )
                batch_tabs_argmax = batch_tab.argmax(dim=-1)
                batch_predicted_tabs_argmax = torch.argmax(
                    batch_predicted_tabs, dim=-1
                ).cpu()
            if self.predict_notes:
                batch_predicted_notes = torch.nn.functional.softmax(
                    batch_predicted_notes, dim=-1
                )
                batch_notes_argmax = batch_notes.argmax(dim=-1)
                batch_predicted_notes_argmax = torch.argmax(
                    batch_predicted_notes, dim=-1
                ).cpu()

            if not self.class_probabilities:
                if self.predict_onsets_and_frets:
                    batch_frets = batch_frets_argmax
                    batch_onsets = batch_onsets_argmax
                    batch_predicted_frets = batch_predicted_frets_argmax
                    batch_predicted_onsets = batch_predicted_onsets_argmax
                elif self.predict_tab:
                    batch_tab = batch_tabs_argmax
                    batch_predicted_tabs = batch_predicted_tabs_argmax
                if self.predict_notes:
                    batch_notes = batch_notes_argmax
                    batch_predicted_notes = batch_predicted_notes_argmax
            if idx % self.log_interval_steps == 0:

                if self.predict_onsets_and_frets:
                    step_onsets_f, step_onsets_p, step_onsets_r = f_measure(
                        batch_predicted_onsets_argmax.numpy(),
                        batch_onsets_argmax.numpy(),
                        num_strings=self.num_strings,
                        offset_ratio=self.predict_tab,
                        ignore_empty=True
                    )
                    self.valid_tensorboard.log_step(
                        "F/onsets", float(step_onsets_f), self.current_update
                    )
                    self.valid_tensorboard.log_step(
                        "P/onsets", float(step_onsets_p), self.current_update
                    )
                    self.valid_tensorboard.log_step(
                        "R/onsets", float(step_onsets_r), self.current_update
                    )
                    f_onsets_epoch.append(step_onsets_f)
                    p_onsets_epoch.append(step_onsets_p)
                    r_onsets_epoch.append(step_onsets_r)
                    step_frets_f, step_frets_p, step_frets_r = f_measure(
                        batch_predicted_frets_argmax.numpy(),
                        batch_frets_argmax.numpy(),
                        num_strings=self.num_strings,
                        offset_ratio=self.predict_tab,
                        ignore_empty=True
                    )
                    self.valid_tensorboard.log_step(
                        "F/frets", float(step_frets_f), self.current_update
                    )
                    self.valid_tensorboard.log_step(
                        "P/frets", float(step_frets_p), self.current_update
                    )
                    self.valid_tensorboard.log_step(
                        "R/frets", float(step_frets_r), self.current_update
                    )
                    f_frets_epoch.append(step_frets_f)
                    p_frets_epoch.append(step_frets_p)
                    r_frets_epoch.append(step_frets_r)
                    
                    f_onsets_total_avg = sum(f_onsets_epoch) / len(f_onsets_epoch)
                    p_onsets_total_avg = sum(p_onsets_epoch) / len(p_onsets_epoch)
                    r_onsets_total_avg = sum(r_onsets_epoch) / len(r_onsets_epoch)
                    f_frets_total_avg = sum(f_frets_epoch) / len(f_frets_epoch)
                    p_frets_total_avg = sum(p_frets_epoch) / len(p_frets_epoch)
                    r_frets_total_avg = sum(r_frets_epoch) / len(r_frets_epoch)
                    
                elif self.predict_tab:
                    step_tabs_f, step_tabs_p, step_tabs_r = f_measure(
                        batch_predicted_tabs_argmax.numpy(),
                        batch_tabs_argmax.numpy(),
                        num_strings=self.num_strings,
                        offset_ratio=self.predict_tab,
                        ignore_empty=True
                    )
                    self.valid_tensorboard.log_step(
                        "F/tabs", float(step_tabs_f), self.current_update
                    )
                    self.valid_tensorboard.log_step(
                        "P/tabs", float(step_tabs_p), self.current_update
                    )
                    self.valid_tensorboard.log_step(
                        "R/tabs", float(step_tabs_r), self.current_update
                    )
                    f_tabs_epoch.append(step_tabs_f)
                    p_tabs_epoch.append(step_tabs_p)
                    r_tabs_epoch.append(step_tabs_r)
                
                    f_tabs_total_avg = sum(f_tabs_epoch) / len(f_tabs_epoch)
                    p_tabs_total_avg = sum(p_tabs_epoch) / len(p_tabs_epoch)
                    r_tabs_total_avg = sum(r_tabs_epoch) / len(r_tabs_epoch)
                    
                if self.predict_notes:
                    step_notes_mse = torch.nn.functional.mse_loss(
                        batch_predicted_notes, batch_notes.float()
                    )
                    self.valid_tensorboard.log_step(
                        "mse/notes", float(step_notes_mse), self.current_update
                    )
                    mse_notes_epoch.append(step_notes_mse)

                    mse_notes_total_avg = sum(mse_notes_epoch) / len(mse_notes_epoch)
                    
                    log = [
                        f"lr: {float(self.optimizer.param_groups[0]['lr']):.2e}",
                        f"updates: {self.current_update}",
                        f"loss: {loss_avg:.3f}↓",
                        # f"loss_tabs: {loss_tabs_avg:.3f}",
                        # f"tabs_f: {f_tabs_total_avg:.3f}"
                    ]
                    if self.predict_onsets_and_frets:
                        log.append(f"loss_onsets: {loss_onsets_avg:.3f}↓")
                        log.append(f"onsets_f: {f_onsets_total_avg:.3f}→1")
                        log.append(f"loss_frets: {loss_frets_avg:.3f}↓")
                        log.append(f"frets_f: {f_frets_total_avg:.3f}→1")
                    elif self.predict_tab:
                        log.append(f"loss_tabs: {loss_tabs_avg:.3f}")
                        log.append(f"tabs_f: {f_tabs_total_avg:.3f}→1")
                    if self.predict_notes:
                        log.append(f"loss_notes: {loss_notes_avg:.3f}↓")
                        log.append(f"notes_mse: {mse_notes_total_avg:.3f}↓")
                    if self.use_tqdm:
                        evaluation_step_it.set_postfix_str(", ".join(log))
                
                log_interval_steps = {
                    "epoch": f"{self.current_epoch}",
                    "updates": f"{self.current_update}",
                    "lr": f"{float(self.optimizer.param_groups[0]['lr']):.2e}",
                    "avg_loss": f"{loss_avg:.3f}",
                }
                if self.predict_onsets_and_frets:
                    log_interval_steps.update(
                        {
                            "avg_loss_onsets": f"{loss_onsets_avg:.3f}",
                            "avg_onsets_f": f"{f_onsets_total_avg:.3f}",
                            "avg_onsets_p": f"{p_onsets_total_avg:.3f}",
                            "avg_onsets_r": f"{r_onsets_total_avg:.3f}",
                            "avg_loss_frets": f"{loss_frets_avg:.3f}",
                            "avg_frets_f": f"{f_frets_total_avg:.3f}",
                            "avg_frets_p": f"{p_frets_total_avg:.3f}",
                            "avg_frets_r": f"{r_frets_total_avg:.3f}",
                        }
                    )
                elif self.predict_tab:
                    log_interval_steps.update(
                        {
                            "avg_loss_tabs": f"{loss_tabs_avg:.3f}",
                            "avg_tabs_f": f"{f_tabs_total_avg:.3f}"
                        }
                    )
                if self.predict_notes:
                    log_interval_steps.update(
                        {
                            "avg_loss_notes": f"{loss_notes_avg:.3f}",
                            "avg_notes_mse": f"{mse_notes_total_avg:.3f}"
                        }
                    )
                logger.info(log_interval_steps)

            if self.predict_onsets_and_frets:
                batch_tabs_argmax = batch_onsets_argmax
                batch_predicted_tabs_argmax = batch_predicted_onsets_argmax
            ex_idx = 0
            while (
                batch_tabs_argmax[ex_idx].count_nonzero() == 0
                and ex_idx < len(batch_tabs_argmax) - 1
            ):
                ex_idx += 1

            max_sequence_length = max(
                len(batch_tabs_argmax[ex_idx, s]) for s in range(self.num_strings)
            )
            print(f"{'='*(os.get_terminal_size().columns)}\nExamples")
            print("Target:")
            for s in range(self.num_strings):
                chars_to_print = min(
                    max_sequence_length, os.get_terminal_size().columns - 25
                )
                print(f"{s+1}", end="|")
                for i in range(chars_to_print):
                    if chars_to_print <= 0:
                        break
                    if int(batch_tabs_argmax[ex_idx, s, i]) != 0:
                        t = int(batch_tabs_argmax[ex_idx, s, i]) - 1
                        if t < 0:
                            print("?", end="-")
                            chars_to_print -= 2
                        else:
                            if t < 10:
                                print(t, end="-")
                                chars_to_print -= 2
                            else:
                                print(f"{t:2d}", end="-")
                                chars_to_print -= 3
                    else:
                        print("-", end="")
                        chars_to_print -= 1
                while chars_to_print > 0:
                    print("-", end="")
                    chars_to_print -= 1
                print(f"|")
            print("Predicted:")  # BUG
            for s in range(self.num_strings):
                chars_to_print = min(
                    max_sequence_length, os.get_terminal_size().columns - 25
                )
                print(f"{s+1}", end="|")
                for i in range(chars_to_print):
                    if chars_to_print <= 0:
                        break
                    if int(batch_predicted_tabs_argmax[ex_idx, s, i]) != 0:
                        tabs_pred = int(batch_predicted_tabs_argmax[ex_idx, s, i]) - 1
                        if tabs_pred != 0:
                            if tabs_pred < 10:
                                print(tabs_pred, end="-")
                                chars_to_print -= 2
                            else:
                                print(f"{tabs_pred:2d}", end="-")
                                chars_to_print -= 3
                        else:
                            print("-", end="")
                            chars_to_print -= 1
                    else:
                        print("-", end="")
                        chars_to_print -= 1
                while chars_to_print > 0:
                    print("-", end="")
                    chars_to_print -= 1
                print(f"|")
            print(f"\n{'='*(os.get_terminal_size().columns)}\n")

            torch.cuda.empty_cache()

        loss_epoch = sum(loss_epoch) / len(loss_epoch)
        if self.predict_onsets_and_frets:
            loss_onsets_epoch = sum(loss_onsets_epoch) / len(loss_onsets_epoch)
            loss_frets_epoch = sum(loss_frets_epoch) / len(loss_frets_epoch)
            f_onsets_epoch = sum(f_onsets_epoch) / len(f_onsets_epoch)
            f_frets_epoch = sum(f_frets_epoch) / len(f_frets_epoch)
            p_onsets_epoch = sum(p_onsets_epoch) / len(p_onsets_epoch)
            p_frets_epoch = sum(p_frets_epoch) / len(p_frets_epoch)
            r_onsets_epoch = sum(r_onsets_epoch) / len(r_onsets_epoch)
            r_frets_epoch = sum(r_frets_epoch) / len(r_frets_epoch)
        elif self.predict_tab:
            loss_tabs_epoch = sum(loss_tabs_epoch) / len(loss_tabs_epoch)
            f_tabs_epoch = sum(f_tabs_epoch) / len(f_tabs_epoch)
            p_tabs_epoch = sum(p_tabs_epoch) / len(p_tabs_epoch)
            r_tabs_epoch = sum(r_tabs_epoch) / len(r_tabs_epoch)
        if self.predict_notes:
            loss_notes_epoch = sum(loss_notes_epoch) / len(loss_notes_epoch)
            mse_notes_epoch = sum(mse_notes_epoch) / len(mse_notes_epoch)
        else:
            mse_notes_epoch = loss_notes_epoch = None

        if self.lr_scheduler_epoch == "reduce_lr_on_plateau":
            if self.lr_scheduler_epoch_metric == "loss":
                self.lr_scheduler_epoch.step(loss_epoch)
            elif self.lr_scheduler_epoch_metric == "acc":
                self.lr_scheduler_epoch.step(f_tabs_epoch)
            logger.info(
                f"Learning rate updated to {float(self.optimizer.param_groups[0]['lr']):.2e}"
            )
            if log_to_tensorboard:
                self.valid_tensorboard.log_epoch(
                    "lr",
                    float(self.optimizer.param_groups[0]["lr"]),
                    self.current_epoch,
                )

        if self.best_loss is None:
            self.best_loss = loss_epoch
        if loss_epoch < self.best_loss:
            self.best_loss = loss_epoch
            logger.info(f"Best loss reached: {self.best_loss}")
            self.save_checkpoint(is_best=True)

        log_valid = {
            "epoch": f"{self.current_epoch}",
            "valid_loss": f"{loss_epoch:.3f}",
        }
        if self.predict_onsets_and_frets:
            log_valid.update(
                {
                    "valid_loss_onsets": f"{loss_onsets_epoch:.3f}",
                    "valid_onsets_total_f": f"{f_onsets_epoch:.3f}",
                    "valid_onsets_total_p": f"{p_onsets_epoch:.3f}",
                    "valid_onsets_total_r": f"{r_onsets_epoch:.3f}",
                    "valid_loss_frets": f"{loss_frets_epoch:.3f}",
                    "valid_frets_total_f": f"{f_frets_epoch:.3f}",
                    "valid_frets_total_p": f"{p_frets_epoch:.3f}",
                    "valid_frets_total_r": f"{r_frets_epoch:.3f}",
                }
            )
        elif self.predict_tab:
            log_valid.update(
                {
                    "valid_loss_tabs": f"{loss_tabs_epoch:.3f}",
                    "valid_tabs_total_f": f"{f_tabs_epoch:.3f}",
                    "valid_tabs_total_p": f"{p_tabs_epoch:.3f}",
                    "valid_tabs_total_r": f"{r_tabs_epoch:.3f}",
                }
            )
        if self.predict_notes:
            log_valid.update(
                {
                    "valid_loss_notes": f"{loss_notes_epoch:.3f}",
                    "valid_notes_total_mse": f"{mse_notes_epoch:.3f}",
                }
            )
        logger.info(log_valid)

        if log_to_tensorboard:
            for s in range(self.num_strings):
                self.valid_tensorboard.log_epoch(
                    f"string-{s+1}/loss",
                    float(sum(strings_losses_epoch[s]) / len(strings_losses_epoch[s])),
                    self.current_epoch,
                )
            self.valid_tensorboard.log_epoch("loss", loss_epoch, self.current_epoch)
            if self.predict_onsets_and_frets:
                self.valid_tensorboard.log_epoch(
                    "loss/onsets", loss_onsets_epoch, self.current_epoch
                )
                self.valid_tensorboard.log_epoch(
                    "loss/frets", loss_frets_epoch, self.current_epoch
                )
                self.valid_tensorboard.log_epoch("F/onsets", f_onsets_epoch, self.current_epoch)
                self.valid_tensorboard.log_epoch("P/onsets", p_onsets_epoch, self.current_epoch)
                self.valid_tensorboard.log_epoch("R/onsets", r_onsets_epoch, self.current_epoch)
                self.valid_tensorboard.log_epoch("F/frets", f_frets_epoch, self.current_epoch)
                self.valid_tensorboard.log_epoch("P/frets", p_frets_epoch, self.current_epoch)
                self.valid_tensorboard.log_epoch("R/frets", r_frets_epoch, self.current_epoch)
            elif self.predict_tab:
                self.valid_tensorboard.log_epoch(
                    "loss/tabs", loss_tabs_epoch, self.current_epoch
                )
                self.valid_tensorboard.log_epoch("F/tabs", f_tabs_epoch, self.current_epoch)
                self.valid_tensorboard.log_epoch("P/tabs", p_tabs_epoch, self.current_epoch)
                self.valid_tensorboard.log_epoch("R/tabs", r_tabs_epoch, self.current_epoch)

            if self.predict_notes:
                self.valid_tensorboard.log_epoch(
                    "loss/notes", loss_notes_epoch, self.current_epoch
                )
                self.valid_tensorboard.log_epoch("mse/notes", mse_notes_epoch, self.current_epoch)
                
        if self.predict_onsets_and_frets:
            return (
                loss_epoch,
                loss_onsets_epoch,
                loss_frets_epoch,
                f_onsets_epoch,
                f_frets_epoch,
                p_onsets_epoch,
                p_frets_epoch,
                r_onsets_epoch,
                r_frets_epoch,
                loss_notes_epoch,
                mse_notes_epoch,
                predicted_tabs,
                target_tabs,
            )
        elif self.predict_tab:
            return (
                loss_epoch,
                loss_tabs_epoch,
                f_tabs_epoch,
                p_tabs_epoch,
                r_tabs_epoch,
                loss_notes_epoch,
                mse_notes_epoch,
                predicted_tabs,
                target_tabs,
            )

    def train(self):
        logger.info(f"Start training")
        self.model.train()

        accumulation_steps = self.accumulation_steps
        assert accumulation_steps > 0, "accumulation_steps must be greater than 0"

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
                if self.predict_onsets_and_frets:
                    (
                        valid_loss,
                        valid_loss_onsets,
                        valid_loss_frets,
                        valid_f_onsets,
                        valid_f_frets,
                        valid_p_onsets,
                        valid_p_frets,
                        valid_r_onsets,
                        valid_r_frets,
                        valid_loss_notes,
                        valid_mse_notes,
                        _,
                        _,
                    ) = self.evaluate()
                    epoch_info = {
                        "valid_loss": f"{valid_loss:.3f}",
                        "valid_loss_onsets": f"{valid_loss_onsets:.3f}",
                        "valid_loss_frets": f"{valid_loss_frets:.3f}",
                        "valid_onsets_f": f"{valid_f_onsets:.3f}",
                        "valid_onsets_p": f"{valid_p_onsets:.3f}",
                        "valid_onsets_r": f"{valid_r_onsets:.3f}",
                        "valid_frets_f": f"{valid_f_frets:.3f}",
                        "valid_frets_p": f"{valid_p_frets:.3f}",
                        "valid_frets_r": f"{valid_r_frets:.3f}",
                    }
                elif self.predict_tab:
                    (
                        valid_loss,
                        valid_loss_tabs,
                        valid_f_tabs,
                        valid_p_tabs,
                        valid_r_tabs,
                        valid_loss_notes,
                        valid_mse_notes,
                        _,
                        _,
                    ) = self.evaluate()
                    epoch_info = {
                        "valid_loss": f"{valid_loss:.3f}",
                        "valid_loss_tabs": f"{valid_loss_tabs:.3f}",
                        "valid_tabs_f": f"{valid_f_tabs:.3f}",
                        "valid_tabs_p": f"{valid_p_tabs:.3f}",
                        "valid_tabs_r": f"{valid_r_tabs:.3f}",
                    }
                if self.predict_notes:
                    epoch_info.update(
                        {
                            "valid_loss_notes": f"{valid_loss_notes:.3f}",
                            "valid_notes_mse": f"{valid_mse_notes:.3f}",
                        }
                    )
                logger.info(f"Training done: {epoch_info}")
                break

            logger.info(f"Training epoch {self.current_epoch}")
            if self.predict_onsets_and_frets:
                (
                    train_loss, # loss_epoch
                    train_loss_onsets, # loss_onsets_epoch
                    train_loss_frets, # loss_frets_epoch
                    train_f_onsets, # f_onsets_epoch
                    train_f_frets, # f_frets_epoch
                    train_p_onsets, # p_onsets_epoch
                    train_p_frets, # p_frets_epoch
                    train_r_onsets, # r_onsets_epoch
                    train_r_frets, # r_frets_epoch
                    train_loss_notes, # loss_notes_epoch
                    train_mse_notes, # mse_notes_epoch
                ) = self.train_one_epoch()
                epoch_info = {
                    "epoch": f"{self.current_epoch}",
                    "updates": f"{self.current_update}",
                    "train_loss": f"{train_loss:.3f}",
                    "train_loss_onsets": f"{train_loss_onsets:.3f}",
                    "train_loss_frets": f"{train_loss_frets:.3f}",
                    "train_f_onsets": f"{train_f_onsets:.3f}",
                    "train_p_onsets": f"{train_p_onsets:.3f}",
                    "train_r_onsets": f"{train_r_onsets:.3f}",
                    "train_f_frets": f"{train_f_frets:.3f}",
                    "train_p_frets": f"{train_p_frets:.3f}",
                    "train_r_frets": f"{train_r_frets:.3f}",
                }
            elif self.predict_tab:
                (
                    train_loss,
                    train_loss_tabs,
                    train_loss_notes,
                    train_f_tabs,
                    train_p_tabs,
                    train_r_tabs,
                    train_loss_notes,
                    train_mse_notes,
                ) = self.train_one_epoch()
                epoch_info = {
                    "epoch": f"{self.current_epoch}",
                    "updates": f"{self.current_update}",
                    "train_loss": f"{train_loss:.3f}",
                    "train_loss_tabs": f"{train_loss_tabs:.3f}",
                    "train_f_tabs": f"{train_f_tabs:.3f}",
                    "train_p_tabs": f"{train_p_tabs:.3f}",
                    "train_r_tabs": f"{train_r_tabs:.3f}",
                }
                if self.predict_notes:
                    epoch_info.update(
                        {
                            "train_loss_notes": f"{train_loss_notes:.3f}",
                            "train_mse_notes": f"{train_mse_notes:.3f}",
                        }
                    )

            if (
                self.validate_interval_epochs is not None
                and self.current_epoch % self.validate_interval_epochs == 0
            ):
                if self.predict_onsets_and_frets:
                    (
                        valid_loss,
                        valid_loss_onsets,
                        valid_loss_frets,
                        valid_f_onsets,
                        valid_f_frets,
                        valid_p_onsets,
                        valid_p_frets,
                        valid_r_onsets,
                        valid_r_frets,
                        valid_loss_notes,
                        valid_mse_notes,
                        _,
                        _,
                    ) = self.evaluate()
                    epoch_info.update(
                        {
                            "valid_loss": f"{valid_loss:.3f}",
                            "valid_loss_onsets": f"{valid_loss_onsets:.3f}",
                            "valid_loss_frets": f"{valid_loss_frets:.3f}",
                            "valid_onsets_f": f"{valid_f_onsets:.3f}",
                            "valid_onsets_p": f"{valid_p_onsets:.3f}",
                            "valid_onsets_r": f"{valid_r_onsets:.3f}",
                            "valid_frets_f": f"{valid_f_frets:.3f}",
                            "valid_frets_p": f"{valid_p_frets:.3f}",
                            "valid_frets_r": f"{valid_r_frets:.3f}",
                        }
                    )
                elif self.predict_tab:
                    (
                        valid_loss,
                        valid_loss_tabs,
                        valid_f_tabs,
                        valid_p_tabs,
                        valid_r_tabs,
                        valid_loss_notes,
                        valid_mse_notes,
                        _,
                        _,
                    ) = self.evaluate()
                    epoch_info.update(
                        {
                            "valid_loss": f"{valid_loss:.3f}",
                            "valid_loss_tabs": f"{valid_loss_tabs:.3f}",
                            "valid_tabs_f": f"{valid_f_tabs:.3f}",
                            "valid_tabs_p": f"{valid_p_tabs:.3f}",
                            "valid_tabs_r": f"{valid_r_tabs:.3f}",
                        }
                    )
                    if self.predict_notes:
                        epoch_info.update(
                            {
                                "valid_loss_notes": f"{valid_loss_notes:.3f}",
                                "valid_notes_mse": f"{valid_mse_notes:.3f}",
                            }
                        )
                logger.info(f"Epoch {self.current_epoch} done: {epoch_info}")
                
            logger.info(f"Epoch {self.current_epoch} done: {epoch_info}")

        print("=" * (os.get_terminal_size().columns))

        logger.info("Training done.")

        if self.run_test_at_end:
            logger.info("Running evaluation on test set")
            if self.predict_onsets_and_frets:
                (
                    test_loss,
                    test_loss_onsets,
                    test_loss_frets,
                    test_f_onsets,
                    test_f_frets,
                    test_p_onsets,
                    test_p_frets,
                    test_r_onsets,
                    test_r_frets,
                    test_mse_notes,
                    _,
                    _,
                ) = self.evaluate(data_loader=self.test_data_loader)
                eval_info = {
                    "test_loss": f"{test_loss:.3f}",
                    "test_loss_onsets": f"{test_loss_onsets:.3f}",
                    "test_loss_frets": f"{test_loss_frets:.3f}",
                    "test_onsets_f": f"{test_f_onsets:.3f}",
                    "test_onsets_p": f"{test_p_onsets:.3f}",
                    "test_onsets_r": f"{test_r_onsets:.3f}",
                    "test_frets_f": f"{test_f_frets:.3f}",
                    "test_frets_p": f"{test_p_frets:.3f}",
                    "test_frets_r": f"{test_r_frets:.3f}",
                }
            elif self.predict_tab:
                (
                    test_loss,
                    test_loss_tabs,
                    test_loss_notes,
                    test_f_tabs,
                    test_p_tabs,
                    test_r_tabs,
                    test_mse_notes,
                    _,
                    _,
                ) = self.evaluate(data_loader=self.test_data_loader)
                eval_info = {
                    "test_loss": f"{test_loss:.3f}",
                    "test_loss_tabs": f"{test_loss_tabs:.3f}",
                    "test_tabs_f": f"{test_f_tabs:.3f}",
                    "test_tabs_p": f"{test_p_tabs:.3f}",
                    "test_tabs_r": f"{test_r_tabs:.3f}",
                }
                if self.predict_notes:
                    eval_info.update(
                        {
                            "test_loss_notes": f"{test_loss_notes:.3f}",
                            "test_notes_mse": f"{test_mse_notes:.3f}",
                        }
                    )
            logger.info(f"Evaluation on test set: {eval_info}")
            
