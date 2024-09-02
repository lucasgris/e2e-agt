import os
import ast
import inspect
import random
import logging
from typing import Union

import shutil
import models
from utils.tensorboard import TensorboardWriter

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm
from torch_warmup_lr import WarmupLR

from core.audio_processor import AudioProcessor
from core.data import DataLoader

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, cfg: DictConfig, run_dir: Union[os.PathLike, str]):
        self.run_dir = run_dir
        self.cfg = cfg
        self.cfg_dict = OmegaConf.to_container(cfg)
        print(OmegaConf.to_yaml(cfg))
        with open(f"{run_dir}/comment.txt", "w") as cf:
            print(cfg.comment, file=cf)
        os.system(f"stickytape {inspect.getfile(getattr(models, cfg.model.arch))} > {run_dir}/model.py")

        logger.info("Setup training")
        self.raise_error = cfg.raise_error
        self.print_examples_validation = cfg.print_examples_validation
        self.current_epoch = 0
        self.current_update = 0
        self.valid_data_loader = None
        self.train_data_loader = None
        self.best_loss = None
        self.use_tqdm = cfg.tqdm
        self.log_interval_steps = cfg.log_interval_steps
        self.save_interval_updates = cfg.checkpoint.save_interval_updates
        self.keep_interval_updates = cfg.checkpoint.keep_interval_updates
        self.validate_interval_updates = cfg.validate_interval_updates
        self.validate_interval_epochs = cfg.validate_interval_epochs
        self.evaluated_once = False
        self.run_test_at_end = cfg.run_test_at_end
        self.num_strings = cfg.num_strings
        self.segment_audio_frames = cfg.segment_audio_frames
        self.model_extra_forward_args = cfg.model.extra_forward_args if "extra_forward_args" in cfg.model else {}

        logger.info(f"Logs will be stored at {self.run_dir}")

        self.checkpoint_save_dir = os.path.join(self.run_dir, cfg.checkpoint.save_dir)
        logger.info(f"Checkpoints will be stored at {self.checkpoint_save_dir}")
        os.makedirs(self.checkpoint_save_dir, exist_ok=True)
        self.last_saved_checkpoints = []
        self.save_weights = cfg.save_weights
        self.last_saved_weights = []

        tensorboard_dir = os.path.join(run_dir, cfg.tb_log_dir)
        logger.info(f"Tensorboard logs will be stored at {tensorboard_dir}")
        self.train_tensorboard = TensorboardWriter(tensorboard_dir, "train")
        self.valid_tensorboard = TensorboardWriter(tensorboard_dir, "valid")

        if cfg.optimization.max_epochs is not None:
            logger.info(f"Max epochs set as {cfg.optimization.max_epochs}")
            self.max_epochs = cfg.optimization.max_epochs
        else:
            raise ValueError("Max epochs must be set")
        if cfg.optimization.max_updates is not None:
            logger.info(f"Max update set as {cfg.optimization.max_updates}")
            self.max_updates = cfg.optimization.max_updates
        else:
            self.max_updates = None

        if cfg.random_seed is not None:
            # fix random seeds for reproducibility
            random.seed(cfg.random_seed)
            torch.manual_seed(cfg.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(cfg.random_seed)

        if cfg.device is not None:
            self.device = torch.device(cfg.device)
        else:
            logger.info("Checking if CUDA is available")
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            logger.info(f"Device: {self.device}")

        logger.info(f"Building model {cfg.model.arch}")
        try:
            self.model = getattr(models, cfg.model.arch)(cfg.model).to(self.device)
        except AttributeError as e:
            raise ValueError(f"Invalid model arch: {cfg.model.arch}. {str(e)}")
        logger.info(self.model)

        with open(os.path.join(run_dir, "model.txt"), "w") as f:
            print(self.model, file=f)
            try:
                from torchinfo import summary as summary

                s = summary(
                    self.model,
                    list(ast.literal_eval(cfg.model.input_size)),
                    batch_dim=0,
                    device=self.device,
                )
                # save to txt
                with open(os.path.join(run_dir, "model.txt"), "a") as f:
                    print(s, file=f)
            except Exception as e:
                try:
                    from torchsummary import summary

                    summary(self.model, list(ast.literal_eval(cfg.model.input_size)))
                    # save to txt
                    with open(os.path.join(run_dir, "model.txt"), "a") as f:
                        summary(self.model, list(ast.literal_eval(cfg.model.input_size)), file=f)
                except Exception as e:
                    logger.error(f"Model summary failed: {str(e)}")
                    # print stack trace
                    import traceback
                    traceback.print_exc()
                    input("Model summary failed. You might want to check the model first. Press any key to continue anyway.")
            try:
                self.train_tensorboard.log_network(
                    self.model,
                    torch.zeros(ast.literal_eval(cfg.model.input_size))
                    .unsqueeze(0)
                    .to(self.device),
                )
            except Exception as e:
                logger.error("Could not log network to tensorboard")
                logger.debug(str(e))

        self._model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters()
        )
        self.params = sum([np.prod(p.size()) for p in self._model_parameters])
        logger.info(f"Total treinable parameters: {self.params}")

        if cfg.optimizer.name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                weight_decay=cfg.optimizer.weight_decay,
                lr=cfg.optimization.learning_rate,
                eps=cfg.optimizer.adam_eps,
                betas=cfg.optimizer.adam_betas,
            )
        else:
            raise ValueError(f"Invalid optimizer: {cfg.optimizer.name}")

        logger.info(f"Setting up learning rate scheduler")
        self.lr_scheduler_warmup = self.lr_scheduler_updates = self.lr_scheduler_epoch = None
        if cfg.lr_scheduler.updates is not None:
            if cfg.lr_scheduler.updates.name == "ExponentialLR":
                self.lr_scheduler_updates = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=cfg.lr_scheduler.updates.gamma, verbose=False
                )
                self.lr_scheduler_updates_interval = cfg.lr_scheduler.updates.interval
            elif cfg.lr_scheduler.updates.name == "MultiStepLR":
                self.lr_scheduler_updates = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=cfg.lr_scheduler.updates.milestones,
                    gamma=cfg.lr_scheduler.updates.gamma,
                    verbose=False,
                )
                self.lr_scheduler_updates_interval = cfg.lr_scheduler.updates.interval
            else:
                raise ValueError(
                    f"Invalid lr_scheduler: {cfg.lr_scheduler.updates.name}"
                )
            if cfg.lr_scheduler.updates.warmup is not None:
                logger.info("Setting up warmup learning rate scheduler for updates")
                self.lr_scheduler_warmup = WarmupLR(
                    self.lr_scheduler_updates, **cfg.lr_scheduler.updates.warmup
                )   
                self.optimizer.zero_grad()
                self.optimizer.step()
                self.lr_scheduler_warmup.step()
        if cfg.lr_scheduler.epoch is not None:
            if cfg.lr_scheduler.updates is not None:
                logger.warning(
                    "Both update and epoch learning rate schedulers are set. Be careful with the configuration or it might lead to unexpected behavior."
                )
            if cfg.lr_scheduler.epoch.name == "StepLR":
                self.lr_scheduler_epoch = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=cfg.lr_scheduler.epoch.step_size,
                    verbose=False,
                    gamma=cfg.lr_scheduler.epoch.gamma,
                )
            elif cfg.lr_scheduler.epoch.name == "ReduceLROnPlateau":
                self.lr_scheduler_epoch = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    "min",
                    verbose=False,
                    factor=cfg.lr_scheduler.epoch.factor,
                    patience=cfg.lr_scheduler.epoch.patience,
                )
                self.lr_scheduler_epoch_metric = cfg.lr_scheduler.epoch.get("metric", "loss")
                if self.lr_scheduler_epoch_metric != "loss":
                    raise NotImplementedError(
                        "Only loss is supported as metric for ReduceLROnPlateau"
                    )
            else:
                raise ValueError(f"Invalid lr_scheduler: {cfg.lr_scheduler.epoch.name}")

        if cfg.audio:
            self.audio_processor = AudioProcessor(cfg.audio)
        else:
            self.audio_processor = None
                
        self.use_amp = cfg.use_amp
        if self.use_amp:
            logger.info("Using AMP (Automatic Mixed Precision)")
            self.scaler = GradScaler()

        self.continue_training_from = None
        if "checkpoint_last.pt" in os.listdir(os.path.join(run_dir, "checkpoints")):
            self.continue_training_from = os.path.join(
                run_dir, "checkpoints", "checkpoint_last.pt"
            )
        elif cfg.checkpoint.continue_training_from is not None:
            self.continue_training_from = cfg.checkpoint.continue_training_from
        if self.continue_training_from is not None:
            logger.info(f"Continuing training from {self.continue_training_from}")
            self.load_checkpoint(self.continue_training_from, continue_training=True)
        elif cfg.checkpoint.finetune_from_model is not None:
            logger.info(
                f"Finetuning from model {cfg.checkpoint.finetune_from_model}. "
                "Optimizers, number of iterations, and previous states will be reset."
            )
            checkpoint = torch.load(cfg.checkpoint.finetune_from_model)
            # self.model = torch.nn.DataParallel(self.model)
            try:
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)
                logger.info(
                    f"Succesfully loaded model from {cfg.checkpoint.finetune_from_model}"
                )
            except:
                new_state_dict = {}
                for k, v in self.model.state_dict().items():
                    try:
                        logger.info(f"Loading {k} from the checkpoint")
                        if k in checkpoint:
                            if v.size() == checkpoint[k].size():
                                new_state_dict[k] = checkpoint[k]
                            else:
                                logger.warning(
                                    f"Size mismatch for {k}: {v.size()} != {checkpoint[k].size()}"
                                )
                                new_state_dict[k] = v
                        else:
                            if v.size() == checkpoint["state_dict"][k].size():
                                new_state_dict[k] = checkpoint.state_dict()[k]
                            else:
                                logger.warning(
                                    f"Size mismatch for {k}: {v.size()} != {checkpoint.state_dict()[k].size()}"
                                )
                                new_state_dict[k] = v
                    except Exception as e:
                        logger.warning("%s is not in the checkpoint" % k)
                        new_state_dict[k] = v
                self.model.load_state_dict(new_state_dict)
    
        self.grad_norm_clip = cfg.optimization.grad_norm_clip
        if self.grad_norm_clip:
            logger.info(f"Gradient norm clipping set as {self.grad_norm_clip}")

        self.accumulation_steps = cfg.optimization.accumulation_steps
        if self.accumulation_steps > 1:
            self.accumulation_reduction = cfg.optimization.accumulation_reduction
            logger.info(f"Accumulation steps set as {self.accumulation_steps}")
            logger.info(f"Accumulation reduction set as {self.accumulation_reduction}")
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        logger.info(f"Freezing finetune updates set as {self.freeze_finetune_updates}")
        self.freeze_only_feature_extractor = cfg.get("freeze_only_feature_extractor", False)
        if self.freeze_only_feature_extractor:
            logger.info(f"Freezing only feature extractor")
            
    def get_model_state(self):
        return {
            "args": self.cfg_dict,
            "epoch": self.current_epoch,
            "lr_epoch": (
                self.lr_scheduler_epoch.state_dict()
                if self.lr_scheduler_epoch is not None
                else None
            ),
            "lr_updates": (
                self.lr_scheduler_updates.state_dict()
                if self.lr_scheduler_updates is not None
                else None
            ),
            "optimizer": self.optimizer.state_dict(),
            "state_dict": self.model.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "step": self.current_update,
            "best_loss": self.best_loss
        }

    def _save_weights(self, is_last=True, is_best=False, step=None):
        logger.info(
            f"Saving weights at {self.checkpoint_save_dir} "
            f"[is_last={is_last}, is_best={is_best}, step={step}]"
        )
        if is_last:
            last_fpath = os.path.join(self.checkpoint_save_dir, "checkpoint_last.weights.pt")
            torch.save(self.get_model_state()["state_dict"], last_fpath)
        if step is not None and self.keep_interval_updates > 0:
            last_fpath = os.path.join(
                self.checkpoint_save_dir, f"checkpoint_{step:010}.weights.pt"
            )
            torch.save(self.get_model_state()["state_dict"], last_fpath)
            self.last_saved_weights.append(last_fpath)
            if len(self.last_saved_weights) > self.keep_interval_updates:
                old_ckpt = self.last_saved_weights.pop(0)
                logger.info(
                    f"Removing old weights {old_ckpt} "
                    f"due to keep_interval_updates={self.keep_interval_updates}"
                )
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                else:
                    logger.warning(f"Unable to remove old weights: {old_ckpt}")
        if is_best:
            best_fpath = os.path.join(self.checkpoint_save_dir, "checkpoint_best.weights.pt")
            shutil.copyfile(last_fpath, best_fpath)

    def save_checkpoint(self, is_last=True, is_best=False, step=None):
        if self.save_weights:
            self._save_weights(is_last, is_best, step)
        logger.info(
            f"Saving checkpoint at {self.checkpoint_save_dir} "
            f"[is_last={is_last}, is_best={is_best}, step={step}]"
        )
        if is_last:
            last_fpath = os.path.join(self.checkpoint_save_dir, "checkpoint_last.pt")
            torch.save(self.get_model_state(), last_fpath)
        if step is not None and self.keep_interval_updates > 0:
            last_fpath = os.path.join(
                self.checkpoint_save_dir, f"checkpoint_{step:010}.pt"
            )
            torch.save(self.get_model_state(), last_fpath)
            self.last_saved_checkpoints.append(last_fpath)
            if len(self.last_saved_checkpoints) > self.keep_interval_updates:
                old_ckpt = self.last_saved_checkpoints.pop(0)
                logger.info(
                    f"Removing old checkpoint {old_ckpt} "
                    f"due to keep_interval_updates={self.keep_interval_updates}"
                )
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
                else:
                    logger.warning(f"Unable to remove old checkpoint: {old_ckpt}")

        if is_best:
            best_fpath = os.path.join(self.checkpoint_save_dir, "checkpoint_best.pt")
            shutil.copyfile(last_fpath, best_fpath)

    def load_checkpoint(self, checkpoint_path, continue_training=True):
        logger.info(f"Loading checkpoint at {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            if continue_training:
                if self.lr_scheduler_warmup is not None:
                    logger.warning("It is not possible to continue training from a checkpoint with warmup. Warmup learning rate scheduler will be ignored.")
                    self.lr_scheduler_warmup = None
                if "lr_epoch" in checkpoint and checkpoint["lr_epoch"] is not None:
                    logger.info("Loading epoch learning rate scheduler state")
                    self.lr_scheduler_epoch.load_state_dict(checkpoint["lr_epoch"])
                if "lr_updates" in checkpoint and checkpoint["lr_updates"] is not None:
                    logger.info("Loading update learning rate scheduler state")
                    self.lr_scheduler_updates.load_state_dict(checkpoint["lr_updates"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.current_epoch = checkpoint["epoch"]
                self.current_update = checkpoint["step"]
                self.best_loss = checkpoint.get("best_loss", None)
                if self.use_amp and "scaler" in checkpoint: 
                    self.scaler.load_state_dict(checkpoint["scaler"]) 
        except Exception as e:
            logger.error(f"Could not load checkpoint: {str(e)}, cannot continue training")
            raise e
            
    def check_valid_gradients(self):
        valid_gradients = True
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        
        return valid_gradients

    def train_one_epoch(self):
        raise NotImplementedError

    def evaluate(self, data_loader: DataLoader=None):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
