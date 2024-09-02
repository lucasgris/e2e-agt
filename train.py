import hydra
import shutil
import torch
import traceback
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)

from core import CETrainer, CTCTrainer


@hydra.main(version_base=None, config_path="configs", config_name="cnn_ctc")
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().run.dir

    if cfg.trainer == "CTCTrainer":
        trainer = CTCTrainer(cfg, run_dir)
    elif cfg.trainer == "CETrainer":
        trainer = CETrainer(cfg, run_dir)
    else:
        raise ValueError(f"Invalid trainer: {cfg.trainer}")
    
    if cfg.dry_run:
        logger.debug("Dry run mode activated.")
        torch.autograd.set_detect_anomaly(True)
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
        except Exception as e:
            traceback.print_exc()
            logger.error(str(e))
        r = input("Keep run directory? (y/N): ")
        if r.lower() != "y":
            logger.info(f"Removing run directory {run_dir}")
            shutil.rmtree(run_dir, ignore_errors=True)
    else:
        trainer.train()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
