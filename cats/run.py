from omegaconf import DictConfig
import hydra

from .cats import CatsExperiment

# Script for running mass experiments
@hydra.main(version_base=None, config_path="../config/cats", config_name="defaults")
def train(cfg: DictConfig) -> None:
    experiment = CatsExperiment(cfg, **cfg.cats)
    experiment.run()

if __name__ == "__main__":
    train()