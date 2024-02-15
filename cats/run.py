from omegaconf import DictConfig
import hydra

# Script for running mass experiments

@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def train(cfg: DictConfig) -> None:
    pass

if __name__ == "__main__":
    train()