import pickle as pkl
from os.path import join

# Training
import torch
from omegaconf import DictConfig

# Kitten
from kitten.common.util import *
from cats.cats import CatsExperiment
from cats.evaluation import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg: DictConfig):
    experiment = CatsExperiment(
        cfg=cfg,
        device=DEVICE,
    )
    experiment.run()
    return experiment


@hydra.main(version_base=None, config_path="./config", config_name="defaults")
def main(cfg: DictConfig):
    path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg.log.path = path
    experiment = run(cfg)
    pkl.dump(experiment, open(join(experiment.logger.path, "experiment.pkl"), "wb"))


if __name__ == "__main__":
    main()
