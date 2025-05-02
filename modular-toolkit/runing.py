import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
from trainer import *




@hydra.main(config_path="./config", config_name="train")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    trainer = Trainer(cfg)
    print(trainer.model_dir)
    trainer.train()


if __name__ == '__main__':
    main()