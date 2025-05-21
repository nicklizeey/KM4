import hydra
from omegaconf import DictConfig, OmegaConf
import datetime
from trainer import *




@hydra.main(config_path="./config", config_name="train_Transformer_D_O")
def main(cfg : DictConfig):
    cfg = OmegaConf.to_container(cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()