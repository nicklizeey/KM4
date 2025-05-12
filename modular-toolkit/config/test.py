import yaml
import hydra

#with open("./KM4/modular-toolkit/config/train.yaml", "r") as f:
#    config = yaml.safe_load(f)

#print(config["train"]["batch_size"])



@hydra.main(config_path="./", config_name="train")

def main(config):
    print(config.train.eval_every)

if __name__ == "__main__":
    main()