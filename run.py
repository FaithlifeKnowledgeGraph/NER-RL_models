import yaml
import random
import torch

from logos_data_loader import LogosDataLoader

def parse_yaml(f_path: str = 'config.yaml'):
    with open(f_path, 'r') as f:
        try:
            args = yaml.safe_load(f)
            return args
        except yaml.YAMLError as exc:
            print(exc)

def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parse_yaml()

    print("Training Model with args: ", args)

    set_global_seed(args['torch']['seed'])

    loader = LogosDataLoader(**args['loader'])
    loader.run()
    print(len(loader.data))
