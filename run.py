import yaml
import random
import torch
import numpy as np

from logos_data_loader import LogosDataLoader
from relation_processor import RelationProcessor
from relationModels import RelationModels, SimpleRelationExtractionModel
from relationTrainer import RelationTrainer

def parse_yaml(f_path: str = 'config.yaml') -> dict:
    """Parse a YAML file containing training configuration 

    Args:
        f_path: Path to the YAML file. Defaults to 'config.yaml'.

    Returns:
        Parsed configurations

    Raises:
        yaml.YAMLError: If there is an error while parsing the YAML file.

    """

    with open(f_path, 'r') as f:
        try:
            args = yaml.safe_load(f)
            return args
        except yaml.YAMLError as exc:
            print(exc)

def set_global_seed(seed: int) -> None:
    """Set the global random seed for reproducibility in RNG.

    Args:
        seed: The seed value to set for random number generation.

    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parse_yaml()

    print("Training Model with args: ", args)

    set_global_seed(args['torch']['seed'])

    loader = LogosDataLoader(**args['loader'])
    data = loader.run()

    processor = RelationProcessor(data, **args['processor'])
    train_loader, val_loader, test_loader, y_test = processor.run()
    vocab_size = processor.get_vocab_size()

    relation_model = SimpleRelationExtractionModel(vocab_size, **args['nn_model'])
    model = RelationModels(relation_model, args)

    trainer = RelationTrainer(args, model, y_test)
    trainer.run(train_loader, val_loader, test_loader)

