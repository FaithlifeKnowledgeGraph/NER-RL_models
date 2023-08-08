import yaml
import random
import torch
import numpy as np
import json

from transformers import AdamW

from tempRelationProcessor import TempRelationProcessor
from relationModel import MyBertForRelation

from torch import nn

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

    data = [json.loads(line) for line in open('../data/all_data_transformed.json', 'r')]

    processor = TempRelationProcessor(data, **args['processor'])
    train_loader, val_loader, test_loader, y_test = processor.run()

    model = MyBertForRelation(model_name='bert-base-uncased', num_rel_labels=2)
    device = device=args['torch']['device']
    model.to(device)

    optimizer = AdamW(model.bert.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx = batch
            input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx = input_ids.to(device), input_mask.to(device), segment_ids.to(device), label_id.to(device), sub_idx.to(device), obj_idx .to(device)
            logits = model(input_ids, segment_ids, input_mask, sub_idx, obj_idx )
            loss = criterion(logits.view(-1, 2), label_ids.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader)}")