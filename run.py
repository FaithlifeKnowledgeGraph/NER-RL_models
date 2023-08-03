import yaml

from relationProcessor import RelationProcessor
from relationModels import RelationModels, SimpleRelationExtractionModel
from relationTrainer import RelationTrainer
from logosDataLoader import LogosPUREDataLoader

def parse_yaml(f_path='config.yaml'):
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

args = parse_yaml()

print("Training Model with args: ", args)

set_global_seed(args['torch']['seed'])

loader = LogosPUREDataLoader(**args['loader'])
processor = RelationProcessor(loader.data, **args['processor'])
processor.add_typed_markers(verbose=True)
train_loader, val_loader, test_loader, y_test = processor.create_dataset()
vocab_size = processor.get_vocab_size()

relation_model = SimpleRelationExtractionModel(vocab_size, **args['nn_model'])
model = RelationModels(relation_model, args)

trainer = RelationTrainer(args, model, y_test)
trainer.run(train_loader, val_loader, test_loader)

