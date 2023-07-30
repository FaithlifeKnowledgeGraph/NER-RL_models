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

def main(args):
    MAX_DATA_SIZE = 10000
    loader = LogosPUREDataLoader("data/train.json", MAX_DATA_SIZE)

    processor = RelationProcessor(args['nn_optimizer']['batch_size'], loader.data)
    train_loader, test_loader, y_test = processor.create_dataset()

    # Create model
    vocab_size = len(processor.vocab)
    model = SimpleRelationExtractionModel(vocab_size, **args['nn_model'])
    print("Created Model")

    model = RelationModels(model, args)
    trainer = RelationTrainer({}, model, y_test)
    trainer.run(train_loader, test_loader)


if __name__ == '__main__':
    args = parse_yaml()
    print("Training Model with args: ", args)

    main(args)
