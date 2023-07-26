# Script to run trainer class and different components

from relationDataLoader import RelationDataLoader
from relationModels import RelationModels
from relationTrainer import RelationTrainer

loader = RelationDataLoader()
loader.load_data()

model = RelationModels(None)

trainer = RelationTrainer(model)

