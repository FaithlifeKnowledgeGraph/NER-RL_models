# Script to run trainer class and different components
import torch

from relationDataLoader import RelationDataLoader, RelationDataset
from relationModels import RelationModels
from relationTrainer import RelationTrainer

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split

loader = RelationDataLoader(json_file='data/dev.json')

numerical_data = loader.load_data()

X_sentences = []
X_entities = []
y = []

for data in numerical_data:
    X_sentences.extend(data['sentences'])
    X_entities.extend(data['ner'])
    y.extend(data['relation'])

print(len(X_entities))

# Convert list of sentences to PyTorch tensors and pad sequences
X_sentences = pad_sequence([torch.tensor(sentence) for sentence in X_sentences], batch_first=True)

# Convert list of entities to PyTorch tensors
# Assuming each entity is a 2D list where each inner list is a pair of numericalized start and end indices
X_entities = torch.tensor(X_entities, dtype=torch.float32)

# Convert list of relations to PyTorch tensors
# Assuming each relation is a list of numericalized indices
y = torch.tensor(y)

X_sentences_train, X_sentences_test, X_entities_train, X_entities_test, y_train, y_test = train_test_split(
    X_sentences, X_entities, y, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = RelationDataset(X_sentences_train, X_entities_train, y_train)
test_dataset = RelationDataset(X_sentences_test, X_entities_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

word_to_idx = numerical_data['word_to_idx']

model = RelationModels(vocab_size=len(word_to_idx),
                       embedding_dim=50,
                       hidden_dim=50,
                       output_dim=2)

trainer = RelationTrainer([], model)
