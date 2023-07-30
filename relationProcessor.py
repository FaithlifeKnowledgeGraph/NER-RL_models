import pandas as pd

from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader


class RelationProcessor:
    def __init__(self, data: dict, batch_size: int, test_ratio: float):
        self.data = data
        self.batch_size = batch_size
        self.vocab = self.create_vocab()

    def create_even_class_size(self):
        df = pd.DataFrame(self.data)

        positive_class_df = df[df['relation'] == 1]
        positive_class_size = positive_class_df.shape[0]

        sampled_negative_class_df = df[df['relation'] == 0].sample(n=positive_class_size, random_state=42)

        df = pd.concat([positive_class_df, sampled_negative_class_df])

        print("Size of positive class: ", positive_class_size)
        print("Size of negative class: ", sampled_negative_class_df.shape[0])
        return df.to_dict(orient='records')

    def create_dataset(self):
        self.data = self.create_even_class_size()
        sequences, labels = self.create_sequences_and_labels()

        # Evenly divide the dataset distribution 
        X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.3, random_state=42)


        # Create Dataloaders
        batch_size = self.batch_size
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, test_loader, y_test

    def create_vocab(self):
        word_counter = Counter()
        for item in self.data:
            word_counter.update(item['sentence'].split())
        
        vocabulary = {word: i+1 for i, (word, _) in enumerate(word_counter.most_common())}
        vocabulary['<UNK>'] = 0
        vocabulary['<PAD>'] = len(vocabulary)

        return vocabulary

    def get_vocab_size(self):
        return len(self.vocab)
    
    def create_sequences_and_labels(self):
        sequences = []
        labels = []

        for item in self.data:
            sequence = [self.vocab[word] for word in item['sentence'].split()]
            sequences.append(sequence)
            labels.append(item['relation'])
        
        sequences = pad_sequence([LongTensor(sequence) for sequence in sequences], 
                                 batch_first=True)
        labels = LongTensor(labels)

        return sequences, labels