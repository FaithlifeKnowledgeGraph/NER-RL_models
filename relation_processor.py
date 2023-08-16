# Process data and create dataloader for models
# For binary relation extraction

import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader


class RelationProcessor:
    """Data Processor for relation extraction task

    Process data and covert text into numbers 
    Create train/val/test DataLoader for models usage

    Attributes:
        data: Data in correct format
        batch_size: Size of each batch being loaded from DataLoader
        val_test_ratio: Total ratio of data is split into val and test set.
            Example: val_test_ratio = 0.3 means val_ratio = test_ration = 0.15
        vocab: Dict of unique tokens (in given data) mapped to numbers

    """

    def __init__(self, data: list[dict], batch_size: int, val_test_ratio: float):
        """Initialize RelationProcessor 

        Args:
            data: Data in correct format
            batch_size: Size of each batch being loaded from DataLoader
            val_test_ratio: Total ratio of data is split into val and test set.
                Example: val_test_ratio = 0.3 means val_ratio = test_ration = 0.15

        """
    
        self.data = data
        self.val_test_ratio = val_test_ratio
        self.batch_size = batch_size
        self.vocab = None

    def run(self) -> tuple[DataLoader, DataLoader, DataLoader, LongTensor]:
        """Run RelationProcessor 

        Create vocab out of given data, add typed markers
        Create train/val/test DataLoader for models usage

        Returns:
            DataLoader for train dataset
            DataLoader for val dataset
            DataLoader for test dataset
            Category labels for test dataset

        """

        self.vocab = self._create_vocab()

        self._add_typed_markers()
        
        sequences, labels = self._create_sequences_and_labels()

        train_loader, val_loader, test_loader, y_test = self._create_dataset(sequences, labels)

        return train_loader, val_loader, test_loader, y_test

    def get_vocab_size(self) -> int:
        """Calculate size of vocabulary 

        Returns:
            Size of vocabulary  

        """

        return len(self.vocab)

    def _create_vocab(self) -> dict:
        word_counter = Counter()

        for item in self.data:
            word_counter.update(item['sentence'])
        
        vocabulary = {word: i+1 for i, (word, _) in enumerate(word_counter.most_common())}
        vocabulary['<UNK>'] = 0
        vocabulary['<PAD>'] = len(vocabulary)

        return vocabulary

    def _add_typed_markers(self) -> None:
        """
        Inspired by PURE
        """

        CLS = "[CLS]"
        SEP = "[SEP]"
        for marker in [CLS, SEP]:
            if marker not in self.vocab:
                self.vocab[marker] = len(self.vocab)

        def get_special_token(w):
            return ('<' + w + '>').lower()
        
        max_tokens = 0
        total_tokens = 0
        for (sent_idx, sentence) in enumerate(self.data):
            if sent_idx % 10000 == 0:
                print(f"Adding typed markers: {sent_idx} of {len(self.data)}")

            subj_entity = sentence['entities'][0]
            obj_entity  = sentence['entities'][1]

            # Create typed markers
            SUBJECT_START_MARKER = get_special_token(f"SUBJ_START={subj_entity['type']}")
            SUBJECT_END_MARKER  = get_special_token(f"SUBJ_END={subj_entity['type']}")
            OBJECT_START_MARKER  = get_special_token(f"OBJ_START={obj_entity['type']}")
            OBJECT_END_MARKER  = get_special_token(f"OBJ_END={obj_entity['type']}")
            
            # Add markers to vocab if not in
            for marker in [SUBJECT_START_MARKER, SUBJECT_END_MARKER, 
                            OBJECT_START_MARKER, OBJECT_END_MARKER]:
                if marker not in self.vocab:
                    self.vocab[marker] = len(self.vocab)

            marked_sentence = []
            marked_sentence.append(CLS)
            for i, token in enumerate(sentence['sentence']):
                if i == subj_entity['start_pos']:
                    marked_sentence.append(SUBJECT_START_MARKER)
                if i == obj_entity['start_pos']:
                    marked_sentence.append(OBJECT_START_MARKER)

                marked_sentence.append(token)

                if i == subj_entity['end_pos']:
                    marked_sentence.append(SUBJECT_END_MARKER)
                if i == obj_entity['end_pos']:
                    marked_sentence.append(OBJECT_END_MARKER)
            marked_sentence.append(SEP)

            self.data[sent_idx]['sentence'] = marked_sentence

            max_tokens = max(max_tokens, len(marked_sentence))
            total_tokens += len(marked_sentence)

        print("Adding typed markers: Done")
        print(f"Total tokens: {total_tokens}")
        print(f"Max tokens  : {max_tokens}")

    def _create_sequences_and_labels(self) -> tuple[LongTensor, LongTensor]:
        sequences = []
        labels = []

        for item in self.data:
            sequence = [self.vocab[word] for word in item['sentence']]
            sequences.append(sequence)
            labels.append(item['relation'])
        
        sequences = pad_sequence([LongTensor(sequence) for sequence in sequences], 
                                batch_first=True)
        labels = LongTensor(labels)
        return sequences, labels

    def _create_dataset(self, sequences, labels) -> tuple[DataLoader, DataLoader, DataLoader, LongTensor]:
        # 'stratify' makes sure classes are distributed evenly 
        X_train, X_val_test, y_train, y_val_test = train_test_split(sequences, labels,
                                    stratify=labels, test_size=self.val_test_ratio, shuffle=True)

        X_val, X_test, y_val, y_test  = train_test_split(X_val_test, y_val_test,
                                    stratify=y_val_test, test_size=0.5, shuffle=True)

        # Create Dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size)

        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader, y_test
