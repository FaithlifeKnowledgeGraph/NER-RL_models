from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch import LongTensor
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader


class RelationProcessor:
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.vocab = self.create_vocab()

    def create_dataset(self):
        sequences, labels = self.create_sequences_and_labels()
        X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.5, random_state=42)


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
            word_counter.update(item['sentence'])
        
        vocabulary = {word: i+1 for i, (word, _) in enumerate(word_counter.most_common())}
        vocabulary['<UNK>'] = 0
        vocabulary['<PAD>'] = len(vocabulary)

        return vocabulary
    
    def create_sequences_and_labels(self):
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

    def add_typed_markers(self, verbose=False):
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
            if verbose and sent_idx % 10000 == 0:
                print(f"Adding typed markers to sentence: {sent_idx} of {len(self.data)}")

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

            # Replace old sentence with new ones
            self.data[sent_idx]['sentence'] = marked_sentence

            if verbose:
                max_tokens = max(max_tokens, len(marked_sentence))
                total_tokens += len(marked_sentence)

        if verbose:
            print("Adding typed markers done")
            print(f"Total tokens: {total_tokens}")
            print(f"Max tokens: {max_tokens}")