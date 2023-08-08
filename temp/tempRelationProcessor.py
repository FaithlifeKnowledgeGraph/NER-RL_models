# Process data and create dataloader for models
# For binary relation extraction

import random
import pickle
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split

from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class TempRelationProcessor:

    def __init__(self, data: list[dict], bert_tokenizer_name: str, batch_size: int):
        self.data = data
        self.bert_tokenizer_name = bert_tokenizer_name
        self.batch_size = batch_size

    def run(self):
        # tokenizer = BertTokenizer.from_pretrained(self.bert_tokenizer_name, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE))
        # ner_labels = ['concept', 'writing', 'person', 'place']

        # tokenizer = self._add_marker_tokens(tokenizer, ner_labels)

        # features = self._add_typed_markers(self.data, tokenizer)

        # print(len(tokenizer))
        # tokenizer_len = 30538

        with open('../data/data.pkl', 'rb') as f:
            features = pickle.load(f)

        train_loader, val_loader, test_loader, y_test = self._create_all_dataset(features)

        return train_loader, val_loader, test_loader, y_test
        
    def _add_marker_tokens(self, tokenizer, ner_labels):
        new_tokens = []

        for label in ner_labels:
            new_tokens.append('<SUBJ_START=%s>' % label)
            new_tokens.append('<SUBJ_END=%s>' % label)
            new_tokens.append('<OBJ_START=%s>' % label)
            new_tokens.append('<OBJ_END=%s>' % label)

        tokenizer.add_tokens(new_tokens)

        return tokenizer

    def _add_typed_markers(self, data, tokenizer):
        """
        Inspired by PURE
        """
        
        CLS = "[CLS]"
        SEP = "[SEP]"

        def get_special_token(w):
            return ('<' + w + '>').lower()
        
        max_tokens = 0
        total_tokens = 0
        features = []
        for (sent_idx, sentence) in enumerate(data):
            if sent_idx % 10000 == 0:
                print(f"Adding typed markers: {sent_idx} of {len(data)}")

            subj_entity = sentence['entities'][0]
            obj_entity  = sentence['entities'][1]

            # Create typed markers
            SUBJECT_START_MARKER = get_special_token(f"SUBJ_START={subj_entity['type']}")
            SUBJECT_END_MARKER  = get_special_token(f"SUBJ_END={subj_entity['type']}")
            OBJECT_START_MARKER  = get_special_token(f"OBJ_START={obj_entity['type']}")
            OBJECT_END_MARKER  = get_special_token(f"OBJ_END={obj_entity['type']}")

            # Create marked sentence
            sub_idx = 0
            obj_idx = 0
            marked_sentence = []
            marked_sentence.append(CLS)
            for i, token in enumerate(sentence['sentence']):
                if i == subj_entity['start_pos']:
                    sub_idx = len(marked_sentence)
                    marked_sentence.append(SUBJECT_START_MARKER)
                if i == obj_entity['start_pos']:
                    obj_idx = len(marked_sentence)
                    marked_sentence.append(OBJECT_START_MARKER)

                for sub_token in tokenizer.tokenize(token):
                    marked_sentence.append(token)

                if i == subj_entity['end_pos']:
                    marked_sentence.append(SUBJECT_END_MARKER)
                if i == obj_entity['end_pos']:
                    marked_sentence.append(OBJECT_END_MARKER)
            marked_sentence.append(SEP)

            max_tokens = max(max_tokens, len(marked_sentence))
            total_tokens += len(marked_sentence)

            input_ids = tokenizer.convert_tokens_to_ids(marked_sentence)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(marked_sentence)
            label_id = sentence['relation']

            features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            sub_idx=sub_idx,
                            obj_idx=obj_idx))

        print("Adding typed markers: Done")
        print(f"Total tokens: {total_tokens}")
        print(f"Max tokens  : {max_tokens}")

        # with open('data/data.pkl', 'wb') as f:
        #     pickle.dump(features, f)

        return features

    def _create_all_dataset(self, features):
        random.shuffle(features)

        features_len = len(features)
        print(features_len)
        start = 0
        end = start + int(features_len * 0.15)
        val_features = features[start:end]

        start = end
        end = start + int(features_len * 0.15)
        test_features = features[start:end]
        
        start = end
        train_features = features[start:]

        print(len(train_features))
        print(len(val_features))
        print(len(test_features))
        
        train_loader = self._create_a_dataset(train_features)
        val_loader = self._create_a_dataset(val_features)
        test_loader, y_test = self._create_a_dataset(test_features, return_labels=True)
        
        return train_loader, val_loader, test_loader, y_test

    def _create_a_dataset(features, return_labels=False):

        all_input_ids = pad_sequence([LongTensor(f.input_ids) for f in features],
                                    batch_first=True)
        all_input_mask = pad_sequence([LongTensor(f.input_mask) for f in features],
                                    batch_first=True)
        all_segment_ids = pad_sequence([LongTensor(f.segment_ids) for f in features],
                                    batch_first=True)
        all_label_id = pad_sequence([LongTensor([f.label_id]) for f in features],
                                    batch_first=True)
        all_sub_idx = pad_sequence([LongTensor([f.sub_idx]) for f in features],
                                    batch_first=True)
        all_obj_idx = pad_sequence([LongTensor([f.obj_idx]) for f in features],
                                    batch_first=True)
        data = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_id, all_sub_idx,
                            all_obj_idx)
        dataloader = DataLoader(data,
                                batch_size=self.batch_size)
        
        if return_labels:
            return dataloader, all_label_id
        
        return dataloader    

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx,
                 obj_idx):
        self.input_ids  : list[int] = input_ids
        self.input_mask : list[int] = input_mask
        self.segment_ids: list[int] = segment_ids
        self.label_id   : int       = label_id
        self.sub_idx    : int       = sub_idx
        self.obj_idx    : int       = obj_idx