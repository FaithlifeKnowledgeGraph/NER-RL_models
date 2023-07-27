# Load the relation data from the database
import json
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


class RelationDataLoader:

    def __init__(self, json_file) -> None:
        self.json_file = json_file
        self.relation_json_data = self._load_json_file()

    def _load_json_file(self) -> List[List[str]]:
        docs = [json.loads(line) for line in open(self.json_file)]
        for doc in docs:
            print(len(doc['sentences']), len(doc['ner']),
                  len(doc['relations']))

        return docs

    def load_data(self):
        data = self._load_json_file()
        entity_to_idx, relation_to_idx, word_to_idx = self.build_vocab(data)
        numerical_data = self.numericalize(data, entity_to_idx,
                                           relation_to_idx, word_to_idx)
        return numerical_data

    def build_vocab(self, data):
        entity_vocab = set()
        relation_vocab = set()
        sentence_vocab = set()

        entity_vocab.add('<UNK>')
        relation_vocab.add('<UNK>')
        sentence_vocab.add('<UNK>')

        for item in data:
            for ner_in_sentence in item['ner']:
                for ner in ner_in_sentence:
                    entity_vocab.add(ner[2])

            for rel_in_sentence in item['relations']:
                for rel in rel_in_sentence:
                    relation_vocab.add(rel[4])

            for sentence in item['sentences']:
                if sentence:
                    for word in sentence:
                        sentence_vocab.add(word)

        entity_to_idx = {entity: i for i, entity in enumerate(entity_vocab)}
        relation_to_idx = {
            relation: i for i, relation in enumerate(relation_vocab)
        }
        word_to_idx = {word: i for i, word in enumerate(sentence_vocab)}

        return entity_to_idx, relation_to_idx, word_to_idx

    def numericalize(self, data, entity_to_idx, relation_to_idx, word_to_idx):
        numerical_data = []

        for item in data:
            numerical_sentences = [[
                word_to_idx.get(word, word_to_idx['<UNK>'])
                for word in sentence
            ]
                                   for sentence in item['sentences']]

            numerical_ner = [[
                ner[0], ner[1],
                entity_to_idx.get(ner[2], entity_to_idx['<UNK>'])
            ] for ner_in_sentence in item['ner'] for ner in ner_in_sentence]

            numerical_relation = [[
                rel[0], rel[1], rel[2], rel[3],
                relation_to_idx.get(rel[4], relation_to_idx['<UNK>'])
            ] for rel_in_sentence in item['relations'] for rel in rel_in_sentence]

            numerical_data.append({
                'sentences': numerical_sentences,
                'ner': numerical_ner,
                'relation': numerical_relation,
                'word_to_idx': word_to_idx,  # delete later
            })

        return numerical_data


class RelationDataset(Dataset):

    def __init__(self, sentences, entities, relations) -> None:
        self.sentences = sentences
        self.entities = entities
        self.relations = relations

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        return self.sentences[idx], self.entities[idx], self.relations[idx]
