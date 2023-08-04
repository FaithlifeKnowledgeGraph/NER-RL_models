# Data loader for Logos data, load data and transform data to correct format if needed
# For binary relation extraction

import json 
from itertools import combinations
from typing import List

class LogosDataLoader:
    """Data loader for Logos data

    Load data and transform data to correct format if needed
    For binary relation extraction

    Attributes:
        data_path: Path to file contain data
        max_data_size: Maximum number of data points to use
        is_PURE_format: If given file contains data in PURE format 
        data: Data in correct format
    """

    def __init__(self, data_path: str, max_data_size: int = 50000, is_PURE_format: bool = False) -> None:
        """Initialize LogosDataLoader 

        Args:
            data_path: Path to file contain data
            max_data_size: Maximum number of data points to use
            is_PURE_format: If given file contains data in PURE format 
        """

        self.data_path = data_path
        self.max_data_size = max_data_size
        self.is_PURE_format = is_PURE_format
        self.data = None

    def run(self) -> List[dict]:
        """Run LogosDataLoader 

        Load data, and transform if specified so
        Data length is at most max_data_size

        Returns:
            List[dict]: data in correct format
        """

        if (self.is_PURE_format):
            PURE_format_data = self._load_json_file(self.data_path)
            transformed_data = self._transform_PURE_format_data(PURE_format_data)
        else:
            transformed_data = self._load_json_file(self.data_path)

        self.data = transformed_data[:self.max_data_size]
        return self.data

    def export_data(self, output_file_path: str)-> None:
        """Export data (in correct format) to a json file, for feature usage

        Args:
            output_file_path: path to desire output file, create file if not exist 
        """

        if self.data is None:
            print("No data to export!")

        with open(output_file_path, 'w') as output_file:
            for d in self.data:
                output_file.write(json.dumps(d, ensure_ascii=False)) 
                output_file.write('\n')

    def _load_json_file(self, file_path: str) -> List[dict]:
        data = [json.loads(line) for line in open(file_path, 'r')]
        return data

    def _transform_PURE_format_data(self, PURE_format_data: List[dict]) -> List[dict]:
        transformed_data = []

        for record in PURE_format_data:
            doc_key = record['doc_key']
            sentences = record['sentences']
            entities = record['ner']
            relations = record['relations']

            cumulative_len = 0

            for sentence, sentence_entities, sentence_relations in zip(sentences, entities, relations):

                # Delete dups entities
                unique_entity = []
                [unique_entity.append(entity) for entity in sentence_entities 
                    if entity not in unique_entity]
                sentence_entities = unique_entity

                # Delete dups relations
                unique_relation = []
                [unique_relation.append(relation) for relation in sentence_relations 
                    if relation not in unique_relation]
                sentence_relations = unique_relation

                if len(sentence_entities) <= 1:
                    cumulative_len += len(sentence)  
                    continue

                sentence_entities = [{'entity': ' '.join(sentence[entity[0]-cumulative_len:entity[1]-cumulative_len+1]), 
                                    'start_pos': entity[0]-cumulative_len, 'end_pos': entity[1]-cumulative_len,
                                    'doc_start_pos': entity[0], 'doc_end_pos': entity[1],
                                    'type': entity[2]} 
                                    for entity in sentence_entities]

                # Generate all possible pairs of entities
                entity_pairs = list(combinations(sentence_entities, 2))

                # Create a separate record for each pair of entities
                for pair in entity_pairs:
                    sub_ent = pair[0]
                    obj_ent = pair[1]

                    # Pos overlapping = anomalies 
                    if (obj_ent['doc_start_pos'] <= sub_ent['doc_start_pos'] <= obj_ent['doc_end_pos']) \
                        or (sub_ent['doc_start_pos'] <= obj_ent['doc_start_pos'] <= sub_ent['doc_end_pos']) \
                        or (obj_ent['doc_start_pos'] <= sub_ent['doc_end_pos'] <= obj_ent['doc_end_pos']) \
                        or (sub_ent['doc_start_pos'] <= obj_ent['doc_end_pos'] <= sub_ent['doc_end_pos']):
                        cumulative_len += len(sentence)  
                        continue

                    relation_val = 0
                    for relation in sentence_relations:
                        if ((sub_ent['doc_start_pos'] == relation[0] and sub_ent['doc_end_pos'] == relation[1]) and
                            (obj_ent['doc_start_pos'] == relation[2] and obj_ent['doc_end_pos'] == relation[3]) and
                            relation[4] == 'rel:TRUE'):
                            relation_val = 1
                            break
                    
                    new_record = {
                        'sentence': sentence,
                        'entities': [{'entity': sub_ent['entity'], 'start_pos': sub_ent['start_pos'], 
                                    'end_pos': sub_ent['end_pos'], 'type': sub_ent['type']},
                                    {'entity': obj_ent['entity'], 'start_pos': obj_ent['start_pos'],
                                    'end_pos': obj_ent['end_pos'], 'type': obj_ent['type']}],
                        'relation': relation_val
                    }
                    transformed_data.append(new_record)

                cumulative_len += len(sentence)

        return transformed_data
