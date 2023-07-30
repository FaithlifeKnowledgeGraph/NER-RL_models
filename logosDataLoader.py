import json 
from itertools import combinations
from typing import List

OUTPUT_FILE_PATH = "data/transformed_data.json"

class LogosPUREDataLoader:
    def __init__(self, path_to_logos_PURE, max_data_size=50000) -> None:
        self.docs = self._load_old_file(path_to_logos_PURE)
        self.max_data_size = max_data_size
        self.transformed_data = self._transform_old_file()
        
        self.data = self._load_json_file(OUTPUT_FILE_PATH)

    
    def _load_json_file(self, output_file_path: str) -> List[List[str]]:
        with open(output_file_path) as f:
            json_data = json.load(f)
        return json_data

    def _load_old_file(self, path_to_logos_PURE):
        docs = [json.loads(line) for line in open(path_to_logos_PURE)]
        return docs
    
    def generate_entity_pairs(self, entities):
        return list(combinations(entities, 2))

    def _transform_old_file(self):
        # Placeholder for the transformed data
        transformed_data = []

        # Process each record in the original data
        for record in self.docs:
            # Unpack the record
            doc_key = record['doc_key']
            sentences = record['sentences']
            entities = record['ner']
            relations = record['relations']

            # Cumulative length of previous sentences
            cumulative_len = 0

            # Iterate over the sentences, entities, and relations
            for sentence, sentence_entities, sentence_relations in zip(sentences, entities, relations):
                sentence_str = ' '.join(sentence)  # convert list of tokens into a string

                # If there are no entities in the sentence, continue to the next sentence
                if not sentence_entities:
                    cumulative_len += len(sentence)  # update cumulative length
                    continue

                # Create entity dictionary with start_pos and end_pos, and add entity name
                sentence_entities = [{'entity': ' '.join(sentence[entity[0]-cumulative_len:entity[1]-cumulative_len+1]), 
                                    'start_pos': entity[0]-cumulative_len, 'end_pos': entity[1]-cumulative_len,
                                    'doc_start_pos': entity[0], 'doc_end_pos': entity[1]} 
                                    for entity in sentence_entities]

                # Generate all possible pairs of entities
                entity_pairs = self.generate_entity_pairs(sentence_entities)

                # Create a separate record for each pair of entities
                for pair in entity_pairs:
                    # Check if there is a relation between the pair
                    relation_val = 0  # default to no relation
                    if sentence_relations:  # if there are relations in the sentence
                        for relation in sentence_relations:
                            if ((pair[0]['doc_start_pos'] == relation[0] and pair[0]['doc_end_pos'] == relation[1]) and
                                (pair[1]['doc_start_pos'] == relation[2] and pair[1]['doc_end_pos'] == relation[3]) and
                                relation[4] == 'rel:TRUE'):
                                relation_val = 1
                                break
                    
                    # Create the new record without the document level positions
                    new_record = {
                        'sentence': sentence_str,
                        'entities': [{'entity': pair[0]['entity'], 'start_pos': pair[0]['start_pos'], 'end_pos': pair[0]['end_pos']},
                                    {'entity': pair[1]['entity'], 'start_pos': pair[1]['start_pos'], 'end_pos': pair[1]['end_pos']}],
                        'relation': relation_val
                    }
                    transformed_data.append(new_record)

                cumulative_len += len(sentence)  # update cumulative length

        # Display the first few entries of the transformed data
        transformed_data = transformed_data[:self.max_data_size]

        self.create_new_file(OUTPUT_FILE_PATH, transformed_data)

        return transformed_data
    
    def create_new_file(self, output_file_path: str, transformed_data: List[List[str]]):
        # Write the transformed data to a JSON file
        with open(output_file_path, 'w') as output_file:
            json.dump(transformed_data, output_file)