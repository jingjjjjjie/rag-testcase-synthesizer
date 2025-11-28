import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..tools.api import call_api_qwen
from ..tools.json_utils import save_json, load_json
from ..tools.string_utils import read_text_file
from .. import PROJECT_ROOT


def extract_entity_from_output(input_text):
    """Extract entities and relationships from model output"""
    entity_pattern = r'\("entity"\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\)'
    relationship_pattern = r'\("relationship"\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\)'

    # Replace placeholders with actual delimiters
    tuple_delimiter = '<tuple_delimiter>'
    record_delimiter = '<record_delimiter>'
    completion_delimiter = '<completion_delimiter>'

    # Parse entities
    entities = re.findall(entity_pattern, input_text)
    parsed_entities = [
        {
            "entity_name": match[0],
            "entity_description": match[1]
        }
        for match in entities
    ]

    # Parse relationships
    relationships = re.findall(relationship_pattern, input_text)
    parsed_relationships = []
    for match in relationships:
        # Extract sentence numbers from the sentences_used field
        sentences_raw = match[3]

        # Remove square brackets if present and split by comma
        sentences_clean = sentences_raw.strip("[]")

        numbers_and_ranges = re.findall(r'\d+-\d+|\d+', sentences_clean)

        # Create a formatted string for sentences_used
        formatted_sentence_numbers = ','.join(numbers_and_ranges)
        formatted_sentence_numbers = f'{formatted_sentence_numbers}'

        parsed_relationships.append({
            "source_entity_name": match[0].strip(),
            "target_entity_name": match[1].strip(),
            "relationship_description": match[2].strip(),
            "sentences_used": formatted_sentence_numbers
        })

    # Validate output format
    is_complete = completion_delimiter in input_text

    return {
        "entities": parsed_entities,
        "relationships": parsed_relationships,
        "is_complete": is_complete
    }


class EntityExtractor:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.TEMPERATURE = None
        self.ENTITY_EXTRACTOR_NUM_WORKERS = None
        self.ENTITY_EXTRACTOR_PROMPT_PATH = None
        self.ENTITY_EXTRACTOR_INPUT_PATH = None
        self.ENTITY_EXTRACTOR_OUTPUT_PATH = None
        self.ENTITY_EXTRACTOR_SAVE_INTERVAL = None

        if os.getenv("ENTITY_EXTRACTOR_PROMPT_PATH", None) != None:
            self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.6"))
            self.ENTITY_EXTRACTOR_NUM_WORKERS = int(os.getenv("ENTITY_EXTRACTOR_NUM_WORKERS", "4"))
            self.ENTITY_EXTRACTOR_SAVE_INTERVAL = int(os.getenv("ENTITY_EXTRACTOR_SAVE_INTERVAL", "20"))
            self.ENTITY_EXTRACTOR_PROMPT_PATH = os.getenv("ENTITY_EXTRACTOR_PROMPT_PATH")
            self.ENTITY_EXTRACTOR_INPUT_PATH = os.getenv("ENTITY_EXTRACTOR_INPUT_PATH")
            self.ENTITY_EXTRACTOR_OUTPUT_PATH = os.getenv("ENTITY_EXTRACTOR_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variables are not defined correctly")

        self.ENTITY_EXTRACTOR_PROMPT_PATH = os.path.join(PROJECT_ROOT, self.ENTITY_EXTRACTOR_PROMPT_PATH)
        self.ENTITY_EXTRACTOR_INPUT_PATH = os.path.join(PROJECT_ROOT, self.ENTITY_EXTRACTOR_INPUT_PATH)
        self.ENTITY_EXTRACTOR_OUTPUT_PATH = os.path.join(PROJECT_ROOT, self.ENTITY_EXTRACTOR_OUTPUT_PATH)

    def process_input(self, cur_input, entity_extractor_prompt, i):
        """Process a single input to extract entities and relationships"""
        try:
            context = cur_input['context']
            cur_entity_extractor_prompt = entity_extractor_prompt.replace('[[CONTEXT]]', context)
            entity_extractor_response, prompt_tokens, completion_tokens, _ = call_api_qwen(
                cur_entity_extractor_prompt,
                temperature=self.TEMPERATURE
            )
            extract_entity = extract_entity_from_output(entity_extractor_response)
            entities, relationships, is_complete = (
                extract_entity['entities'],
                extract_entity['relationships'],
                extract_entity['is_complete']
            )

            # Filter entities
            filtered_entities = []
            for entity in entities:
                if "entity_name" in entity and "entity_description" in entity:
                    filtered_entities.append(entity)

            # Filter relationships to only include those with valid entities
            entity_names = [entity['entity_name'] for entity in filtered_entities]
            filtered_relationships = []
            for relationship in relationships:
                source_entity_name = relationship['source_entity_name']
                target_entity_name = relationship['target_entity_name']
                if source_entity_name in entity_names and target_entity_name in entity_names:
                    filtered_relationships.append(relationship)

            result = {
                **cur_input,
                'entity': filtered_entities,
                'relationship': filtered_relationships
            }
            return result, i, prompt_tokens, completion_tokens
        except Exception as e:
            if self.verbose:
                print(f"Error processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None, 0, 0

    def run(self):
        """Main entity extraction pipeline"""
        if self.verbose:
            print("Loading entity extraction prompt and input data......")

        # Load input data (resume from output if exists)
        if os.path.exists(self.ENTITY_EXTRACTOR_OUTPUT_PATH):
            inputs = load_json(self.ENTITY_EXTRACTOR_OUTPUT_PATH)
            if self.verbose:
                print(f"Resuming from existing output: {len(inputs)} entries loaded")
        else:
            inputs = load_json(self.ENTITY_EXTRACTOR_INPUT_PATH)
            if self.verbose:
                print(f"Loaded {len(inputs)} inputs from {self.ENTITY_EXTRACTOR_INPUT_PATH}")

        # Load prompt
        entity_extractor_prompt = read_text_file(self.ENTITY_EXTRACTOR_PROMPT_PATH)

        all_num, success_num = 0, 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Multiple workers extracting entities concurrently
        if self.verbose:
            print(f"Extracting entities with {self.ENTITY_EXTRACTOR_NUM_WORKERS} workers......")

        with ThreadPoolExecutor(max_workers=self.ENTITY_EXTRACTOR_NUM_WORKERS) as executor:
            futures = []
            for i, cur_input in enumerate(inputs):
                if "entity" not in cur_input:
                    futures.append(executor.submit(self.process_input, cur_input, entity_extractor_prompt, i))

            all_num = len(futures)

            if self.verbose:
                iterator = tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True, desc="Extracting entities")
            else:
                iterator = as_completed(futures)

            for future in iterator:
                result, i, prompt_tokens, completion_tokens = future.result(timeout=10*60)
                if result != None:
                    inputs[i] = result
                    success_num += 1
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

                    # Save intermediate results
                    if success_num % self.ENTITY_EXTRACTOR_SAVE_INTERVAL == 0:
                        save_json(inputs, self.ENTITY_EXTRACTOR_OUTPUT_PATH)
                        if self.verbose:
                            print(f"Saved checkpoint: {success_num}/{all_num} processed")

        # Final save
        save_json(inputs, self.ENTITY_EXTRACTOR_OUTPUT_PATH)

        if self.verbose:
            print(f"Completed: {success_num}/{all_num} inputs processed successfully")

        # Print summary
        print(f"\nTotal prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Success rate: {success_num}/{all_num} = {success_num/all_num*100:.2f}%" if all_num > 0 else "Success rate: N/A")

        return total_prompt_tokens, total_completion_tokens, success_num, all_num


if __name__ == "__main__":
    entity_extractor = EntityExtractor()
    entity_extractor.run()
