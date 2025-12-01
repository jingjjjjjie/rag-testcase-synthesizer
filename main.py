import sys
import os
import re
import argparse
from dotenv import load_dotenv

from src.components.preprocessor import Preprocessor
from src.components.fact_extractor import FactExtractor
from src.components.entity_extractor import EntityExtractor
from src.components.add_entity_id import AddEntityId
from src.components.entity_eliminator import EntityEliminator
from src.components.propose_generator import ProposeGenerator
from src.components.final_answer_generator import FinalAnswerGenerator

parser = argparse.ArgumentParser(description='RAG Test Case Synthesizer')
parser.add_argument('--env', type=str, default='multi_hop.env', help='Path to environment file (default: single_hop.env)')
                    
args = parser.parse_args()

load_dotenv()
load_dotenv(args.env)


# # Run the preprocessor
# if os.getenv("PREPROCESSOR_PDF_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING PREPROCESSOR".center(80))
#     print("=" * 80)
#     preprocessor = Preprocessor(verbose=False)
#     prompt_tokens, completion_tokens, success_num, all_num = preprocessor.run()
#     print("\n" + "=" * 80)
#     print(f"Preprocessor Complete: {success_num}/{all_num} steps | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

# # Running the content Fact Extractor
# if os.getenv("FACT_EXTRACTOR_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING FACT EXTRACTOR".center(80))
#     print("=" * 80)
#     fact_extractor = FactExtractor(save_interval=10)
#     prompt_tokens, completion_tokens, success_num, all_num = fact_extractor.run()
#     print("\n" + "=" * 80)
#     print(f"Fact Extractor Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

# Running the Entity Extractor
# if os.getenv("ENTITY_EXTRACTOR_PROMPT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING ENTITY EXTRACTOR".center(80))
#     print("=" * 80)
#     entity_extractor = EntityExtractor(verbose=False)
#     prompt_tokens, completion_tokens, success_num, all_num = entity_extractor.run()
#     print("\n" + "=" * 80)
#     print(f"Entity Extractor Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

# if os.getenv("ADD_ENTITY_ID_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING ADD ENTITY ID".center(80))
#     print("=" * 80)
#     add_entity_id = AddEntityId()
#     processed_count, entity_id_count = add_entity_id.run()
#     print("\n" + "=" * 80)
#     print(f"Add Entity ID Complete: {processed_count} entries processed | {entity_id_count} entity IDs assigned")
#     print("=" * 80 + "\n")

# if os.getenv("ENTITY_ELIMINATOR_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING ENTITY ELIMINATOR".center(80))
#     print("=" * 80)
#     entity_eliminator = EntityEliminator()
#     entity_eliminator.run()
#     print("\n" + "=" * 80)
#     print("Entity Eliminator Complete")
#     print("=" * 80 + "\n")

# if os.getenv("PROPOSE_GENERATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING PROPOSE GENERATOR".center(80))
#     print("=" * 80)
#     PROPOSE_GENERATOR_SAVE_INTERVAL = int(os.getenv("PROPOSE_GENERATOR_SAVE_INTERVAL", None))
#     propose_generator = ProposeGenerator(save_interval=PROPOSE_GENERATOR_SAVE_INTERVAL)
#     prompt_tokens, completion_tokens, success_num, all_num = propose_generator.run()
#     print("\n" + "=" * 80)
#     print(f"Propose Generator Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

if os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
    print("=" * 80)
    print("RUNNING FINAL ANSWER GENERATOR".center(80))
    print("=" * 80)
    FINAL_ANSWER_GENERATOR_SAVE_INTERVAL = int(os.getenv("FINAL_ANSWER_GENERATOR_SAVE_INTERVAL", None))
    final_answer_generator = FinalAnswerGenerator(save_interval=FINAL_ANSWER_GENERATOR_SAVE_INTERVAL)
    prompt_tokens, completion_tokens, success_num, all_num = final_answer_generator.run()
    print("\n" + "=" * 80)
    print(f"Final Answer Generator Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
    print("=" * 80 + "\n")



# total_prompt_tokens = preprocessor_prompt_tokens + fact_extractor_prompt_tokens
# total_completion_tokens = preprocessor_completion_tokens + fact_extractor_completion_tokens
# total_tokens = total_prompt_tokens + total_completion_tokens