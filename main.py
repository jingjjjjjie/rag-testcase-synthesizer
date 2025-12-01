import sys
import os
import re
import argparse
from dotenv import load_dotenv

from src import PROJECT_ROOT
from src.components.preprocessor import Preprocessor
from src.components.fact_extractor import FactExtractor
from src.components.entity_extractor import EntityExtractor
from src.components.add_entity_id import AddEntityId
from src.components.entity_eliminator import EntityEliminator
from src.components.propose_generator import ProposeGenerator
from src.components.final_answer_generator import FinalAnswerGenerator
from src.components.rephrase_generator import RephraseGenerator
from src.components.rephrase_generator_part import RephraseGeneratorPart
from src.components.rephrase_generator_hybrid import RephraseGeneratorHybrid
from src.components.paraphraser import Paraphraser
from src.components.rephrase_evaluator import rephrase_evaluator

parser = argparse.ArgumentParser(description='RAG Test Case Synthesizer')
parser.add_argument('--env', type=str, default='single_hop.env', help='Path to environment file (default: single_hop.env)')
                    
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

# if os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING FINAL ANSWER GENERATOR".center(80))
#     print("=" * 80)
#     FINAL_ANSWER_GENERATOR_SAVE_INTERVAL = int(os.getenv("FINAL_ANSWER_GENERATOR_SAVE_INTERVAL", None))
#     final_answer_generator = FinalAnswerGenerator(save_interval=FINAL_ANSWER_GENERATOR_SAVE_INTERVAL)
#     prompt_tokens, completion_tokens, success_num, all_num = final_answer_generator.run()
#     print("\n" + "=" * 80)
#     print(f"Final Answer Generator Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

# if os.getenv("REPHRASE_GENERATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("REPHRASE_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING REPHRASE GENERATOR".center(80))
#     print("=" * 80)
#     REPHRASE_GENERATOR_SAVE_INTERVAL = int(os.getenv("REPHRASE_GENERATOR_SAVE_INTERVAL", None))
#     rephrase_generator = RephraseGenerator(save_interval=REPHRASE_GENERATOR_SAVE_INTERVAL)
#     prompt_tokens, completion_tokens, success_num, all_num = rephrase_generator.run()
#     print("\n" + "=" * 80)
#     print(f"Rephrase Generator Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

# if os.getenv("REPHRASE_GENERATOR_PART_CONTENT_INPUT_PATH", None) != None or os.getenv("REPHRASE_GENERATOR_PART_ENTITYGRAPH_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING REPHRASE GENERATOR PART".center(80))
#     print("=" * 80)
#     REPHRASE_GENERATOR_SAVE_INTERVAL = int(os.getenv("REPHRASE_GENERATOR_SAVE_INTERVAL", None))
#     rephrase_generator_part = RephraseGeneratorPart(save_interval=REPHRASE_GENERATOR_SAVE_INTERVAL)
#     prompt_tokens, completion_tokens, success_num, all_num = rephrase_generator_part.run()
#     print("\n" + "=" * 80)
#     print(f"Rephrase Generator Part Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

# if os.getenv("REPHRASE_GENERATOR_HYBRID_CONTENT_INPUT_PATH", None) != None or os.getenv("REPHRASE_GENERATOR_HYBRID_ENTITYGRAPH_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING REPHRASE GENERATOR HYBRID".center(80))
#     print("=" * 80)
#     REPHRASE_GENERATOR_SAVE_INTERVAL = int(os.getenv("REPHRASE_GENERATOR_SAVE_INTERVAL", None))
#     rephrase_generator_hybrid = RephraseGeneratorHybrid(save_interval=REPHRASE_GENERATOR_SAVE_INTERVAL)
#     prompt_tokens, completion_tokens, success_num, all_num = rephrase_generator_hybrid.run()
#     print("\n" + "=" * 80)
#     print(f"Rephrase Generator Hybrid Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

# if os.getenv("SENTENCE_ORDER_CHANGER_CONTENT_INPUT_PATH", None) != None or os.getenv("SENTENCE_ORDER_CHANGER_ENTITYGRAPH_INPUT_PATH", None) != None:
#     print("=" * 80)
#     print("RUNNING SENTENCE ORDER CHANGER".center(80))
#     print("=" * 80)
#     SENTENCE_ORDER_CHANGER_SAVE_INTERVAL = int(os.getenv("SENTENCE_ORDER_CHANGER_SAVE_INTERVAL", None))
#     paraphraser = Paraphraser(save_interval=SENTENCE_ORDER_CHANGER_SAVE_INTERVAL)
#     prompt_tokens, completion_tokens, success_num, all_num = paraphraser.run()
#     print("\n" + "=" * 80)
#     print(f"Sentence Order Changer Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
#     print("=" * 80 + "\n")

if os.getenv("REPHRASE_EVALUATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("REPHRASE_EVALUATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
    print("Running Rephrase Evaluator")
    REPHRASE_EVALUATOR_INPUT_PATH, REPHRASE_EVALUATOR_OUTPUT_PATH = None, None
    if os.getenv("REPHRASE_EVALUATOR_CONTENT_INPUT_PATH", None) != None:
        REPHRASE_EVALUATOR_INPUT_PATH = os.getenv("REPHRASE_EVALUATOR_CONTENT_INPUT_PATH")
        REPHRASE_EVALUATOR_OUTPUT_PATH = os.getenv("REPHRASE_EVALUATOR_CONTENT_OUTPUT_PATH")
    elif os.getenv("REPHRASE_EVALUATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
        REPHRASE_EVALUATOR_INPUT_PATH = os.getenv("REPHRASE_EVALUATOR_ENTITYGRAPH_INPUT_PATH")
        REPHRASE_EVALUATOR_OUTPUT_PATH = os.getenv("REPHRASE_EVALUATOR_ENTITYGRAPH_OUTPUT_PATH")
    else:
        raise EnvironmentError("Environment variable 'REPHRASE_EVALUATOR_CONTENT_INPUT_PATH' or 'REPHRASE_EVALUATOR_ENTITYGRAPH_INPUT_PATH' is not set.")
    REPHRASE_EVALUATOR_SAVE_INTERVAL = int(os.getenv("REPHRASE_EVALUATOR_SAVE_INTERVAL", 100))
    REPHRASE_EVALUATOR_NUM_WORKERS = int(os.getenv("REPHRASE_EVALUATOR_NUM_WORKERS", 4))
    REPHRASE_EVALUATOR_MAX_GEN_TIMES = int(os.getenv("REPHRASE_EVALUATOR_MAX_GEN_TIMES", 300))

    rephrase_evaluator(
        os.path.join(PROJECT_ROOT, REPHRASE_EVALUATOR_INPUT_PATH),
        os.path.join(PROJECT_ROOT, REPHRASE_EVALUATOR_OUTPUT_PATH),
        REPHRASE_EVALUATOR_SAVE_INTERVAL,
        REPHRASE_EVALUATOR_NUM_WORKERS,
        REPHRASE_EVALUATOR_MAX_GEN_TIMES
    )
    print("-" * 50)


# total_prompt_tokens = preprocessor_prompt_tokens + fact_extractor_prompt_tokens
# total_completion_tokens = preprocessor_completion_tokens + fact_extractor_completion_tokens
# total_tokens = total_prompt_tokens + total_completion_tokens