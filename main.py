import sys
import os
import re
import argparse
from dotenv import load_dotenv

from src.components.preprocessor import Preprocessor
from src.components.fact_extractor import FactExtractor
from src.components.entity_extractor import EntityExtractor

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
if os.getenv("ENTITY_EXTRACTOR_PROMPT_PATH", None) != None:
    print("=" * 80)
    print("RUNNING ENTITY EXTRACTOR".center(80))
    print("=" * 80)
    entity_extractor = EntityExtractor(verbose=False)
    prompt_tokens, completion_tokens, success_num, all_num = entity_extractor.run()
    print("\n" + "=" * 80)
    print(f"Entity Extractor Complete: {success_num}/{all_num} items | Tokens: {prompt_tokens + completion_tokens:,}")
    print("=" * 80 + "\n")






# total_prompt_tokens = preprocessor_prompt_tokens + fact_extractor_prompt_tokens
# total_completion_tokens = preprocessor_completion_tokens + fact_extractor_completion_tokens
# total_tokens = total_prompt_tokens + total_completion_tokens