import sys
import os
import re

from src.components.preprocessor_n import Preprocessor
from src.components.fact_extractor_n import FactExtractor



# Run the preprocessor
print("Running Preprocesor")
preprocessor = Preprocessor(verbose=False)
preprocessor_prompt_tokens, preprocessor_completion_tokens = preprocessor.run()
print(preprocessor_prompt_tokens,preprocessor_completion_tokens)

# Running the content Fact Extractor
print("Running Fact Extractor")
fact_extractor = FactExtractor(verbose=False)
fact_extractor_prompt_tokens, fact_extractor_completion_tokens, success_rate = fact_extractor.run()
print(fact_extractor_prompt_tokens,fact_extractor_completion_tokens, f"fact extractor success rate: {success_rate}")


total_prompt_tokens = preprocessor_prompt_tokens + fact_extractor_prompt_tokens
total_completion_tokens = preprocessor_completion_tokens + fact_extractor_completion_tokens
total_tokens = total_prompt_tokens + total_completion_tokens