import os
import transformers
import torch
import logging
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai
import mlflow
import time

from constants import GEMINI_MODELS
from utils.processing import format_llm_input, format_llm_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LLM():
    def __init__(self, model_id, temperature=None, top_p=None, top_k=None):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._init_model()

    def _init_model(self):
        if self.model_id in GEMINI_MODELS:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(self.model_id)
        elif self.model_id == "meta-llama/Llama-3.1-8B-Instruct":
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto"
            )
        else:
            logger.error(f"Unrecognized model id: {self.model_id}")

    def _generate_with_retry(self, prompt, synthetic_data_schema=None, max_retries=5, retry_delay=65):
        """Generates content using the GenAI API with retry logic for ResourceExhausted errors.

        Args:
            model: The GenAI GenerativeModel object.
            prompt: The prompt string.
            synthetic_data_schema: The response schema.
            max_retries: The maximum number of retries.
            retry_delay: The delay (in seconds) between retries.

        Returns:
            The API response if successful, or None if the maximum retries are exceeded.
        """
        retries = 0
        while retries < max_retries:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        # response_mime_type="application/json",
                        # response_schema=synthetic_data_schema,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        top_p=self.top_p
                    )
                )
                return response.text, response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count

            except Exception as e:
                retries += 1
                print(f"Error (attempt {retries}/{max_retries}): {e}")
                time.sleep(retry_delay) # time.sleep(retry_delay * retries)
                continue

        print(f"Maximum retries ({max_retries}) exceeded.")
        return None

    def generate(self, prompt):
        # TODO: clean up
        input = format_llm_input(self.model_id, prompt)

        if self.model_id in GEMINI_MODELS:
            response, input_token_count, output_token_count = self._generate_with_retry(input)
        elif self.model_id == "meta-llama/Llama-3.1-8B-Instruct":
            # TODO: token count
            response = self.pipeline(input, max_new_tokens=256)
        else:
            logger.error(f"Unsupported model: {self.model_id}")

        formatted_out = format_llm_output(self.model_id, response)

        # print(f"Generated content: {formatted_out}")
        return formatted_out, input_token_count, output_token_count