from abc import ABC, abstractmethod
import os
import litellm
from i3_testsuite.utils import log_kv_pairs

class PromptDesignStrategy(ABC):
    """Abstract base class for prompt design strategies.

    Subclasses should implement the i3_prompt method to modify the prompt
    before sending it to the model.
    """
    def __init__(self, base_data_path, task_strategy, max_output_tokens):
        self.base_data_path = base_data_path
        self.task_strategy = task_strategy
        self.max_output_tokens = max_output_tokens

    def execute_api_calls(self, model_name, litellm_prompt): 
        """Calls the LiteLLM API with the given model and prompt and returns the response.

        Given the input prompt, format a prompt for the LLM in the the litellm API format
        (which matches the OpenAI ChatCompletions API format), make the API call and 
        return the llm's response. 

        Args:
            model_name (str): Name of the model to call (e.g., "gpt-4o").
            litellm_prompt (list): Prompt in LiteLLM API format to be sent as the message content.

        Returns:
            dict: The full response from the API call.
        """
        response = ""
        # Execute the API call to the specified LLM model 
        try:
            response = litellm.completion(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a virtual assistant tasked with solving a problem 
                        after being shown some training examples of how to solve the problem. You 
                        should always output your response in the specified output format and you 
                        should always attempt to solve the task problem. 
                        """
                    },
                    {
                        "role": "user",
                        "content": litellm_prompt
                    }
                ],
                max_completion_tokens = self.max_output_tokens
            )
            message = response['choices'][0]['message']['content']
            print(f"✅ API test succeeded. Response:\n{message}")
        except Exception as e:
            print(f"❌ API test failed: {e}")

        return response 

class BasicStrategy(PromptDesignStrategy):
    """Strategy that uses the input prompt without modification."""
    pass
class BasicWithContextStrategy(PromptDesignStrategy):
    """Strategy that injects a knowledge module / context document into the prompt."""
    pass

class I3Strategy(PromptDesignStrategy):
    """Strategy for executing multiple LLM queries per input prompt."""
    def execute_api_calls(self, model_name, litellm_prompt): 
        pass
