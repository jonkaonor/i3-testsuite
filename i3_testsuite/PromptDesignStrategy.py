from abc import ABC, abstractmethod
import os
import litellm
from i3_testsuite.utils import log_kv_pairs

class PromptDesignStrategy(ABC):
    """Abstract base class for prompt design strategies.

    Subclasses should implement the i3_prompt method to modify the prompt
    before sending it to the model.
    """
    def __init__(self, base_data_path, max_output_tokens):
        self.base_data_path = base_data_path
        self.max_output_tokens = max_output_tokens

    @abstractmethod
    def i3_prompt(self, litellm_prompt):
        pass

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

    def i3_prompt(self, litellm_prompt):
        # Return the original prompt since the baseline strategy is to add no context
        return litellm_prompt

class BasicWithContextStrategy(PromptDesignStrategy):
    """Strategy that injects a knowledge module / context document into the prompt."""
    def i3_prompt(self, litellm_prompt):
        # Load the context prompt / knowledge module from file 
        context_prompt_file_path = os.path.join(self.base_data_path, "prompts", "context_prompt")

        with open(context_prompt_file_path, "r", encoding="utf-8") as f:
            context_prompt_text = f.read()

        # Add the context prompt after the initial task prompt but before the training / test examples 
        litellm_prompt.insert(1, {"type": "text", "text": context_prompt_text})

        # Log the context prompt used in the log file
        log_kv_pairs(self.base_data_path, {"Input Context Text": context_prompt_text})

        return litellm_prompt

class I3Strategy(PromptDesignStrategy):
    """Strategy for executing multiple LLM queries per input prompt."""
    def i3_prompt(self, litellm_prompt):
        pass

    def execute_api_calls(self, model_name, litellm_prompt): 
        pass
