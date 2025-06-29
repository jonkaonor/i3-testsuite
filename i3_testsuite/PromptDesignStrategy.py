from abc import ABC, abstractmethod
import os
import litellm
from i3_testsuite.utils import log_kv_pairs, log_delimiter

class PromptDesignStrategy(ABC):
    """Abstract base class for different prompt design strategies.

    This class implements different prompting approaches and prompt designs for interacting
    with a LLM. Subclasses should implement the execute_api_calls method, which determines
    what types of prompts are sent to the LLM and therefore how the LLM approaches the task. 
    """
    def __init__(self, base_data_path, task_strategy, num_test_examples, max_output_tokens, experiment_metadata):
        self.base_data_path = base_data_path
        self.task_strategy = task_strategy
        self.num_test_examples = num_test_examples
        self.max_output_tokens = max_output_tokens
        self.experiment_metadata = experiment_metadata

    @abstractmethod
    def execute_api_calls(self, model_name): 
        pass

class BasicStrategy(PromptDesignStrategy):
    """Strategy that prompts the LLM with a minimal task prompt.

    Prompts the LLM with a minimal task-specific prompt, typically 
    used to establish baseline LLM performance on the task. 
    """

    def execute_api_calls(self, model_name): 
        """Run an interactive test loop that calls the LiteLLM API

        This method continuously prompts the user to either:
        - Run an LLM test iteration using a task-specific prompt generated by 
        `self.task_strategy.basic_prompt()`, or
        - Exit the loop.

        When 'run' is selected:
        - Logs experiment metadata and prompt type.
        - Sends the formatted prompt to the LiteLLM API using the specified model.
        - Receives and displays the LLM's response.
        - Scores the response using `self.task_strategy.llm_task_score()`.
        - Logs the score, response content, and token usage.

        Args:
            model_name (str): The name of the model to use for the API call (e.g., "gpt-4o").

        Returns:
            None
        """
        while True:
            action = input("Type 'run' to execute one iteration of testing or 'exit' to quit: ").strip().lower()
            
            if action == 'run':
                # Log the experiment metadata
                log_kv_pairs(self.base_data_path, self.experiment_metadata)
                log_kv_pairs(self.base_data_path, {"Prompt Type": "Train and Test"})

                response = ""
                # Execute the API call to the specified LLM model 
                try:
                    response = litellm.completion(
                        model=model_name,
                        messages= self.task_strategy.basic_prompt(),
                        max_completion_tokens = self.max_output_tokens
                    )
                    message = response['choices'][0]['message']['content']
                    print(f"✅ API test succeeded. Response:\n{message}")

                    # Calculate the score of the llm, log the scores, and print the score to console
                    score = self.task_strategy.llm_task_score(response)
                    print(f"Overall score is {score} out of {self.num_test_examples} test examples")

                    # Log the llm input and response
                    log_kv_pairs(self.base_data_path, {"Total Tokens Used": response.get("usage", {}).get("total_tokens"),"Model Response": response['choices'][0]['message']['content']})
                    log_delimiter(self.base_data_path)

                except Exception as e:
                    print(f"❌ API test failed: {e}")

            elif action == 'exit':
                print("Exiting the program.")
                break

            else:
                print("Invalid input. Please type 'run' or 'exit'.")

class BasicWithContextStrategy(PromptDesignStrategy):
    """Strategy that adds a knowledge module / additional context to the basic prompt

    Prompts the LLM with the 'basic' prompt that has dditional user-defined context 
    to help the LLM tailor its response to the specific task. 
    """

    def execute_api_calls(self, model_name): 
        """ Run an interactive test loop that calls the LiteLLM API

        This method continuously prompts the user to either:
        - Run an LLM test iteration using a task-specific prompt generated by 
        `self.task_strategy.basic_with_context_prompt()`, or
        - Exit the loop.

        When 'run' is selected:
        - Logs experiment metadata and prompt type.
        - Sends the formatted prompt to the LiteLLM API using the specified model.
        - Receives and displays the LLM's response.
        - Scores the response using `self.task_strategy.llm_task_score()`.
        - Logs the score, response content, and token usage.

        Args:
            model_name (str): The name of the model to use for the API call (e.g., "gpt-4o").

        Returns:
            None
        """
        while True:
            action = input("Type 'run' to execute one iteration of testing or 'exit' to quit: ").strip().lower()
            
            if action == 'run':
                # Log the experiment metadata
                log_kv_pairs(self.base_data_path, self.experiment_metadata)
                log_kv_pairs(self.base_data_path, {"Prompt Type": "Train and Test"})

                response = ""
                # Execute the API call to the specified LLM model 
                try:
                    response = litellm.completion(
                        model=model_name,
                        messages= self.task_strategy.basic_with_context_prompt(),
                        max_completion_tokens = self.max_output_tokens
                    )
                    message = response['choices'][0]['message']['content']
                    print(f"✅ API test succeeded. Response:\n{message}")

                    # Calculate the score of the llm, log the scores, and print the score to console
                    score = self.task_strategy.llm_task_score(response)
                    print(f"Overall score is {score} out of {self.num_test_examples} test examples")

                    # Log the llm input and response
                    log_kv_pairs(self.base_data_path, {"Total Tokens Used": response.get("usage", {}).get("total_tokens"),"Model Response": response['choices'][0]['message']['content']})
                    log_delimiter(self.base_data_path)

                except Exception as e:
                    print(f"❌ API test failed: {e}")

            elif action == 'exit':
                print("Exiting the program.")
                break

            else:
                print("Invalid input. Please type 'run' or 'exit'.")

class I3Strategy(PromptDesignStrategy):
    """Strategy for testing the i3 methodology for utilizing a LLM.
    
    Prompts the LLM utilizing the i3 methodology, where the LLM is prompted to 
    generate a classification prompt (text) during the training phase which is 
    then used to improve the LLM's performance on the test set of the task 
    during the testing phase. 
    """

    def execute_api_calls(self, model_name): 
        """Run an interactive loop to execute training or testing LLM API calls

        This method allows the user to interactively choose between 'train', 'test', or 'exit':
        
        - 'train':
            - Logs experiment metadata with prompt type 'Train'
            - Builds a training prompt via `self.task_strategy.i3_train_prompt()`
            - Sends the prompt to the LiteLLM API using the specified model
            - Parses and logs the model-generated classification prompt using `i3_classification_prompt_parser()`
            - Logs token usage and response content

        - 'test':
            - Logs experiment metadata with prompt type 'Test'
            - Builds a test prompt via `self.task_strategy.i3_test_prompt()`
            - Sends the prompt to the LiteLLM API using the specified model
            - Scores the response using `llm_task_score()` and logs the result
            - Logs token usage and response content

        - 'exit': Ends the loop and terminates the program.

        Args:
            model_name (str): The name of the model to use for the API call (e.g., "gpt-4o").

        Returns:
            None
        """

        while True:
            action = input("Type 'train', 'test', or 'exit' to quit: ").strip().lower()

            if action == 'train':
                # Log the experiment metadata
                log_kv_pairs(self.base_data_path, self.experiment_metadata)
                log_kv_pairs(self.base_data_path, {"Prompt Type": "Train"})

                response = ""
                # Execute the API call to the specified LLM model 
                try:
                    response = litellm.completion(
                        model=model_name,
                        messages= self.task_strategy.i3_train_prompt(),
                        max_completion_tokens = self.max_output_tokens
                    )
                    message = response['choices'][0]['message']['content']
                    print(f"✅ API test succeeded. Response:\n{message}")

                    # Parse and log the LLM's generated classification prompt 
                    self.task_strategy.i3_classification_prompt_parser(message)

                    # Log the llm input and response
                    log_kv_pairs(self.base_data_path, {"Total Tokens Used": response.get("usage", {}).get("total_tokens"),"Model Response": response['choices'][0]['message']['content']})
                    log_delimiter(self.base_data_path)

                except Exception as e:
                    print(f"❌ API test failed: {e}")

            elif action == 'test':
                # Log the experiment metadata
                log_kv_pairs(self.base_data_path, self.experiment_metadata)
                log_kv_pairs(self.base_data_path, {"Prompt Type": "Test"})

                response = ""
                # Execute the API call to the specified LLM model 
                try:
                    response = litellm.completion(
                        model=model_name,
                        messages= self.task_strategy.i3_test_prompt(),
                        max_completion_tokens = self.max_output_tokens
                    )
                    message = response['choices'][0]['message']['content']
                    print(f"✅ API test succeeded. Response:\n{message}")

                    # Calculate the score of the llm, log the scores, and print the score to console
                    score = self.task_strategy.llm_task_score(response)
                    print(f"Overall score is {score} out of {self.num_test_examples} test examples")

                    # Log the llm input and response
                    log_kv_pairs(self.base_data_path, {"Total Tokens Used": response.get("usage", {}).get("total_tokens"),"Model Response": response['choices'][0]['message']['content']})
                    log_delimiter(self.base_data_path)

                except Exception as e:
                    print(f"❌ API test failed: {e}")

            elif action == 'exit':
                print("Exiting the program.")
                break
            else:
                print("Invalid input. Please type 'train', 'test', or 'exit'.")