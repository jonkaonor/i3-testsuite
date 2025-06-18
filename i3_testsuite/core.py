import os
from datetime import datetime
from i3_testsuite.TaskStrategy import ImageClassificationStrategy, ARCStrategy  
from i3_testsuite.PromptDesignStrategy import BasicStrategy, BasicWithContextStrategy, I3Strategy
from i3_testsuite.utils import log_kv_pairs, log_delimiter

class I3TestSuite:
    """Class implements functionality for testing how various LLMs and prompt design approaches 
    perform on ML tasks. 


    This class accepts configuration settings, sets up the testing environment, 
    initializes task and prompt design-specific logic, builds prompts, sends them to the model 
    via the LLM API, scores the results, and logs configuration settings, prompts, and task results. 
    It supports multiple types of task strategies (like image classification) and prompt design strategies 
    (like basic, basic with context, or i3).

    Attributes:
        base_data_path (str): Path to the base directory containing prompt and image data.
        model_name (str): Name of the LLM model to be used.
        task_strategy (TaskStrategy): The task logic to apply, such as classification or ARC.
        prompt_design_strategy (PromptDesignStrategy): The prompt design to apply, such as basic or i3.
        num_train_examples (int): Number of training examples per class to include in the prompt. 
        num_test_examples (int): Number of test examples to include in the prompt, used in scoring.
    """
    def __init__(self,
                 base_data_path: str,
                 model_name: str,
                 task_strategy: str,
                 prompt_design_strategy: str,
                 select_train_examples: str,
                 num_train_examples: int,
                 num_test_examples: int,
                 max_output_tokens: int):

        """Initializes the test suite with the chosen model, task, and prompt design strategy.

        Verifies that the base data directory exists and that the specified task
        and prompt design strategies are valid. Logs the experimental configuration.

        Raises:
            FileNotFoundError: If the base data directory does not exist.
            ValueError: If the task or prompt design strategy names are invalid.
        """

        self.base_data_path = base_data_path
        self.model_name = model_name
        self.select_train_examples = select_train_examples
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples 
        self.max_output_tokens = max_output_tokens

        # Check that the specified base data directory exists 
        if not os.path.isdir(self.base_data_path):
            raise FileNotFoundError(f"Data directory not found: {self.base_data_path}")

        # Define the tasks that are supported by the test suite 
        task_strategy_map = {
            "image_classification": ImageClassificationStrategy,
            "arc_agci": ARCStrategy
        }

        # Check that the task strategy string is valid and set the task strategy
        if task_strategy not in task_strategy_map:
            raise ValueError(f"Unsupported task_strategy: {task_strategy}")
        self.task_strategy = task_strategy_map[task_strategy](self.base_data_path, self.select_train_examples, self.num_train_examples, self.num_test_examples)

        # Define the prompt modes that are supported by the test suite
        prompt_design_strategy_map = {
            "basic": BasicStrategy,
            "basic_with_context": BasicWithContextStrategy,
            "i3": I3Strategy,
        }

        # Check that the prompt mode input is valid and set the prompt mode 
        if prompt_design_strategy not in prompt_design_strategy_map:
            raise ValueError(f"Unsupported prompt_design_strategy: {prompt_design_strategy}")
        self.prompt_design_strategy = prompt_design_strategy_map[prompt_design_strategy](self.base_data_path, self.max_output_tokens)

        # Log the configuration settings to the log file 
        experiment_metadata = { 
            "Date/Time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_strategy":      task_strategy,
            "prompt_design_strategy": prompt_design_strategy,
            "model_name":         model_name,
            "select_train_examples": select_train_examples,
            "num_train_examples": num_train_examples,
            "num_test_examples":  num_test_examples,
        }

        log_kv_pairs(self.base_data_path, experiment_metadata)

    def execute_test(self): 
        """Runs the main test pipeline for the selected task and strategy.

        This includes:
            - Generating a prompt using the task strategy.
            - Enhancing the prompt using the prompt design strategy (e.g., basic, i3).
            - Sending the prompt to the LLM via the litellm API.
            - Scoring the LLM's output and logging results.
            - Logs the LLM response, total token usage, and final score.
        """
        # Generate initial llm prompt with train/test examples and task prompt
        litellm_prompt = self.task_strategy.task_prompt()

        # Add additional context if applicable to the prompt 
        litellm_prompt = self.prompt_design_strategy.i3_prompt(litellm_prompt)

        # Send the prompt to the specified LLM model via the litellm API for processing 
        response = self.prompt_design_strategy.execute_api_calls(self.model_name, litellm_prompt)

        # Calculate the score of the llm, log the scores, and print the score to console
        score = self.task_strategy.llm_task_score(response)
        print(f"Overall score is {score} out of {self.num_test_examples} test examples")

        # Log the llm input and response
        log_kv_pairs(self.base_data_path, {"Total Tokens Used": response.get("usage", {}).get("total_tokens"),"Model Response": response['choices'][0]['message']['content']})
        log_delimiter(self.base_data_path)

