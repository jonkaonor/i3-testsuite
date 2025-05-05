import os
from datetime import datetime
from i3_testsuite.TaskStrategy import ImageClassificationStrategy, ARCStrategy  
from i3_testsuite.I3Strategy import BaselineStrategy, ContextDocumentStrategy, MultiQueryStrategy, CombinedStrategy 
from i3_testsuite.utils import log_kv_pairs, log_delimiter

class I3TestSuite:
    """Runs a full test cycle of the i3 / LLM framework for a specific task  


    This class accepts configuration settings, sets up the testing environment, 
    initializes task and i3-specific logic, builds prompts, sends them to the model, 
    scores the results, and logs configuration settings, prompts, and task results. 
    It supports multiple types of task strategies (like image classification) and i3
    framework strategies (like context or multiquery).

    Attributes:
        base_data_path (str): Path to the base directory containing prompt and image data.
        model_name (str): Name of the LLM model to be used.
        task_strategy (TaskStrategy): The task logic to apply, such as classification or ARC.
        i3_strategy (I3Strategy): The i3 logic / strategy to apply, such as baseline or context.
        num_train_examples (int): Number of training examples per class to include in the prompt.
        num_test_examples (int): Number of test examples to include in the prompt, used in scoring.
    """
    def __init__(self,
                 base_data_path: str,
                 model_name: str,
                 task_strategy: str,
                 i3_strategy: str,
                 num_train_examples: int,
                 num_test_examples: int):

        """Initializes the test suite with the chosen model, task, and i3 strategy.

        Verifies that the base data directory exists and that the specified task
        and i3 strategies are valid. Logs the experimental configuration.

        Raises:
            FileNotFoundError: If the base data directory does not exist.
            ValueError: If the task or i3 strategy names are invalid.
        """

        self.base_data_path = base_data_path
        self.model_name = model_name
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples 

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
        self.task_strategy = task_strategy_map[task_strategy](self.base_data_path, self.num_train_examples, self.num_test_examples)

        # Define the i3 frameworks that are supported by the test suite
        i3_strategy_map = {
            "baseline": BaselineStrategy,
            "context": ContextDocumentStrategy,
            "multiquery": MultiQueryStrategy,
            "combined": CombinedStrategy
        }

        # Check that the i3 strategy input is valid and set the i3 strategy 
        if i3_strategy not in i3_strategy_map:
            raise ValueError(f"Unsupported i3_strategy: {i3_strategy}")
        self.i3_strategy = i3_strategy_map[i3_strategy](self.base_data_path)

        # Log the configuration settings to the log file 
        experiment_metadata = { 
            "Date/Time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task_strategy":      task_strategy,
            "i3_strategy":        i3_strategy,
            "model_name":         model_name,
            "num_train_examples": num_train_examples,
            "num_test_examples":  num_test_examples,
        }

        log_kv_pairs(self.base_data_path, experiment_metadata)

    def execute_test(self): 
        """Runs the main test pipeline for the selected task and strategy.

        This includes:
            - Generating a prompt using the task strategy.
            - Enhancing the prompt using the i3 strategy (e.g., context, multiquery).
            - Sending the prompt to the LLM via the litellm API.
            - Scoring the LLM's output and logging results.
            - Logs the LLM response, total token usage, and final score.
        """
        # Generate initial llm prompt with train/test examples and task prompt
        litellm_prompt = self.task_strategy.task_prompt()

        # Add additional context if applicable to the prompt 
        litellm_prompt = self.i3_strategy.i3_prompt(litellm_prompt)

        # Send the prompt to the specified LLM model via the litellm API for processing 
        response = self.i3_strategy.execute_api_calls(self.model_name, litellm_prompt)

        # Calculate the score of the llm, log the scores, and print the score to console
        score = self.task_strategy.llm_task_score(response)
        print(f"Overall score is {score} out of {self.num_test_examples} test examples")

        # Log the llm input and response
        log_kv_pairs(self.base_data_path, {"Total Tokens Used": response.get("usage", {}).get("total_tokens"),"Model Response": response['choices'][0]['message']['content']})
        log_delimiter(self.base_data_path)


