from abc import ABC, abstractmethod
import os 
import re
from i3_testsuite.utils import log_kv_pairs, log_delimiter
from i3_testsuite.utils import load_images_as_dict_arr
from i3_testsuite.utils import image_train_test_split
from i3_testsuite.utils import encode_image_to_base64_data_uri

class TaskStrategy(ABC):
    """Abstract base class for defining task-specific strategies.

    Subclasses must implement methods for generating a prompt and scoring the LLM's response
    against a test set. Shared attributes include file paths and the number of training/test
    examples to use.
    """
    def __init__(self, base_data_path, select_train_examples, num_train_examples, num_test_examples):
        self.base_data_path = base_data_path
        self.select_train_examples = select_train_examples
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples 

    @abstractmethod
    def task_prompt(self):
        pass

    @abstractmethod
    def llm_task_score(self, response):
        pass

class ImageClassificationStrategy(TaskStrategy):
    def __init__(self, base_data_path, select_train_examples, num_train_examples, num_test_examples):
        super().__init__(base_data_path, select_train_examples, num_train_examples, num_test_examples)
        self.train_set = []
        self.test_set = []

    def task_prompt(self):
        """
        Loads the task images and the task prompt from the 'images' and 'prompts' folders 
        in the base data directory. Creates a structured prompt suitable for the litellm 
        API / OpenAI ChatCompletions API.

        Return:
            list: A list of dictionaries representing the structured prompt to send to the 
            model. Each dictionary is either a text instruction or an image in base64 format.
    
        """
        # Create an array of image dictionaries, one per image class label 
        image_dict_arr = load_images_as_dict_arr(self.base_data_path)

        # Create a training set and test set from the 'images' directory
        self.train_set, self.test_set = image_train_test_split(self.base_data_path, image_dict_arr, self.select_train_examples, self.num_train_examples, self.num_test_examples)

        # Load the task_prompt from file 
        task_prompt_file_path = os.path.join(self.base_data_path, "prompts", "image_classification_prompt")

        with open(task_prompt_file_path, "r", encoding="utf-8") as f:
            task_prompt_text = f.read()

        # Create the initial litellm message prompt and add the task prompt to it 
        litellm_prompt = [
            {"type": "text", "text": task_prompt_text},
        ]

        # Add the training images to the prompt
        for image_path, label in self.train_set:
            litellm_prompt.append({
                "type": "text",
                "text": f"The following image is a training image whose label is: {label}. Please use this to help you in differentiating between the different images"
            })
            litellm_prompt.append({
                "type": "image_url",
                "image_url": {
                    "url": encode_image_to_base64_data_uri(image_path)
                }
            })

        # Add the test images to the prompt
        for image_path, label in self.test_set:
            litellm_prompt.append({
                "type": "text",
                "text": "The following image is a test image. Please predict it's classification and include it's classification in your output as part of 'Answers:'"
            })
            litellm_prompt.append({
                "type": "image_url",
                "image_url": {
                    "url": encode_image_to_base64_data_uri(image_path)
                }
            })

        # Log the task prompt 
        log_kv_pairs(self.base_data_path, {"Task Input Text": task_prompt_text})

        return litellm_prompt

    def llm_task_score(self, response):
        """Scores predictions from an LLM response against a test set.

        This function looks for a line in the LLM's response that starts with "Answers:",
        extracts the predicted labels, and compares them to the correct labels in self.test_set.

        Args:
            response (dict): The dictionary representing the complete LLM output from the litellm API call.

        Returns:
            int: The number of correct predictions made by the LLM.

        Raises:
            ValueError: If the response format is unexpected or if the number of predicted
                        labels doesn't match the number of test examples.
        """
        # Extract the model's text content
        try:
            content = response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            error_message = "Could not find text content in the response object."
            log_kv_pairs(self.base_data_path, {"Fatal Error": error_message})
            log_delimiter(self.base_data_path)
            raise ValueError(error_message)
        
        # Strip leading whitespace and verify the output starts with "Answers:"
        stripped = content.lstrip()
        if not stripped.startswith("Answers:"):
            error_message = f"Expected response to start with 'Answers:', but got:\n{stripped[:50]!r}"
            log_kv_pairs(self.base_data_path, {"Fatal Error": error_message})
            log_delimiter(self.base_data_path)
            raise ValueError(error_message)
        
        # Create a list of the predicted classes from the first line of output 
        first_line = stripped.splitlines()[0]
        # Regex to capture everything after "Answers:"
        m = re.match(r"^Answers:\s*(.*)$", first_line)
        if not m:
            error_message = f"Could not parse answers line:\n{first_line!r}"
            log_kv_pairs(self.base_data_path, {"Fatal Error": error_message})
            log_delimiter(self.base_data_path)
            raise ValueError(error_message)
        answers_part = m.group(1).strip()      # e.g. "A, B, A, B"
        predicted_labels = [lbl.strip() for lbl in answers_part.split(",")]

        # Check that the number of predicted labels from the LLM matches the number of test set examples
        if len(predicted_labels) != len(self.test_set): 
            error_message = f"Number of predicted labels ({len(predicted_labels)}) does not match " f"number of test examples ({len(self.test_set)})."
            log_kv_pairs(self.base_data_path, {"Fatal Error": error_message})
            log_delimiter(self.base_data_path)
            raise ValueError(error_message)

        # Compare the predicted labels and the known labels to calculate the score 
        score_count = 0
        for (pred, (_, true_label)) in zip(predicted_labels, self.test_set):
            if pred == true_label:
                score_count += 1

        # Log the task score
        log_kv_pairs(self.base_data_path, {"score": score_count})

        # Log each test image and its true label + predicted label
        for (image_path, true_label), pred_label in zip(self.test_set, predicted_labels):
            fname = os.path.basename(image_path)
            log_kv_pairs(self.base_data_path, {
                fname: f"({true_label}, {pred_label})"
            })

        return score_count


class ARCStrategy(TaskStrategy):
    """Placeholder strategy class for ARC-style reasoning tasks.

    Intended to implement task_prompt and scoring logic for ARC (Abstraction and
    Reasoning Corpus) tasks in the future. Currently not implemented.
    """
    def task_prompt(self):
        pass

    def llm_task_score(self, response):
        pass

