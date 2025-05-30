# i3-testsuite

This repository contains a Python script and package for automating testing and logging of i3 framework / LLM experiments. The package goal is to speed up LLM experimentation by automating prompt submission to the LLM API through `litellm`, task scoring, and experiment results logging. Currently the test suite supports image classification tasks for two basic LLM prompting modes, `baseline` and `context`. The test suite has been tested for functionality with the `gpt-4o` model of OpenAI. Extensions to handle additional LLM prompting modes, tasks, and LLM models are planned for the near future. 

## Structure Overview
### Package Overview
- `core.py` - The primary class is `I3TestSuite` which takes a task-specific input prompt and training / test data and converts it into the format specified by the OpenAI ChatCompletions API. The class then makes an API call to the specified LLM via the `litellm` package and logs the results to a log file.  
- `TaskStrategy.py` and `I3Strategy.py` - These modules contain classes that support the `I3TestSuite` class by varying the behavior of certain functions depending on the current task or i3 framework.
- `utils.py` - This module contains general utility functions. 

### Directory Overview
| Folder/File      | Description                             |
| ---------------- | --------------------------------------- |
| `data/`          | Contains task data, prompts, and logs   |
| `data/images/`   | Stores image files grouped by label     |
| `data/prompts/`  | Contains task input prompts for the LLM |
| `data/logs/`     | Directory where logs are saved          |
| `i3-testsuite/`  | Core Python package code                |
| `test_script.py` | Example usage script                    |


## Requirements
- Python 3.8+
- OpenAI API key with credits
- Git

## Installation Instructions
### Quick Start Procedure
```bash
git clone https://github.com/jonkaonor/i3-testsuite.git
cd i3-testsuite
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python test_script.py
```
Note: You must replace `sk-...` with your own API key and export will only set the environment variable for the current terminal session. 

### Full Procedure
1. Navigate to your project directory.
2. Clone the repository with `git clone https://github.com/jonkaonor/i3-testsuite.git` and navigate into the main directory with `cd i3-testsuite`.
3. Activate a virtual environment if desired. Use `pip install -r requirements.txt` to install necessary Python packages.
4. Set the environment variable `OPENAI_API_KEY` to the value of the API key for whichever LLM API you are using. This is how `litellm` accesses the API key. Note: You must purchase OpenAI API credits separately, as access is not included with a ChatGPT Plus subscription. The cost of each API call to `gpt-4o` depends on the number and resolution of images and the length of the prompt. With downscaled images, an 8-image call typically costs around half a cent. You can further reduce costs by lowering image resolution.

## Using the Test Suite Package

### Configuration Settings 
- `base_data_path`: This is the path to the directory containing the `images`, `prompts` and `logs` folders.
- `model_name`: This is the name of the LLM model as specified by litellm.
- `select_train_examples`: This parameter sets the method for selecting training examples and can be `random` or `manual`. When using the `manual` selection mode, relative file paths to the training images should be written to the `training_images_list.txt` file in the `data/images` directory. You must specify at least one training example for every possible image class. 
- `num_train_examples`: This is the number of training examples to be used per class.  This parameter is only used for `random` selection of training examples and is ignored for `manual` selection.
- `max_output_tokens`: This parameter specifies the maximum number of output tokens used in the LLM response. If set too low, LLM behavior becomes unpredictable and generally poor. 
- `num_test_examples`: This is the number of test examples for the LLM to classify in total. 
- `task_strategy`: This parameter specifies the task. Currently `image_classification` is supported and `arcagi` support is pending.
- `i3_strategy`: This parameter specifies the i3 framework, which is the LLM prompting strategy. Currently there are two modes: `baseline` which adds only basic task information to the LLM prompt from `data/prompts/image_classification` and `context` which adds both the basic task information from `baseline` and additional task knowledge from `data/prompts/context_prompt` to the LLM prompt. `multiquery` and `combined` options are pending. 

### General Task-Setup / Experimentation Procedure for Image Classification Tasks
1. Set the configuration settings in `test_script.py` for the specific task. 
2. Place the images you wish to classify in the `data/images` directory. Each image should be placed in a directory with the class label as the directory name. For example, in the oranges dataset images are either labeled A or B so there are two directories in `images`. Set the method for selecting training images via the `select_train_examples` input parameter. 
3. Modify the prompts in the `data/prompts` directory, changing the text in `context_prompt` and `image_classification_prompt`. These are added at the beginning of the prompt sent to the LLM. 
4. Run the test script by executing the command `python test_script.py` or `python3 test_script.py`. There will be a pause while the request is processed followed by the LLM output being shown along with the score. 
5. Experiment results will be logged to `data/logs/experiment_log`.
6. Repeat for further experiments.

### Sample Test Script Output
<pre> ✅ API test succeeded. Response:
Answers: A, B, A, B, A, B, B, A

Reasoning: 

- **First Test Image**: The stem appears green with a noticeable protrusion, matching the characteristics of orange A.
- **Second Test Image**: The stem appears flat with more black, resembling orange B.
- **Third Test Image**: The protruding green stem suggests it is orange A.
- **Fourth Test Image**: The stem looks flat and darker, matching orange B.
- **Fifth Test Image**: The green protruding stem is indicative of orange A.
- **Sixth Test Image**: The stem appears flat with more black, similar to orange B.
- **Seventh Test Image**: Again, a flat, less prominent stem points to orange B.
- **Eighth Test Image**: The green stem is protruding, indicating it is orange A.
Overall score is 6 out of 8 test examples
</pre>

## License
MIT License. See `LICENSE` file for details.

## Contributing
Pull requests and suggestions are welcome! Feel free to open an issue or PR.


