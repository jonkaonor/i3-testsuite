# i3-testsuite

This repository contains a Python script and package for automating testing of LLM performance on various ML tasks using different prompt designs and approaches. The package goal is to speed up LLM experimentation by automating the following steps: prompt submission to the LLM API through `litellm`, task scoring, and experiment results logging. Currently the test suite supports image classification tasks for three LLM prompting approaches, `basic` and `basic_with_context`, and `i3`, which are described in more detail below. The test suite has been tested for functionality with the `gpt-4o` OpenAI model. Extensions to handle additional LLM prompting approaches, tasks, and LLM models are planned for the near future. Currently the default prompts and images stored in the `data` folder and the `test_script.py` and log files are configured for image classification of oranges. This can be used as a model example for familiarization with the test suite. 

## Structure Overview
### Package Overview
- `core.py` - The primary class is `I3TestSuite` which takes several user-defined prompts and training / test data and converts it into a LLM prompt in the format specified by the OpenAI ChatCompletions API. The class then makes an API call to the specified LLM via the `litellm` package and logs the experiment results to a log file.  
- `TaskStrategy.py` and `PromptDesignStrategy.py` - These modules contain classes that support the `I3TestSuite` class by varying the behavior of certain functions depending on the current task or prompting approach.
- `utils.py` - This module contains general utility functions. 

### Directory Overview
| Folder/File      | Description                              |
| ---------------- | -----------------------------------------|
| `data/`          | Contains task data, prompts, and logs    |
| `data/images/`   | Stores image files grouped by label      |
| `data/prompts/`  | Contains user-defined prompts for the LLM|
| `data/logs/`     | Directory where logs are saved           |
| `i3-testsuite/`  | Core Python package code                 |
| `test_script.py` | Example usage script                     |

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
- `prompt_design_strategy`: This parameter specifies the prompt design / prompting approach, which controls how the LLM is prompted. Currently there are three supported approaches: `basic` which adds only basic task instructions to the LLM prompt from `data/prompts/image_classification.txt` and `basic_with_context` which combines the final prompt generated by the `basic` approach and additional user-defined knowledge from `data/prompts/context_prompt.txt` to the LLM prompt. The `i3` approach splits training and testing into separate phases. In the training phase, `image_classification_i3_train_prompt.txt` is used to give task instructions and the user-defined `i3_context_prompt.txt` adds task-specific context to the prompt the LLM to generate a classifiation prompt that is stored in `image_classification_prompt.txt`. In the test phase, the classification prompt is utilized along with the `i3_context_prompt.txt` to perform the specified task on a test set. In all approaches a general system prompt for orienting the LLM to its purpose is added from `image_classification_system_prompt.txt`

### General Task-Setup / Experimentation Procedure for Image Classification Tasks
1. Set the configuration settings in `test_script.py` for the specific task. 
2. Place the images you wish to classify in the `data/images` directory. Each image should be placed in a directory with the class label as the directory name. For example, in the oranges dataset images are either labeled A or B so there are two directories in `images`. Set the method for selecting training images via the `select_train_examples` input parameter. Refer to the `Configuration Settings` section if using `manual` selection of training images. 
3. Modify the relevant prompts in the `data/prompts` directory depending on your chosen `prompt_design_strategy`. Refer to the `Configuration Settings` part of the README for specifics of the functionality of each prompt. 
4. Run the test script by executing the command `python test_script.py` or `python3 test_script.py`. An simple interactive loop will prompt you for input and allow you to run experiments. 
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


