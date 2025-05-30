import os
import random
import base64
import mimetypes

def log_kv_pairs(base_data_path: str, kv_dict: dict):
    """Appends key-value pairs to the experiment log file.

    Each key/value pair in the dictionary is written on its own line
    in the format: 'key: value'. A blank line is added after all entries.

    Args:
        base_data_path (str): Base directory containing the logs subdirectory.
        kv_dict (dict): Dictionary of key-value pairs to write to the log.
    """
    log_path = os.path.join(base_data_path, "logs", "experiment_log")

    with open(log_path, "a", encoding="utf-8") as f:
        for key, value in kv_dict.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")


def log_delimiter(base_data_path: str, delimiter: str = "-"*150):
    """Appends a delimiter line to the experiment log file to distinguish between test runs

    Args:
        base_data_path (str): Base directory containing the logs subdirectory.
        delimiter (str, optional): A string to write to the log, typically dashes.
            Defaults to 150 dashes.
    """
    log_path = os.path.join(base_data_path, "logs", "experiment_log")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(delimiter + "\n")


def encode_image_to_base64_data_uri(path):
    """Encodes an image file as a base64 data URI string.

    This is the format used by the OpenAI API to embed image data directly 
    in structured prompts sent to LLMs, preserving format and portability.

    Args:
        path (str): Path to the image file.

    Returns:
        str: A base64-encoded data URI string representing the image.
    """
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "image/jpeg"  # fallback default
    with open(path, "rb") as f:
        encoded_bytes = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_bytes}"


def image_train_test_split(base_data_path, image_dict_arr, select_train_examples, num_train_examples, num_test_examples):
    """Splits image data into training and test sets via either random or manual sampling.

    In random sampling: 
    From each per-class dictionary of image paths, selects 'num_test_examples' training
    examples. Then selects test examples from the pool of remaining images randomly.
    Sampling is done without replacement (images are only added once to either set).

    In manual sampling:
    From the `select_train_examples.txt` file in the `images` directory, read in training 
    images and create the training set. Then selects test examples from the pool of 
    remaining images randomly. Sampling is done without replacement (images are only 
    added once to either set).

    Args:
        image_dict_arr (list): List of dictionaries mapping image_path -> label.
        num_train_examples (int): Number of training samples to select per class.
        num_test_examples (int): Number of total test samples to select across classes.

    Returns:
        tuple: Two lists — (train_set, test_set), each a list of (image_path, label) pairs.

    Raises:
        ValueError: If there are not enough images in a class or overall for sampling.
    """
    
    # Creates the training set, either with manually selected images or randomly selected images
    train_set = []
    test_set  = []

    if select_train_examples == 'random':  
        # Check that there are enough of each images for the number of training examples specified
        total_images = 0
        for dic in image_dict_arr:
            total_images += len(dic)
            if len(dic) < num_train_examples:
                label = next(iter(dic.values()), "<unknown>")
                raise ValueError(
                    f"Class '{label}' only has {total} examples, " +
                    f"but you requested {num_train_examples} examples for train"
                )

        # Check that there are enough images for the number of training / test examples specified
        if total_images < num_train_examples * len(image_dict_arr) + num_test_examples :
            raise ValueError(f"There are only {total_images} images which is insufficient for the number of train/test examples specified")

        # Randomly select a number of training examples from each class equal to num_train_examples and add them to the train_set
        for d in image_dict_arr:
            items = list(d.items())
            train_samples = random.sample(items, num_train_examples)
            train_set.extend(train_samples)
            # remove selected examples from the dict
            for path, _ in train_samples:
                d.pop(path)

    elif select_train_examples == 'manual':
        # Load the manually selected image file paths
        list_file = os.path.join(base_data_path, 'images', 'training_images_list.txt')
        if not os.path.isfile(list_file):
            raise ValueError(f"Manual training list file not found at '{list_file}'")
        with open(list_file, 'r') as f:
            rel_paths = [ln.strip() for ln in f if ln.strip()]

        # Add manually selected images to the training set 
        required_labels = {next(iter(d.values())) for d in image_dict_arr}
        seen_labels = set()
        for rel in rel_paths:
            abs_path = os.path.abspath(os.path.join(base_data_path, 'images', rel))
            if not os.path.isfile(abs_path):
                raise ValueError(f"Listed file '{rel}' does not exist.")
            matched = False
            for d in image_dict_arr:
                if abs_path in d:
                    label = d.pop(abs_path)
                    train_set.append((abs_path, label))
                    seen_labels.add(label)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"File '{rel}' not found in class directories.")
        missing = required_labels - seen_labels
        if missing:
            raise ValueError(f"Manual list missing images for classes: {missing}")

        # After manual image selection, ensure there are enough images remaininng for the test set 
        remaining = sum(len(d) for d in image_dict_arr)
        if remaining < num_test_examples:
            raise ValueError(f"After manual selection, only {remaining} images remain for {num_test_examples} tests.")

    else:
        # Case for invalid training example selection mode
        raise ValueError(
            f"Parameter select_train_examples must have value 'random' or 'manual', got '{select_train_examples}'"
        )

    # Randomly select a number of test examples from each class equal to num_test_examples and add them to the test_set
    remaining_images_arr = []
    for dic in image_dict_arr:
        remaining_images_arr.extend(dic.items())

    # Sample without replacement from the combined image pool
    test_samples = random.sample(remaining_images_arr, num_test_examples)
    test_set.extend(test_samples)

    # Remove sampled paths from dict
    for path, _ in test_samples:
        for d in image_dict_arr:
            if path in d:
                d.pop(path)
                break

    return train_set, test_set 

def load_images_as_dict_arr(base_data_path):
    """Loads labeled images from the 'images' directory into per-class dictionaries.

    Scans the subdirectories of base_data_path/images, treating each folder as a
    class label and mapping image file paths to that label.

    Args:
        base_data_path (str): Path to the root directory containing an 'images' folder.

    Returns:
        list: A list of dictionaries, each mapping image file paths to their label.
              Each dictionary represents one class.
    """
    images_dir = os.path.join(base_data_path, 'images')
    if not os.path.isdir(images_dir):
        # If 'images' directory doesn't exist, return empty list
        return []
    
    # Define allowed image extensions (lowercase for comparison)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    result = []
    
    # Iterate through each label folder inside the images directory
    for label_name in os.listdir(images_dir):
        label_path = os.path.join(images_dir, label_name)
        if not os.path.isdir(label_path):
            continue  # skip if not a directory
        
        # Initialize dictionary for this label
        label_mapping = {}
        for filename in os.listdir(label_path):
            # Check file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext in image_extensions:
                # Get absolute path and map it to the label
                file_abs_path = os.path.abspath(os.path.join(label_path, filename))
                label_mapping[file_abs_path] = label_name
        # Add the mapping dict if it contains any entries (ignore empty folders)
        if label_mapping:
            result.append(label_mapping)
    return result