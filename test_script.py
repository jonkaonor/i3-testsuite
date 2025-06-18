import i3_testsuite

# Create and run an i3 instance with current directory
bot = i3_testsuite.I3TestSuite(
    base_data_path="./data",
    model_name="gpt-4o",
    select_train_examples = 'random',
    num_train_examples = 1,
    num_test_examples = 8,
    max_output_tokens = 5000,
    task_strategy="image_classification",
    prompt_design_strategy="basic_with_context"
)

bot.execute_test()

