import i3_testsuite

# Create and run an i3 instance with current directory
bot = i3_testsuite.I3TestSuite(
    base_data_path="./data",
    model_name="gpt-4o",
    num_train_examples = 1,
    num_test_examples = 8,
    task_strategy="image_classification",
    i3_strategy="context"
)

bot.execute_test()

