from pathlib import Path
from beam import App, Image, Runtime, Volume, VolumeType

from training import TrainingAPI
from training_modules.config import training_arguments
from training_modules.initialize import initialize

training_app = App(
    name="train_qa",
    runtime=Runtime(
        cpu=4,
        memory="32Gi",
        gpu="A10G",
        image=Image(python_version="python3.10", python_packages="reqs.txt"),
    ),
    volumes=[
        Volume(path="./qa_dataset", name="qa_dataset"),
        Volume(
            path="./output",
            name="train_qa_output",
            volume_type=VolumeType.Persistent,
        ),
        Volume(
            path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)

@training_app.run()
def train():
    """
    Trains a machine learning model using the specified configuration file and dataset.

    Args:
        config_file (str): The path to the configuration file for the training process.
        output_dir (str): The directory where the trained model will be saved.
        dataset_dir (str): The directory where the training dataset is located.
        env_file_path (str, optional): The path to the environment variables file. Defaults to ".env".
        logging_config_path (str, optional): The path to the logging configuration file. Defaults to "logging.yaml".
        model_cache_dir (str, optional): The directory where the trained model will be cached. Defaults to None.
    """

    # Be sure to initialize the environment variables before importing any other modules.
    logging_config_path = "logging.yaml"
    env_file_path = ".env"
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    path_dataset = "./qa_dataset/dataset" #"dataset"
    path_model_cache = "./model_cache" #if path_model_cache else None

    training_api = TrainingAPI(
        training_config=training_arguments,
        path_dataset=path_dataset,
        path_model_cache=path_model_cache,
    )
    training_api.train_model()

if __name__ == "__main__":
    train()