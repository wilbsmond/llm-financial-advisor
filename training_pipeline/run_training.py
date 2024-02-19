from pathlib import Path
from beam import App, Image, Runtime, Volume, VolumeType

training_app = App(
    name="train_qa",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="A10G",
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(path="./finance_qa_dataset", name="finance_qa_dataset"),
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

    from training import TrainingAPI
    from training_modules.config import training_arguments

    path_output = './output'
    path_dataset = Path.cwd().parent / "dataset"
    path_model_cache = "./model_cache" if path_model_cache else None

    training_api = TrainingAPI(
        training_config=training_arguments,
        path_output=path_output,
        path_dataset=path_dataset,
        path_model_cache=path_model_cache,
    )
    training_api.train_model()

if __name__ == "__main__":
    train()