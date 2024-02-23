from pathlib import Path
from typing import Optional, Tuple
from datasets import Dataset
import comet_ml

import numpy as np
from peft import PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTTrainer

from training_modules import data, models

import logging
logger = logging.getLogger(__name__)


class BestModelToModelRegistryCallback(TrainerCallback):
    """
    Callback that logs the best model checkpoint to the Comet.ml model registry.

    Args:
        model_id (str): The ID of the model to log to the model registry.
    """
    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        """
        Returns the name of the model to log to the model registry.
        """
        return f"financial_assistant/{self._model_id}"
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of training.

        Logs the best model checkpoint to the Comet.ml model registry.
        """

        best_model_checkpoint = state.best_model_checkpoint
        has_best_model_checkpoint = best_model_checkpoint is not None
        if has_best_model_checkpoint:
            best_model_checkpoint = Path(best_model_checkpoint)
            logger.info(
                f"Logging best model from {best_model_checkpoint} to the model registry..."
            )

            self.to_model_registry(best_model_checkpoint)
        else:
            logger.warning(
                "No best model checkpoint found. Skipping logging it to the model registry..."
            )
    
    def to_model_registry(self, checkpoint_dir: Path):
        """
        Logs the given model checkpoint to the Comet.ml model registry.

        Args:
            checkpoint_dir (Path): The path to the directory containing the model checkpoint.
        """

        checkpoint_dir = checkpoint_dir.resolve()

        assert (
            checkpoint_dir.exists()
        ), f"Checkpoint directory {checkpoint_dir} does not exist"

        # Get the stale experiment from the global context to grab the API key and experiment ID.
        stale_experiment = comet_ml.get_global_experiment()
        # Resume the expriment using its API key and experiment ID.
        experiment = comet_ml.ExistingExperiment(
            api_key=stale_experiment.api_key, experiment_key=stale_experiment.id
        )
        logger.info(f"Starting logging model checkpoint @ {self.model_name}")
        experiment.log_model(self.model_name, str(checkpoint_dir))
        experiment.end()
        logger.info(f"Finished logging model checkpoint @ {self.model_name}")


class TrainingAPI:
    def __init__(self, training_config, path_dataset, path_model_cache):
        self.path_dataset = path_dataset
        self.path_model_cache = path_model_cache

        self.llm_template_name = "falcon"
        self.model_id = "tiiuae/falcon-7b-instruct"
        self.training_arguments = training_config

        self.training_dataset, self.validation_dataset = self.load_data()
        self.model, self.tokenizer, self.peft_config = self.load_model()

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Load training and validation datasets & returns them in preprocessed LLM-format dataset.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        """
        print(f"Loading QA datasets from {self.path_dataset}")
        logger.info(f"logger: Loading QA datasets from {self.path_dataset=}")

        training_dataset = data.FinanceDataset(
                                path_data = f"{self.path_dataset}/training_data.json",
                                template_name = self.llm_template_name,
                            ).preprocess_to_llm_format()

        validation_dataset = data.FinanceDataset(
                                path_data = f"{self.path_dataset}/testing_data.json",
                                template_name = self.llm_template_name,
                            ).preprocess_to_llm_format()
        
        logger.info(f"Training dataset size: {len(training_dataset)}")
        logger.info(f"Validation dataset size: {len(validation_dataset)}")

        return training_dataset, validation_dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        """
        Loads the model.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]: A tuple containing the model, tokenizer,
                and PeftConfig.
        """
        print(f"Loading model using {self.model_id}")
        logger.info(f"Loading model using {self.model_id=}")
        model, tokenizer, peft_config = models.load_model(
            pretrained_model_path=self.model_id,
            gradient_checkpointing=True,
            cache_dir=self.path_model_cache,
        )
        return model, tokenizer, peft_config

    def train_model(self) -> SFTTrainer:
        """
        Trains the model.

        Returns:
            SFTTrainer: The trained model.
        """
        print("Training model...")
        logger.info("Training model...")

        trainer = SFTTrainer(
            model = self.model,
            train_dataset = self.training_dataset,
            eval_dataset = self.validation_dataset, # mlabonne doesn't use this
            peft_config = self.peft_config,
            dataset_text_field = "prompt",
            max_seq_length = 1024, # mlabonne uses None
            tokenizer = self.tokenizer,
            args = self.training_arguments,
            packing = True,
            compute_metrics = self.compute_metrics,
            #callbacks=[BestModelToModelRegistryCallback(model_id=self.model_id)],
        )
        trainer.train()

        return trainer

    def compute_metrics(self, eval_pred: EvalPrediction):
        """
        Computes the perplexity metric.

        Args:
            eval_pred (EvalPrediction): The evaluation prediction.

        Returns:
            dict: A dictionary containing the perplexity metric.
        """
        perplexity = np.exp(eval_pred.predictions.mean()).item()
        return {"perplexity": perplexity}

# uncomment below to train locally, and then just run this script
""" 
from training_modules.config import training_arguments

path_dataset = "./dataset"
path_model_cache = "./model_cache" #if path_model_cache else None

training_api = TrainingAPI(
    training_config=training_arguments,
    path_dataset=path_dataset,
    path_model_cache=path_model_cache,
)
training_api.train_model()
"""