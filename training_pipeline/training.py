from pathlib import Path
from typing import Optional, Tuple
from datasets import Dataset

from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

from training_modules import data, models

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
        training_dataset = data.FinanceDataset(
                                path_data = self.path_dataset / "training_data.json",
                                template_name = self.llm_template_name,
                            ).preprocess_to_llm_format()

        validation_dataset = data.FinanceDataset(
                                path_data = self.path_dataset / "testing_data.json",
                                template_name = self.llm_template_name,
                            ).preprocess_to_llm_format()
        
        return training_dataset, validation_dataset

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        """
        Loads the model.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]: A tuple containing the model, tokenizer,
                and PeftConfig.
        """

        model, tokenizer, peft_config = models.load_model(
            pretrained_model_path=self.model_id,
            #gradient_checkpointing=True,
            cache_dir=self.path_model_cache,
        )
        return model, tokenizer, peft_config

    def train_model(self) -> SFTTrainer:
        """
        Trains the model.

        Returns:
            SFTTrainer: The trained model.
        """

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
            #compute_metrics = ,
            #callbacks=[BestModelToModelRegistryCallback(model_id=self._model_id)],
        )
        trainer.train()

        return trainer

from training_modules.config import training_arguments

output_dir = './output'
path_dataset = Path.cwd().parent / "dataset"
path_model_cache = "./model_cache" #if path_model_cache else None

training_api = TrainingAPI(
    training_config=training_arguments,
    path_dataset=path_dataset,
    path_model_cache=path_model_cache,
)
training_api.train_model()

# Metrics?