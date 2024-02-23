from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

from transformers import TrainingArguments

@dataclass
class TrainingConfig:
    """
    Training configuration class used to load and store the training configuration.

    Attributes:
    -----------
    training : TrainingArguments
        The training arguments used for training the model.
    model : Dict[str, Any]
        The dictionary containing the model configuration.
    """

    training: TrainingArguments
    model: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path, output_dir: Path) -> "TrainingConfig":
        """
        Load a configuration file from the given path.

        Parameters:
        -----------
        config_path : Path
            The path to the configuration file.
        output_dir : Path
            The path to the output directory.

        Returns:
        --------
        TrainingConfig
            The training configuration object.
        """

        # Load yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        config["training"] = cls._dict_to_training_arguments(
            training_config=config["training"], output_dir=output_dir
        )

        return cls(**config)

    @classmethod
    def _dict_to_training_arguments(
        cls, training_config: dict, output_dir: Path
    ) -> TrainingArguments:
        """
        Build a TrainingArguments object from a configuration dictionary.

        Parameters:
        -----------
        training_config : dict
            The dictionary containing the training configuration.
        output_dir : Path
            The path to the output directory.

        Returns:
        --------
        TrainingArguments
            The training arguments object.
        """

        return TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1, # Batch size per GPU for training
    per_device_eval_batch_size=1, # Batch size per GPU for evaluation
    gradient_accumulation_steps=2, # Number of update steps to accumulate the gradients for
    optim="paged_adamw_32bit", # Optimizer to use
    save_steps=5, # Save checkpoint every X updates steps
    logging_steps=5, # Log every X updates steps
    learning_rate=2e-4, # Initial learning rate (AdamW optimizer)
    weight_decay=0.001, # Weight decay to apply to all layers except bias/LayerNorm weights
    fp16=True, # Enable fp16/bf16 training 
    #bf16=bf16, #(set bf16 to True with an A100)
    max_grad_norm=0.3, # # Maximum gradient normal (gradient clipping)
    max_steps=-1, # Number of training steps (overrides num_train_epochs)
    warmup_ratio=0.03, # Ratio of steps for a linear warmup (from 0 to learning rate)
    #group_by_length=True, # Group sequences into batches with same length. Saves memory and speeds up training considerably
    lr_scheduler_type="constant", # Learning rate schedule
    evaluation_strategy="steps",
    eval_steps=5,
    report_to="comet_ml", # or W&B / comet_ml
    seed=42,
    load_best_model_at_end=True
)
