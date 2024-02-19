from pathlib import Path
from typing import Optional, Tuple

import torch as th
from peft import LoraConfig, PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def load_model(
    pretrained_model_path: str = "tiiuae/falcon-7b-instruct",
    #peft_pretrained_model_name_or_path: Optional[str] = None,
    #gradient_checkpointing: bool = True,
    cache_dir: Optional[Path] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
    """
    Function that builds a QLoRA LLM model based on the given HuggingFace name:
        1.   Create and prepare the bitsandbytes configuration for QLoRa's quantization
        2.   Download, load, and quantize on-the-fly Falcon-7b
        3.   Create and prepare the LoRa configuration
        4.   Load and configuration Falcon-7B's tokenizer

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model to use.
        peft_pretrained_model_name_or_path (Optional[str]): The name or path of the pretrained model to use
            for PeftModel.
        gradient_checkpointing (bool): Whether to use gradient checkpointing or not.
        cache_dir (Optional[Path]): The directory to cache the model in.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]: A tuple containing the built model, tokenizer,
            and PeftConfig.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=th.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        revision="main",
        quantization_config=bnb_config,
        load_in_4bit=True,
        device_map="auto", # {"": 0} to load entire model on GPU 0
        trust_remote_code=False,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        trust_remote_code=False,
        truncation=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        with th.no_grad():
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id    

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["query_key_value"],
    )

    return model, tokenizer, lora_config