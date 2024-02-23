import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from training_modules.constants import Scope
from training_modules.prompt_templates import get_llm_template

# Deserialize data
@dataclass(frozen=True)
class DataSample:
    """
    A data sample for a question answering model.

    Attributes:
        user_context (str): The user's context for the question.
        news_context (str): The news context for the question.
        chat_history (str): The chat history for the question.
        question (str): The question to be answered.
        answer (str): The answer to the question.
    """

    user_context: str = field(repr=False)
    news_context: str = ""
    chat_history: str = ""
    question: str = ""
    answer: str = ""

class FinanceDataset:
    def __init__(self, path_data: Path, scope: Scope = Scope.TRAINING, template_name: str = "falcon"):
        """
        A class representing a finance dataset.

        Args:
            data_path (Path): The path to the data file.
            template (str, optional): The template to use for the dataset. Defaults to "falcon".
        """

        self.path_data = path_data
        self.scope = scope
        self.template_name = template_name

    def load_data(self, path_data: Path) -> List:
        """
        Loads the data from the specified path.

        Args:
            data_path (Path): The path to the data file.

        Returns:
            List[DataSample]: The loaded data.
        """

        # Load data from specified path
        with open(path_data) as f:
            data = json.load(f)

        return data

    def deserialize_data(self, data: List[dict]) -> List[DataSample]:
        """
        Deserializes the data.

        Args:
            data (List[dict]): The data to deserialize.

        Returns:
            List[DataSample]: The deserialized data.
        """

        if self.scope == Scope.TRAINING:
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                    answer=sample["response"],
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                )
                for sample in data
            ]
        
    def clean_data(self, samples: Dict[str, str]) -> Dict[str, str]:
        """
        Cleans the samples.

        Args:
            samples (Dict[str, str]): The samples to clean.

        Returns:
            Dict[str, str]: The cleaned samples.
        """

        for key, sample in samples.items():
            cleaned_sample = clean_extra_whitespace(sample)
            cleaned_sample = group_broken_paragraphs(cleaned_sample)

            samples[key] = cleaned_sample

        return samples


    def preprocess_to_llm_format(self) -> Dataset:
        """
        Preprocesses the data & returns a LLM-format dataset.

        Returns:
            Dataset: The LLM-format dataset.
        """
        raw_data = self.load_data(self.path_data)
        deserialized_data = self.deserialize_data(raw_data)
        data_as_dict = [asdict(sample) for sample in deserialized_data]
        dataset = Dataset.from_list(data_as_dict)

        template = get_llm_template(self.template_name)
        if self.scope == Scope.TRAINING:
            template_mapping_func = template.format_train
        else:
            template_mapping_func = template.format_infer

        dataset = dataset.map(self.clean_data)
        dataset = dataset.map(template_mapping_func, remove_columns=dataset.column_names)

        return dataset