import json
import os
from pathlib import Path
import datasets

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
Custom Non-Factoid Question Dataset is derived from a specific collection using machine learning techniques. \
The dataset contains questions along with their corresponding answers. Each question includes various details \
such as the date, store information, transactions recorded, crude oil price, product category, and specific conditions. \
The output text indicates the sales quantity for each scenario.
"""

class CustomDataset(datasets.GeneratorBasedBuilder):
    """Custom Non-Factoid Question Dataset"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {   "id": datasets.Value("string"),
                    "input_text": datasets.Value("string"),
                    "output_text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        # Specify the path to the dataset file
        filepath = "/content/drive/Shareddrives/dataset/clients/research/formatted_data.json"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for idx, example in enumerate(data):
                    yield idx, {
                        "input_text": example["input_text"],
                        "output_text": example["output_text"],
                    }
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file '{filepath}' not found.")
