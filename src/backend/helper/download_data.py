from datasets import load_dataset
from src.backend.utilities.code_util import project_root
import os

# Load the dataset split
dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k", split="train")

# sampled = dataset.select(range(10))

# Save to JSON
dataset.to_json(os.path.join(project_root,"data/customer_support_dataset_unprocessed.json"), orient="records", lines=True)




