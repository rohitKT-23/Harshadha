from datasets import load_dataset
import pandas as pd

# Step 1: Load the dataset
dataset = load_dataset("pkufool/libriheavy_long", split="train[:500]")  # Sample 500 for quick view

# Step 2: Convert to Pandas DataFrame
df = dataset.to_pandas()

# Step 3: Save to CSV
df.to_csv("libriheavy_sample.csv", index=False)

print("CSV saved as libriheavy_sample.csv")
