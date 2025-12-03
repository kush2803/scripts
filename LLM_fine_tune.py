# pip install transformers torch pandas scikit-learn
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
	AutoTokenizer,
	AutoModelForCausalLM,
	LineByLineTextDataset,
	TrainingArguments,
	Trainer,
	DataCollatorForLanguageModeling
)

# Define the model name you want to fine-tune
model_name = "ncfrey/ChemGPT-4.7M"

# Load the pre-trained model and tokenizer
print("Loading pre-trained model and tokenizer...")
try:
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(model_name)
	
	tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	model.resize_token_embeddings(len(tokenizer)) # Important to resize the model's embeddings
	
except Exception as e:
#	print(f"Error loading model or tokenizer: {e} ")
	exit()

# Read and split SELFIES data from the CSV file ---
csv_file_path = "All_ligands.csv"
selfies_column = "Ligand_Selfies"
print(f"Reading and splitting SELFIES data from '{csv_file_path}'...")
try:
	df = pd.read_csv(csv_file_path)
	selfies_data = df[selfies_column].tolist()
	if not selfies_data:
		print("Error: The column 'Ligand_Selfies' is empty or contains no valid data.")
		exit()
except FileNotFoundError:
	print(f"Error: The file '{csv_file_path}' was not found.")
	exit()
except KeyError:
	print(f"Error: The column '{selfies_column}' was not found in the CSV file.")
	exit()

# Split the data into a training set and a validation set
train_selfies, val_selfies = train_test_split(selfies_data, test_size=0.1, random_state=42)

# Prepare the datasets and data collator
train_txt_file = "train_selfies_data.txt"
val_txt_file = "val_selfies_data.txt"

print("Saving training and validation data to temporary text files...")
with open(train_txt_file, 'w') as f:
	for selfies in train_selfies:
		f.write(f"{selfies}\n")
		

with open(val_txt_file, 'w') as f:
	for selfies in val_selfies:
		f.write(f"{selfies}\n")

# Load datasets
train_dataset = LineByLineTextDataset(
	tokenizer=tokenizer,
	file_path=train_txt_file,
	block_size=128
)
eval_dataset = LineByLineTextDataset(
	tokenizer=tokenizer,
	file_path=val_txt_file,
	block_size=128
)

data_collator = DataCollatorForLanguageModeling(
	tokenizer=tokenizer,
	mlm=False #ignores padding tokens
)

# Define training parameters and fine-tune the model
training_args = TrainingArguments(
	output_dir="./chemgpt_finetuned",
	overwrite_output_dir=True,
	num_train_epochs=500,
	per_device_train_batch_size=40,
	save_steps=1000,
	save_total_limit=3,
	eval_strategy="steps", #evaluate performance after "steps"
	eval_steps=500,
	logging_steps=5
)

trainer = Trainer(
	model=model,
	args=training_args,
	data_collator=data_collator,
	train_dataset=train_dataset,
	eval_dataset=eval_dataset
)

print("Starting fine-tuning process...")
# Capture the training output
training_output = trainer.train()

# Save the fine-tuned model and the log history
output_directory = "./chemgpt_finetuned"
print(f"\nFine-tuning complete. Saving model and tokenizer to '{output_directory}'...")
trainer.save_model(output_directory)
tokenizer.save_pretrained(output_directory)

# Explicitly save the log history to a JSON file
log_file_path = "training_log.json"
print(f"Saving loss information to '{log_file_path}'...")
with open(log_file_path, 'w') as f:
	json.dump(trainer.state.log_history, f, indent=4)

print("Process finished successfully.")