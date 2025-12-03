import os
import pandas as pd
import selfies as sf #1.0.4
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Config
#MODEL_NAME = "./chemgpt_finetuned"
MODEL_NAME = "ncfrey/ChemGPT-4.7M"
NUM_SAMPLES = 50          # generate this many molecules
MAX_LEN = 30
OUTPUT_FILE = "molecules.csv"

#FRAGMENT_SMILES = "c1ccccc1"
#FRAGMENT_SELFIES = sf.encoder(FRAGMENT_SMILES)

# Load pretrained ChemGPT 
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Generate molecules 
inputs = tokenizer("", return_tensors="pt")  # unconditional
#inputs = tokenizer(FRAGMENT_SELFIES, return_tensors="pt")  # Conditional generation 

outputs = model.generate(
    **inputs,
    max_length=MAX_LEN,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    top_p=0.95,
    temperature=0.9
)

valid_smiles = []

#converting selfies to smiles
for idx, o in enumerate(outputs, 1):
    selfies_str = tokenizer.decode(o, skip_special_tokens=True).strip()
    selfies_clean = selfies_str.replace(" ", "")

    print(f"\nAttempt {idx}: SELFIES = {selfies_clean}")

    try:
        smiles = sf.decoder(selfies_clean)
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print("Invalid SMILES (could not parse)")
            continue

        else:
            valid_smiles.append(smiles)

    except Exception as e:
        print(f"Error decoding or processing: {e}")

# Save results 
df = pd.DataFrame(valid_smiles)
df.to_csv(OUTPUT_FILE, index=False, header=False)



