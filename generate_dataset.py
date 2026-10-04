import random
import pandas as pd
from pathlib import Path

random.seed(42)

# DNA bases
BASES = "ACGT"

# Gene-family patterns
FAMILIES = {
    "Family_A": ["ATGCGT", "GATTACA", "CGTACG"],
    "Family_B": ["TTAACG", "CCGTTA", "AACCGG"],
    "Family_C": ["GGATCC", "TGCATG", "GCGTGC"],
    "Family_D": ["ATATGC", "CGCGTA", "GGTACC"],
}

def generate_sequence(patterns, length=100):
    sequence = ""

    while len(sequence) < length:
        sequence += random.choice(patterns)

        # Add random DNA bases
        if random.random() < 0.5:
            sequence += random.choice(BASES)

    return sequence[:length]


rows = []

# Generate 500 samples for each family
for family, patterns in FAMILIES.items():
    for _ in range(500):
        sequence = generate_sequence(patterns)

        rows.append({
            "sequence": sequence,
            "family": family
        })

# Shuffle dataset
random.shuffle(rows)

df = pd.DataFrame(rows)

# Save dataset
output_path = Path(__file__).parent.parent / "data" / "dna_dataset.csv"
df.to_csv(output_path, index=False)

print("Dataset created successfully!")
print(f"Total sequences: {len(df)}")
print(f"Saved to: {output_path}")
print("\nClass distribution:")
print(df["family"].value_counts())