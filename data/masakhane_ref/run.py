
from datasets import load_dataset

# Yoruba
yor = load_dataset("masakhane/mafand", "en-yor", split="train")

# Hausa
hau = load_dataset("masakhane/mafand", "en-hau", split="train")

# Igbo
ibo = load_dataset("masakhane/mafand", "en-ibo", split="train")

# See what's inside
print(yor[0])