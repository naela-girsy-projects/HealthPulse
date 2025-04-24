import os
import shutil
import random
from pathlib import Path

# Create directories for our classification task
base_dir = Path("/Users/naelamacbookair/desktop backup/self projects/HealthPulse")
processed_dir = base_dir / "data/processed/indiana"
organized_dir = base_dir / "data/organized/indiana"

# Create normal and pneumonia directories
normal_dir = organized_dir / "normal"
pneumonia_dir = organized_dir / "pneumonia"

normal_dir.mkdir(parents=True, exist_ok=True)
pneumonia_dir.mkdir(parents=True, exist_ok=True)

# For demonstration purposes, randomly assign images to classes
# In a real scenario, you would use actual labels or annotations
all_images = list(processed_dir.glob("*.png"))
random.seed(42)  # For reproducibility
random.shuffle(all_images)

# Assign 70% as normal, 30% as pneumonia (typical class distribution)
split_index = int(len(all_images) * 0.7)
normal_images = all_images[:split_index]
pneumonia_images = all_images[split_index:]

# Copy images to appropriate directories
for img in normal_images:
    shutil.copy(img, normal_dir / img.name)

for img in pneumonia_images:
    shutil.copy(img, pneumonia_dir / img.name)

print(f"Organized dataset: {len(normal_images)} normal images, {len(pneumonia_images)} pneumonia images")