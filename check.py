import os

base_dir = "Data/train"
for folder in sorted(os.listdir(base_dir)):
    path = os.path.join(base_dir, folder)
    if os.path.isdir(path):
        images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
        print(f"{folder}: {len(images)} images")
