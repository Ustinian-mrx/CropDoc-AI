from pathlib import Path

from config import PLANT_VILLAGE_DIR, NUM_CLASSES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(class_dir: Path) -> int:
    count = 0

    for file_path in class_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1

    return count


def main():
    print("Checking PlantVillage dataset...")
    print(f"Dataset path: {PLANT_VILLAGE_DIR}")

    if not PLANT_VILLAGE_DIR.exists():
        print("ERROR: PlantVillage dataset directory does not exist.")
        print("Please place the dataset at:")
        print(PLANT_VILLAGE_DIR)
        return

    class_dirs = [
        path for path in PLANT_VILLAGE_DIR.iterdir()
        if path.is_dir()
    ]

    class_dirs = sorted(class_dirs, key=lambda path: path.name)

    print(f"Found {len(class_dirs)} class folders.")

    total_images = 0

    for index, class_dir in enumerate(class_dirs, start=1):
        image_count = count_images(class_dir)
        total_images += image_count
        print(f"{index:02d}. {class_dir.name}: {image_count} images")

    print("-" * 40)
    print(f"Total images: {total_images}")

    if len(class_dirs) != NUM_CLASSES:
        print(
            f"WARNING: Expected {NUM_CLASSES} classes, "
            f"but found {len(class_dirs)}."
        )
    else:
        print("Class count looks good.")

    if total_images == 0:
        print("WARNING: No images found. Please check your dataset files.")
    else:
        print("Dataset check finished.")


if __name__ == "__main__":
    main()
