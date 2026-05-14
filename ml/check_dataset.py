from pathlib import Path

from config import PLANT_VILLAGE_DIR, NUM_CLASSES


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(class_dir: Path) -> int:
    count = 0

    for file_path in class_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            count += 1

    return count


def check_split(split_name: str):
    split_dir = PLANT_VILLAGE_DIR / split_name

    print(f"\nChecking {split_name} split...")
    print(f"Path: {split_dir}")

    if not split_dir.exists():
        print(f"ERROR: {split_name} directory does not exist.")
        return False

    class_dirs = [
        path for path in split_dir.iterdir()
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
    print(f"{split_name} total images: {total_images}")

    if len(class_dirs) != NUM_CLASSES:
        print(
            f"WARNING: Expected {NUM_CLASSES} classes, "
            f"but found {len(class_dirs)}."
        )
        return False

    if total_images == 0:
        print("WARNING: No images found.")
        return False

    print(f"{split_name} split looks good.")
    return True


def main():
    print("Checking PlantVillage dataset...")
    print(f"Dataset root: {PLANT_VILLAGE_DIR}")

    if not PLANT_VILLAGE_DIR.exists():
        print("ERROR: PlantVillage dataset directory does not exist.")
        print("Please place the dataset at:")
        print(PLANT_VILLAGE_DIR)
        return

    train_ok = check_split("train")
    val_ok = check_split("val")

    print("\nSummary")
    print("-" * 40)

    if train_ok and val_ok:
        print("Dataset check finished. Train and val splits look good.")
    else:
        print("Dataset check finished with warnings. Please inspect the messages above.")


if __name__ == "__main__":
    main()
