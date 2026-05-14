def clean_text(value: str) -> str:
    return (
        value
        .replace("_", " ")
        .replace(",", ", ")
        .replace("  ", " ")
        .strip()
    )


def parse_class_name(class_name: str) -> dict:
    if "___" not in class_name:
        cleaned_name = clean_text(class_name)

        return {
            "class_name": class_name,
            "crop": "Unknown",
            "disease": cleaned_name,
            "is_healthy": cleaned_name.lower() == "healthy",
        }

    crop_raw, disease_raw = class_name.split("___", maxsplit=1)

    crop = clean_text(crop_raw)
    disease = clean_text(disease_raw)

    is_healthy = disease.lower() == "healthy"

    return {
        "class_name": class_name,
        "crop": crop,
        "disease": disease,
        "is_healthy": is_healthy,
    }
