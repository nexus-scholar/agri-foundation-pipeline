"""Label normalization helpers shared across datasets."""
from __future__ import annotations

from functools import lru_cache

CROP_KEYWORDS = {
    "tomato": "tomato",
    "tom": "tomato",
    "potato": "potato",
    "pepper": "pepper",
    "bell_pepper": "pepper",
    "bell pepper": "pepper",
    "apple": "apple",
    "corn": "corn",
    "grape": "grape",
    "peach": "peach",
    "cherry": "cherry",
    "blueberry": "blueberry",
    "strawberry": "strawberry",
    "raspberry": "raspberry",
    "soyabean": "soybean",
    "soybean": "soybean",
    "squash": "squash",
}


@lru_cache(maxsize=None)
def normalize_label(folder_name: str) -> tuple[str, str, str]:
    """Convert folder name to standardized label format: crop_disease."""
    name = folder_name.replace("__", "_").replace("  ", " ")
    name_lower = name.lower()

    crop = "unknown"
    for keyword, crop_name in CROP_KEYWORDS.items():
        if keyword in name_lower:
            crop = crop_name
            break

    disease_part = name_lower
    for keyword in CROP_KEYWORDS.keys():
        disease_part = disease_part.replace(keyword, "")

    for word in ["leaf", "leaves", "__", "  "]:
        disease_part = disease_part.replace(word, " ")

    disease_part = "_".join(disease_part.split()).strip("_")
    if not disease_part or disease_part in {"healthy", "normal"}:
        disease_part = "healthy"

    label = f"{crop}_{disease_part}" if crop != "unknown" else disease_part
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_"), crop, disease_part

