from pipeline.label_utils import normalize_label


def test_normalize_label_basic():
    label, crop, disease = normalize_label("Tomato Early blight leaf")
    assert label == "tomato_early_blight"
    assert crop == "tomato"
    assert disease == "early_blight"


def test_normalize_label_handles_unknown():
    label, crop, disease = normalize_label("mystery class")
    assert crop == "unknown"
    assert label == "mystery_class"


def test_normalize_label_healthy():
    label, crop, disease = normalize_label("Tomato healthy leaf")
    assert label == "tomato_healthy"
    assert disease == "healthy"


def test_normalize_label_cached():
    first = normalize_label("Tomato Early blight leaf")
    second = normalize_label("Tomato Early blight leaf")
    assert first == second
