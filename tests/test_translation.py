from benchmark.translation import missing_entities, protected_entities


def test_technical_entities_are_detected():
    text = "Edit `src/app.py` and run `pytest grader/test_task.py`."
    entities = protected_entities(text)
    assert "src/app.py" in entities
    assert "pytest grader/test_task.py" in entities


def test_missing_entity_is_reported():
    source = "Edit `src/app.py`."
    variant = "Edit the application file."
    assert "src/app.py" in missing_entities(source, variant)
