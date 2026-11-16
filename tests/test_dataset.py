from pathlib import Path

import pytest

from scripts.validate_dataset import (
    DATASET,
    REQUIRED_SECTIONS,
    validate_content,
)


@pytest.fixture
def dataset_content():
    return DATASET.read_text(encoding="utf-8")


def test_dataset_exists():
    assert DATASET.exists()


def test_dataset_not_empty(dataset_content):
    assert dataset_content.strip()


def test_dataset_utf8_encoding():
    content = DATASET.read_text(encoding="utf-8")
    assert isinstance(content, str)


def test_dataset_has_main_heading(dataset_content):
    assert dataset_content.lstrip().startswith("# ")


def test_required_sections_exist(dataset_content):
    for section in REQUIRED_SECTIONS:
        assert section in dataset_content


def test_validate_content_returns_dict(dataset_content):
    result = validate_content(dataset_content)
    assert isinstance(result, dict)


def test_validation_checks_file_not_empty(dataset_content):
    result = validate_content(dataset_content)
    assert result["File not empty"] is True


def test_validation_checks_main_heading(dataset_content):
    result = validate_content(dataset_content)
    assert result["Has main heading"] is True


@pytest.mark.parametrize(
    "content, expected_not_empty, expected_heading",
    [
        ("", False, False),
        ("## Missing Main Heading\nSome content.", True, False),
        ("# Valid Dataset\n\n## Section\nContent.", True, True),
    ],
)
def test_invalid_dataset_detection(
    content,
    expected_not_empty,
    expected_heading,
):
    result = validate_content(content)

    assert result["File not empty"] is expected_not_empty
    assert result["Has main heading"] is expected_heading
