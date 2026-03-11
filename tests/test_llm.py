from __future__ import annotations

from squashvid.pipeline.llm import _coerce_item_text, _coerce_text_list


def test_coerce_item_text_extracts_preferred_field_from_python_dict_string() -> None:
    raw = "{'drill': 'Backhand length ladder', 'focus': 'deep targets'}"
    assert _coerce_item_text(raw) == "Backhand length ladder"


def test_coerce_text_list_flattens_structured_items() -> None:
    values = [
        {"title": "Ghosting set", "description": "Recover to T after each corner"},
        {"focus": "Backhand depth under pressure"},
    ]
    flattened = _coerce_text_list(values, limit=10)
    assert flattened == ["Ghosting set", "Backhand depth under pressure"]
