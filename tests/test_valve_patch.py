from dota2tuned.clients.valve import flatten_patch_notes


def test_flatten_patch_notes_extracts_general_and_hero_notes():
    payload = {
        "patch_number": "7.41d",
        "patch_timestamp": 1780556400,
        "general_notes": [{"title": "Mechanics", "generic": [{"note": "A", "indent_level": 1}]}],
        "heroes": [
            {
                "hero_id": 1,
                "hero_notes": [{"note": "B"}],
                "abilities": [{"ability_id": 5004, "ability_notes": [{"note": "C"}]}],
            }
        ],
        "items": [{"ability_id": 104, "ability_notes": [{"note": "D"}]}],
    }
    rows = flatten_patch_notes(payload)
    assert [row["text"] for row in rows] == ["A", "D", "B", "C"]
    assert rows[0]["patch"] == "7.41d"
    assert rows[-1]["subject_type"] == "ability"
