import json
from pathlib import Path

from quran_verse_id.phonetic_db import PhoneticQuranDB


def test_phonetic_db_load_save(tmp_path: Path):
    entries = [
        {"surah": 1, "ayah": 1, "phonetic": "bismi llah"},
        {"surah": 1, "ayah": 2, "phonetic": "al hamdu"},
    ]
    p = tmp_path / "db.json"
    p.write_text(json.dumps(entries), encoding="utf-8")

    db = PhoneticQuranDB.load_json(p)
    assert len(db) == 2
    assert db.get(1, 1).phonetic == "bismi llah"
