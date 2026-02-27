from quran_verse_id.normalizer import normalize_phonetic, normalize_arabic, strip_arabic_diacritics

def test_normalize_phonetic_basic():
    assert normalize_phonetic("  SALAAM!!  ") == "salaam"
    assert normalize_phonetic("a   b   c") == "a b c"

def test_normalize_arabic_diacritics():
    # بِسْمِ -> بسم
    assert strip_arabic_diacritics("بِسْمِ") == "بسم"
    assert normalize_arabic("إِنَّ") == "ان"
