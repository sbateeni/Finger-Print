from features.minutiae_taxonomy import (
    POSTER_36_KEY,
    POSTER_36_SUMMARY,
    normalize_minutiae_type,
)


def test_poster_36_counts():
    assert len(POSTER_36_KEY) == 36
    assert POSTER_36_SUMMARY["ending_ridge"] == 12
    assert POSTER_36_SUMMARY["bifurcation"] == 23
    assert POSTER_36_SUMMARY["island"] == 1


def test_type_aliases():
    assert normalize_minutiae_type("Ending Ridge") == "endpoint"
    assert normalize_minutiae_type("island") == "island"
