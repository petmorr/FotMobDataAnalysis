import warnings

import pandas as pd

from fotmob_analytics.util import concat_frames, md_escape, safe_csv_bytes


class TestSafeCsv:
    def test_escapes_formula_prefixes(self):
        df = pd.DataFrame({"name": ["=cmd()", "+SUM(A1)", "@x", "normal"],
                           "value": [1, 2, 3, 4]})
        text = safe_csv_bytes(df).decode()
        assert "'=cmd()" in text
        assert "'+SUM(A1)" in text
        assert "'@x" in text
        assert ",normal" in text or "\nnormal" in text

    def test_numeric_columns_untouched(self):
        df = pd.DataFrame({"v": [-1.5, 2.0]})
        text = safe_csv_bytes(df).decode()
        assert "-1.5" in text and "'" not in text

    def test_original_frame_not_mutated(self):
        df = pd.DataFrame({"name": ["=x"]})
        safe_csv_bytes(df)
        assert df["name"].iloc[0] == "=x"


class TestMdEscape:
    def test_escapes_markdown_specials(self):
        assert md_escape("a*b_c[d]") == r"a\*b\_c\[d\]"
        assert md_escape("<img>") == r"\<img\>"

    def test_plain_text_unchanged(self):
        assert md_escape("Erling Haaland") == "Erling Haaland"

    def test_non_string_input(self):
        assert md_escape(42) == "42"


class TestConcatFrames:
    def test_skips_empty_and_all_na(self):
        a = pd.DataFrame({"id": [1, 2], "goals": [1.0, 2.0]})
        empty = pd.DataFrame()
        all_na = pd.DataFrame({"id": [pd.NA, pd.NA], "goals": [pd.NA, pd.NA]})
        b = pd.DataFrame({"id": [3], "assists": [1.0]})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = concat_frames([a, empty, all_na, b])
        assert not any(issubclass(w.category, FutureWarning) for w in caught)
        assert list(out["id"]) == [1, 2, 3]
        assert "goals" in out.columns and "assists" in out.columns

    def test_all_empty_returns_empty(self):
        out = concat_frames([pd.DataFrame(), pd.DataFrame({"a": [pd.NA]})])
        assert out.empty

    def test_single_frame_copy(self):
        a = pd.DataFrame({"id": [1], "x": [2.0]})
        out = concat_frames([a])
        assert list(out["id"]) == [1]
        out.loc[0, "id"] = 99
        assert a.loc[0, "id"] == 1

