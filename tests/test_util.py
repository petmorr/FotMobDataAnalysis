import pandas as pd

from fotmob_analytics.util import md_escape, safe_csv_bytes


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
