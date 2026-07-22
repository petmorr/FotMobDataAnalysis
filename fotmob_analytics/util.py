"""Small shared utilities."""

from __future__ import annotations

import pandas as pd

# Leading characters that spreadsheet apps interpret as formulas. Exported
# CSVs are opened in Excel/Sheets, so cells starting with these are escaped
# to prevent CSV/formula injection (OWASP: CSV Injection).
_FORMULA_PREFIXES = ("=", "+", "@", "\t", "\r")


def _sanitize_cell(value: object) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def safe_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to CSV with formula-injection escaping applied
    to string cells (numeric columns are untouched)."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(_sanitize_cell)
    return out.to_csv(index=False).encode()


_MD_SPECIALS = "\\`*_{}[]()#+-.!$<>|"


def md_escape(text: object) -> str:
    """Escape markdown control characters in externally sourced text (e.g.
    player names from the API) before interpolating into ``st.markdown``."""
    result = []
    for ch in str(text):
        if ch in _MD_SPECIALS:
            result.append("\\" + ch)
        else:
            result.append(ch)
    return "".join(result)
