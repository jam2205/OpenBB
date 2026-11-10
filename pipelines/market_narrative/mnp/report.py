"""Render the regime-specific template to Markdown (and optional HTML)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

_TPL = Path(__file__).parent / "templates"
_CSS = ("<style>.mnp table{border-collapse:collapse;margin:.5em 0 1em}.mnp th,.mnp td{border:1px solid #8884;"
        "padding:3px 9px;font-size:13px}.mnp th{background:#8882}.mnp td[style*=right]{font-variant-numeric:tabular-nums}</style>")


def _cell(v, whole: bool) -> str:
    """One number as text: whole-number columns and large values get no decimals."""
    if v is None or pd.isna(v):
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    v = float(v)
    s = f"{v:,.0f}" if whole or abs(v) >= 10_000 else f"{v:,.2f}"
    return s.lstrip("-") if s.strip("-0.,") == "" else s   # no "-0.00"


def _text(v) -> str:
    """Text safe inside a markdown table cell ("|" splits cells, "$" starts maths in notebooks)."""
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return ""
    return str(v).replace("|", " / ").replace("$", "&#36;").replace("\n", " ")


def _md(df: pd.DataFrame | None) -> str:
    """A frame as a markdown table, formatted here so no renderer re-parses the numbers.

    * a default 0..n index is dropped (it is a row counter, not data)
    * a numeric column holding only whole numbers prints without decimals (counts, days)
    * missing values print as blanks
    * blank lines surround the table: without one above, a caption line directly over the table
      is read as its header row and the "|:---|" separator shows up as data
    """
    if df is None or df.empty:
        return "\n_no data_\n"
    df = df.copy()
    show_index = not (isinstance(df.index, pd.RangeIndex) and df.index.name is None)
    align = ["left"] if show_index else []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            df[c] = col.map(lambda v: "yes" if v else "")
            align.append("left")
        elif pd.api.types.is_numeric_dtype(col):
            vals = col.dropna().astype(float)
            whole = bool(len(vals)) and bool((vals == vals.round()).all())
            df[c] = col.map(lambda v, w=whole: _cell(v, w))
            align.append("right")
        else:
            df[c] = col.map(_text)
            align.append("left")
    if show_index:
        df.index = df.index.map(_text)
    return "\n" + df.to_markdown(index=show_index, disable_numparse=True, colalign=align) + "\n"


def _plain(s) -> str:
    """Free text (headlines, event names) for use outside tables."""
    return _text(s).replace("*", "&#42;").replace("_", "&#95;").replace("`", "'")


def render(ctx: dict, template: str, out_dir: str | Path, html: bool = True) -> Path:
    env = Environment(loader=FileSystemLoader(_TPL), trim_blocks=True, lstrip_blocks=True)
    env.filters["md"] = _md
    env.filters["plain"] = _plain
    env.filters["pct"] = lambda x: "n/a" if pd.isna(x) else f"{x:+.2%}"
    text = env.get_template(template).render(**ctx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"report_{ctx['asof']}_{ctx['regime']['name']}.md"
    md_path.write_text(text)
    if html:
        try:
            import markdown

            body = markdown.markdown(text, extensions=["tables"])
            md_path.with_suffix(".html").write_text(f"{_CSS}<div class='mnp'>{body}</div>")
        except ImportError:
            pass
    return md_path


def show(path: str | Path):
    """Display a report in a notebook. Uses the HTML file, because a notebook's Markdown view
    typesets anything between two dollar signs as a formula and garbles prices in headlines.
    """
    from IPython.display import HTML, Markdown

    h = Path(path).with_suffix(".html")
    return HTML(h.read_text()) if h.exists() else Markdown(Path(path).read_text())


def quiet_progress() -> None:
    """Stop model download / weight-loading progress bars from printing into the report cell."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass
    try:
        from transformers.utils import logging as tlog

        tlog.disable_progress_bar()
        tlog.set_verbosity_error()
    except Exception:  # noqa: BLE001
        pass
