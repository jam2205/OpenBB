"""Render the regime-specific template to Markdown (and optional HTML)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

_TPL = Path(__file__).parent / "templates"


def _md(df: pd.DataFrame | None, floatfmt: str = ".3f") -> str:
    if df is None or df.empty:
        return "_no data_"
    return df.to_markdown(floatfmt=floatfmt)


def render(ctx: dict, template: str, out_dir: str | Path, html: bool = True) -> Path:
    env = Environment(loader=FileSystemLoader(_TPL), trim_blocks=True, lstrip_blocks=True)
    env.filters["md"] = _md
    env.filters["pct"] = lambda x: "n/a" if pd.isna(x) else f"{x:+.2%}"
    text = env.get_template(template).render(**ctx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"report_{ctx['asof']}_{ctx['regime']['name']}.md"
    md_path.write_text(text)
    if html:
        try:
            import markdown

            md_path.with_suffix(".html").write_text(markdown.markdown(text, extensions=["tables"]))
        except ImportError:
            pass
    return md_path
