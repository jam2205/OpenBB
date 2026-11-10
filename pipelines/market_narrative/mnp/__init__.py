"""Market Narrative Pipeline: NLP + cross-asset + regime-aware reports on top of OpenBB."""

__all__ = ["run"]


def run(*args, **kwargs):
    from .run import run as _run

    return _run(*args, **kwargs)
