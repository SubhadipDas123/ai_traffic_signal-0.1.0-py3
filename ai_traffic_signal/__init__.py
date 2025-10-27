"""ai_traffic_signal

Lightweight package init. Avoid importing heavy optional dependencies at import time so
that the package can be imported in environments without all runtime deps.
"""
__all__ = ["__version__", "load_module"]

__version__ = "0.1.0"

def load_module(name: str):
    """Dynamically import a submodule.

    Example: from ai_traffic_signal import load_module
             rrt = load_module('rrt_dijkstra')
             rrt.rrt_star_algorithm(...)
    """
    import importlib
    return importlib.import_module(f"ai_traffic_signal.{name}")
