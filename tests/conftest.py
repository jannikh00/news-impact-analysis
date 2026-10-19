import importlib.util
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))


def load_script(filename):
    """Import a numbered pipeline script (e.g. 1_calculate_impacts_values.py) as a module."""
    spec = importlib.util.spec_from_file_location(Path(filename).stem.lstrip("0123456789_"), PIPELINE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
