"""Create a new output file atomically without replacing an existing result."""
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


@contextmanager
def _new_output(path, suffix):
    path = Path(path)
    if path.suffix != suffix:
        raise ValueError(f"output file must end in {suffix}")
    if path.exists():
        raise ValueError(f"output already exists: {path}; choose a new filename")
    with tempfile.NamedTemporaryFile(dir=path.parent) as temp:
        yield temp
        temp.flush()
        os.link(temp.name, path)
