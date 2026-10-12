"""Common utilities."""

import datetime
import pathlib


def package_root() -> pathlib.Path:
    """Root of the repository."""
    return pathlib.Path(__file__).resolve().parent.parent


def resolve_path(path: str | pathlib.Path) -> pathlib.Path:
    """Resolve `path` as either absolute or relative to the package root."""
    path = pathlib.Path(path)
    return path if path.is_absolute() else package_root() / path


def timestamp() -> str:
    """The current time as `YYYY-MM-DD_HH-MM-SS`."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_time(seconds: float) -> str:
    """Convert `seconds` to a duration of time with an easier-to-interpret unit."""
    if seconds < 1.0:
        return f"{seconds * 1000:.2f} ms"
    if seconds > 3600.0:
        return f"{int(seconds // 3600)} hr {int((seconds % 3600) // 60)} min"
    if seconds > 60.0:
        return f"{int(seconds // 60)} min {seconds % 60:.2f} s"
    return f"{seconds:.2f} s"
