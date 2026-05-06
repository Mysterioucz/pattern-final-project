"""Pytest configuration: add repo root to sys.path so that
'from projects.src.data.sampler import ...' works without installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
