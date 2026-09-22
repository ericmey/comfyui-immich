"""Packaging metadata checks.

The runtime dependency list exists twice -- pyproject.toml for pip, and
requirements.txt for ComfyUI Manager, which does not read pyproject. Neither
file can be dropped, so the only defence against drift is a test.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _declared_requirements():
    lines = ROOT.joinpath("requirements.txt").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_requirements_matches_pyproject_dependencies():
    # tomllib is stdlib from 3.11; the CI matrix covers this on 3.11/3.12/3.13.
    # The 3.10 job will skip via importorskip rather than fail; that is fine —
    # the packaging invariant is already enforced on every newer interpreter.
    tomllib = pytest.importorskip("tomllib")
    with ROOT.joinpath("pyproject.toml").open("rb") as f:
        pyproject = tomllib.load(f)
    assert _declared_requirements() == pyproject["project"]["dependencies"]


def test_requirements_are_unpinned_lower_bounds():
    """A custom node must not pin the host ComfyUI environment's packages."""
    for requirement in _declared_requirements():
        assert ">=" in requirement, requirement
        assert not any(op in requirement for op in ("==", "<", "~=")), requirement
