"""Tests to verify imports of power_spherical and TSW variants after pip install -e ."""

import pytest


def test_import_power_spherical():
    """Power spherical distributions should be importable."""
    from power_spherical import (
        PowerSpherical,
        HypersphericalUniform,
        MarginalTDistribution,
    )
    assert PowerSpherical is not None
    assert HypersphericalUniform is not None
    assert MarginalTDistribution is not None


def test_import_tsw_concurrent():
    """Concurrent-line TSW (TSW, DbTSW) should be importable."""
    from tree_sliced.tsw import TSW, DbTSW
    assert TSW is not None
    assert DbTSW is not None


def test_import_tssobolev_concurrent():
    """Concurrent-line TS-Sobolev (TSSobolev, SbTS) should be importable."""
    from tree_sliced.ts_sobolev import TSSobolev, SbTS
    assert TSSobolev is not None
    assert SbTS is not None


def test_import_tsw_chain():
    """Chain-structured TSW should be importable."""
    from tree_sliced.tsw_chain import TSWChain
    assert TSWChain is not None


def test_import_tssobolev_chain():
    """Chain-structured TS-Sobolev should be importable."""
    from tree_sliced.ts_sobolev_chain import TSSobolevChain, generate_trees_frames
    assert TSSobolevChain is not None
    assert callable(generate_trees_frames)


def test_import_tree_sliced_utils():
    """Tree-sliced utils (generate_trees_frames) should be importable."""
    from tree_sliced.utils import generate_trees_frames
    assert callable(generate_trees_frames)
