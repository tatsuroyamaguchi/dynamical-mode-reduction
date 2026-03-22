"""
dmr
===
Dynamical Mode Reduction (DMR): a novel dimensionality reduction method
inspired by classical mechanics.

Quick start
-----------
>>> from dmr import DynamicalModeReduction
>>> dmr = DynamicalModeReduction(n_components=2)
>>> embedding = dmr.fit_transform(X)
"""

from .core import DynamicalModeReduction

__version__ = "0.1.0"
__author__ = "DMR authors"
__all__ = ["DynamicalModeReduction"]
