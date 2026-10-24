"""Compatibility module: the basis now lives in maxent_upgrade.basis.

Kept so that scripts written against 1.0 (`from maxent_upgrade.dy_method import cheb, umap`)
keep working.  The earlier exploratory Drell-Yan driver that used to live here is not part of
the package.
"""
from .basis import cheb, umap, BW_M, BW_G  # noqa: F401
