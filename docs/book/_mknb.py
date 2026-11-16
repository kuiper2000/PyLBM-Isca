"""Tiny helper: build a .ipynb from a list of (kind, source) cells.

Usage (from another generator script):
    from _mknb import write_nb
    write_nb("03-steady-response.ipynb", [("md", "# title"), ("code", "1+1")])

Notebooks are written WITHOUT outputs; `jupyter-book build --execute` runs them.
"""
import json
import os


def _cell(kind, src):
    src = src.rstrip("\n")
    lines = [l + "\n" for l in src.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


def write_nb(path, cells, kernel="python3"):
    nb = {
        "cells": [_cell(k, s) for k, s in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": kernel},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, path)
    with open(out, "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", out)


# Standard preamble every computational chapter uses: find the repo root, put
# both it and the book dir on sys.path, pick a backend, build the T21 grid.
PREAMBLE = '''\
import os, sys, warnings
warnings.filterwarnings("ignore")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
root = os.path.abspath("")
while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, "pylbm_isca")):
    root = os.path.dirname(root)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "docs", "book"))
import numpy as np
import matplotlib.pyplot as plt
from pylbm_isca import Config, LBM, Spectral, VerticalGrid
from pylbm_isca.backend import asnumpy, gpu_available
import lbmtools as lt

DEVICE = "gpu" if gpu_available() else "cpu"
sp = Spectral(nlon=64, nlat=32, nmax=21, device=DEVICE)   # T21 L20, runs offline
vg = VerticalGrid(device=DEVICE)
lon, lat, sig = sp.lon_deg, sp.lat_deg, asnumpy(vg.sig)
print(f"PyLBM-Isca  T{sp.nmax}  {sp.nlon}x{sp.nlat}  L{vg.kmax}  backend={DEVICE}")'''
