"""Execute a notebook in place, embedding outputs. Usage: python _run.py nb.ipynb"""
import os, sys, time
import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
os.environ.setdefault("OMP_NUM_THREADS", "4")
t0 = time.time()
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=900, kernel_name="python3",
                        resources={"metadata": {"path": os.path.dirname(path) or "."}})
client.execute()
nbformat.write(nb, path)
print(f"OK {path}  ({time.time()-t0:.0f}s)")
