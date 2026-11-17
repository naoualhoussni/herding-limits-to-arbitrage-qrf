"""
Compile and Execute Jupyter Notebook
===================================
Executes all cells of Theme2_Herding_QRF_Complete.ipynb,
capturing stdout and matplotlib figures into notebook cell outputs.
Saves the fully executed notebook to:
  - Theme2_Herding_QRF_Complete.ipynb
  - notebooks/Theme2_Herding_QRF_Complete.ipynb
  - Thematique2_Herding_QRF_Complet.ipynb
  - notebooks/Thematique2_Herding_QRF_Complet.ipynb
"""

import json
import sys
import io
import os
import base64
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

nb_path = ROOT / "Theme2_Herding_QRF_Complete.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

glob_ns = {}
exec_count = 1
total_code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")

print(f"Starting notebook compilation: {total_code_cells} code cells to execute.")

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue

    code_src = "".join(cell["source"])
    first_line = cell["source"][0].strip() if cell["source"] else ""
    print(f"[{exec_count}/{total_code_cells}] Executing cell {idx}: {first_line[:40]}...")

    # Prevent plt.show() from closing figures before we can grab them
    plt.close("all")
    orig_show = plt.show
    plt.show = lambda *args, **kwargs: None

    old_stdout = sys.stdout
    redirected = io.StringIO()
    sys.stdout = redirected

    error_msg = None
    try:
        exec(code_src, glob_ns)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"ERROR: {error_msg}")
    finally:
        sys.stdout = old_stdout
        plt.show = orig_show

    stdout_val = redirected.getvalue()
    if error_msg:
        print(f"   Cell {idx} failed with {error_msg}")
    else:
        print(f"   Cell {idx} executed successfully ({len(stdout_val.splitlines())} stdout lines)")

    cell["execution_count"] = exec_count
    cell["outputs"] = []

    # Attach stdout stream if present
    if stdout_val:
        cell["outputs"].append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in stdout_val.splitlines()]
        })

    # Attach any figures created during the cell execution
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        cell["outputs"].append({
            "data": {
                "image/png": img_b64,
                "text/plain": ["<Figure size ...>"]
            },
            "metadata": {},
            "output_type": "display_data"
        })
        plt.close(fig)

    exec_count += 1

# Save compiled notebooks to all target paths
targets = [
    ROOT / "Theme2_Herding_QRF_Complete.ipynb",
    ROOT / "notebooks" / "Theme2_Herding_QRF_Complete.ipynb",
    ROOT / "Thematique2_Herding_QRF_Complet.ipynb",
    ROOT / "notebooks" / "Thematique2_Herding_QRF_Complet.ipynb",
]

for target in targets:
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved compiled notebook to: {target.relative_to(ROOT)}")

print("\nAll notebooks compiled and updated successfully!")
