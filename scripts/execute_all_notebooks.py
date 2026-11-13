import json
import sys
import io
import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_dir = r'C:\Users\hp\.gemini\antigravity\scratch\herding_qrf_project'
os.chdir(project_dir)

nb_source_path = os.path.join(project_dir, 'Theme2_Herding_QRF_Complete.ipynb')

with open(nb_source_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

glob_ns = {}
exec_count = 1

total_cells = len(nb['cells'])
sys.stderr.write(f"Total cells to process: {total_cells}\n")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = "".join(cell['source'])
        sys.stderr.write(f"Executing code cell {exec_count}...\n")
        
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        plt.close('all')
        
        try:
            def mock_display(obj):
                if isinstance(obj, pd.DataFrame):
                    print(obj.to_string())
                else:
                    print(obj)
            glob_ns['display'] = mock_display
            
            exec(code, glob_ns)
            stdout_text = redirected_output.getvalue()
        except Exception as e:
            stdout_text = redirected_output.getvalue() + f"\nError: {e}"
            sys.stderr.write(f"Error in cell {exec_count}: {e}\n")
        finally:
            sys.stdout = old_stdout
            
        cell['execution_count'] = exec_count
        cell['outputs'] = []
        
        if stdout_text.strip():
            cell['outputs'].append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in stdout_text.splitlines()]
            })
            
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                cell['outputs'].append({
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            except Exception as e_fig:
                sys.stderr.write(f"Figure save warning in cell {exec_count}: {e_fig}\n")
            finally:
                plt.close(fig)
            
        exec_count += 1

targets = [
    os.path.join(project_dir, 'Theme2_Herding_QRF_Complete.ipynb'),
    os.path.join(project_dir, 'notebooks', 'Theme2_Herding_QRF_Complete.ipynb'),
    os.path.join(project_dir, 'Thematique2_Herding_QRF_Complet.ipynb'),
    os.path.join(project_dir, 'notebooks', 'Thematique2_Herding_QRF_Complet.ipynb')
]

for t in targets:
    os.makedirs(os.path.dirname(t), exist_ok=True)
    with open(t, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

sys.stderr.write(f"Successfully executed all {exec_count-1} cells and populated notebooks!\n")
