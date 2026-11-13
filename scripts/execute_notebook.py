import json
import sys
import io
import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nb_path = r'C:\Users\hp\.gemini\antigravity\scratch\herding_qrf_project\Thematique2_Herding_QRF_Complet.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Change directory to herding_qrf_project so relative paths like ./data work
os.chdir(r'C:\Users\hp\.gemini\antigravity\scratch\herding_qrf_project')

# Global namespace for execution
glob_ns = {}
exec_count = 1

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code = "".join(cell['source'])
        # Skip plt.show() blocking
        # Redirect stdout
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        
        # Track figures
        plt.close('all')
        
        try:
            # We mock display() to print as text / markdown table
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
            
        # Capture any matplotlib figures created
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
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
            plt.close(fig)
            
        exec_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Successfully executed and populated {exec_count-1} cells in the notebook!")
