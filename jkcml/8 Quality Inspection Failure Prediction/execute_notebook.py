"""
Executes each cell in the notebook in order, captures standard output and
matplotlib figures (as PNG base64), and writes them directly into the .ipynb JSON format.
This ensures the notebook opens in VS Code / Jupyter with all plots and outputs pre-rendered.
"""
import os
import io
import base64
import contextlib
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_notebook(notebook_path: str):
    print(f"[INFO] Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    exec_env = {
        '__name__': '__main__',
        '__file__': os.path.abspath(notebook_path),
        'display': lambda x: print(x)
    }

    execution_count = 1
    total_code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f"[INFO] Executing {total_code_cells} code cells across the 21 steps...")

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        print(f"  -> Executing Code Cell {execution_count}/{total_code_cells}...")
        cell.outputs = []
        cell.execution_count = execution_count

        stdout_buffer = io.StringIO()
        plt.close('all')

        code = cell.source
        lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('%matplotlib') or stripped.startswith('!'):
                continue
            lines.append(line)
        executable_code = '\n'.join(lines)

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
                exec(executable_code, exec_env)

            text_output = stdout_buffer.getvalue()
            if text_output:
                cell.outputs.append(nbf.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=text_output
                ))

            # Capture any matplotlib figures generated in this cell
            figs = [plt.figure(i) for i in plt.get_fignums()]
            for fig in figs:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
                cell.outputs.append(nbf.v4.new_output(
                    output_type='display_data',
                    data={'image/png': img_data, 'text/plain': '<Figure size>'},
                    metadata={}
                ))
            plt.close('all')

        except Exception as e:
            print(f"  [ERROR] Execution failed at cell {execution_count}: {e}")
            cell.outputs.append(nbf.v4.new_output(
                output_type='error',
                ename=type(e).__name__,
                evalue=str(e),
                traceback=[str(e)]
            ))
            raise e

        execution_count += 1

    print(f"[INFO] Writing executed notebook back to {notebook_path}...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[SUCCESS] Notebook {notebook_path} successfully executed and saved with all outputs.")


if __name__ == '__main__':
    for path in ["08_Quality_Inspection_Failure_Prediction_ML.ipynb", "notebooks/08_Quality_Inspection_Failure_Prediction_ML.ipynb"]:
        if os.path.exists(path):
            run_notebook(path)
