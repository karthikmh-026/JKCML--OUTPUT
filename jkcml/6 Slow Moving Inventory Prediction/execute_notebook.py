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
            fig_nums = plt.get_fignums()
            for fig_num in fig_nums:
                fig = plt.figure(fig_num)
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=130)
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
                plt.close(fig)

                cell.outputs.append(nbf.v4.new_output(
                    output_type='display_data',
                    data={
                        'image/png': img_base64,
                        'text/plain': '<Figure size ...>'
                    }
                ))
        except Exception as e:
            err_msg = f"ERROR in cell {idx + 1}: {str(e)}\n"
            print(f"  [ERROR] {err_msg}")
            cell.outputs.append(nbf.v4.new_output(
                output_type='stream',
                name='stderr',
                text=err_msg
            ))

        execution_count += 1

    print(f"[INFO] Writing executed notebook back to {notebook_path}...")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[SUCCESS] {notebook_path} executed and saved successfully with pre-rendered outputs.")


if __name__ == "__main__":
    nb_path = os.path.join("notebooks", "06_Slow_Moving_Inventory_Prediction_ML.ipynb")
    if os.path.exists(nb_path):
        run_notebook(nb_path)

    root_nb = "06_Slow_Moving_Inventory_Prediction_ML.ipynb"
    if os.path.exists(root_nb):
        run_notebook(root_nb)
