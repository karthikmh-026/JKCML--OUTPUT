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


def run_notebook(notebook_path):
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

            # Capture any figures created by cell
            fig_nums = plt.get_fignums()
            for fnum in fig_nums:
                fig = plt.figure(fnum)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
                cell.outputs.append(nbf.v4.new_output(
                    output_type='display_data',
                    data={'image/png': img_b64, 'text/plain': '<Figure size>'},
                    metadata={}
                ))
            plt.close('all')

        except Exception as e:
            err_msg = f"[ERROR in cell {execution_count}]: {str(e)}"
            print(err_msg)
            cell.outputs.append(nbf.v4.new_output(
                output_type='error',
                ename=type(e).__name__,
                evalue=str(e),
                traceback=[err_msg]
            ))
            break

        execution_count += 1

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[INFO] Successfully executed and rendered {execution_count - 1}/{total_code_cells} cells in {notebook_path}")


if __name__ == "__main__":
    nb_path = os.path.join("notebooks", "04_Predictive_Maintenance_ML.ipynb")
    if not os.path.exists(nb_path):
        import generate_notebook
        generate_notebook.create_predictive_maintenance_notebook()
    run_notebook(nb_path)
