"""Execute the existing notebook over SSH and preserve outputs on failure."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def configured_notebook(path, parameters):
    notebook = nbformat.read(path, as_version=4)
    cells = [cell for cell in notebook.cells
             if "parameters" in cell.metadata.get("tags", [])]
    if len(cells) != 1:
        raise ValueError("Expected exactly one parameters cell")
    tree = ast.parse(cells[0].source)
    names = {target.id for node in tree.body if isinstance(node, ast.Assign)
             for target in node.targets if isinstance(target, ast.Name)}
    if not isinstance(parameters, dict) or set(parameters) - names:
        raise ValueError("Overrides must name existing notebook parameters")
    if any(isinstance(value, (dict, list)) for value in parameters.values()):
        raise ValueError("This runner accepts only scalar parameter overrides")
    payload = json.dumps(parameters, allow_nan=False)
    cells[0].source += ("\n\nimport json as _parameter_json\n"
                        f"globals().update(_parameter_json.loads({payload!r}))\n")
    return notebook


def execute_notebook(source, output, parameters):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.exists():
        raise ValueError("Use a new output path; preserve the source and previous attempts")
    notebook = configured_notebook(source, parameters)
    output.parent.mkdir(parents=True, exist_ok=True)

    def started(cell_index, **kwargs):
        # Save the preceding cell before the next potentially long GPU stage.
        nbformat.write(notebook, output)
        print(f"Starting cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(notebook, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(source.parent)}},
                            on_cell_start=started)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
        print(f"Notebook outputs: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameters", type=Path,
                        help="JSON overrides; enable GPU only after external billing/stop checks")
    args = parser.parse_args()
    parameters = json.loads(args.parameters.read_text()) if args.parameters else {}
    execute_notebook(args.notebook, args.output, parameters)


if __name__ == "__main__":
    main()
