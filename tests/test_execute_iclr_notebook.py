import nbformat
import pytest

from scripts.execute_iclr_notebook import configured_notebook, execute_notebook


def sample_notebook(tmp_path, body="print(EXECUTE_GPU, RUN_ID)"):
    parameters = nbformat.v4.new_code_cell('EXECUTE_GPU = False\nRUN_ID = "pilot"')
    parameters.metadata["tags"] = ["parameters"]
    notebook = nbformat.v4.new_notebook(cells=[parameters, nbformat.v4.new_code_cell(body)])
    source = tmp_path / "source.ipynb"
    nbformat.write(notebook, source)
    return source


def test_default_is_plan_only_and_source_is_not_modified(tmp_path):
    source = sample_notebook(tmp_path)
    original = source.read_bytes()
    notebook = configured_notebook(source, {})
    namespace = {}
    exec(notebook.cells[0].source, namespace)
    assert namespace["EXECUTE_GPU"] is False
    assert source.read_bytes() == original


def test_explicit_overrides_are_data_not_code(tmp_path):
    source = sample_notebook(tmp_path)
    run_id = "pilot'; raise RuntimeError('injected') #"
    notebook = configured_notebook(source, {"RUN_ID": run_id})
    namespace = {}
    exec(notebook.cells[0].source, namespace)
    assert namespace["RUN_ID"] == run_id
    assert namespace["EXECUTE_GPU"] is False


@pytest.mark.parametrize("overrides", [{"EXECUTE_GP": True}, [], {"RUN_ID": {}},
                                       {"RUN_ID": float("nan")}])
def test_invalid_overrides_are_rejected(tmp_path, overrides):
    with pytest.raises(ValueError):
        configured_notebook(sample_notebook(tmp_path), overrides)


def test_refuses_to_overwrite_source(tmp_path):
    source = sample_notebook(tmp_path)
    with pytest.raises(ValueError, match="new output path"):
        execute_notebook(source, source, {})


def test_completed_cells_are_saved_before_next_stage(tmp_path):
    output = tmp_path / "executed.ipynb"
    source = sample_notebook(tmp_path, "import nbformat\n"
                             f"saved = nbformat.read({str(output)!r}, as_version=4)\n"
                             "assert saved.cells[0].execution_count == 1\n")
    execute_notebook(source, output, {})


@pytest.mark.parametrize("fails", [False, True])
def test_execution_preserves_outputs_even_on_error(tmp_path, fails):
    source = sample_notebook(tmp_path, "raise ValueError('pilot failure')" if fails else "print(RUN_ID)")
    output = tmp_path / "executed.ipynb"
    if fails:
        with pytest.raises(Exception, match="pilot failure"):
            execute_notebook(source, output, {"RUN_ID": "checked"})
    else:
        execute_notebook(source, output, {"RUN_ID": "checked"})
    result = nbformat.read(output, as_version=4)
    assert result.cells[0].execution_count == 1
    assert result.cells[1].outputs[0].output_type == ("error" if fails else "stream")
    with pytest.raises(ValueError, match="new output path"):
        execute_notebook(source, output, {})
