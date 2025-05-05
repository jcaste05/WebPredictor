import os
import importlib.util
import pytest

PROJECT_DIR = "./"

def find_python_files(base_dir):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("test_"):
                yield os.path.join(root, file)

@pytest.mark.parametrize("file_path", list(find_python_files(PROJECT_DIR)))
def test_import_file(file_path):
    rel_path = os.path.relpath(file_path, PROJECT_DIR)
    module_name = rel_path[:-3].replace(os.path.sep, ".")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"Error importing {module_name}: {e}")
