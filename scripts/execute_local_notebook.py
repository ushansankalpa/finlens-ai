"""Execute a notebook in the project's interpreter and preserve actual outputs."""
import argparse
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    path = Path(args.path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Notebook must be inside workspace")
    kernel_dir = ROOT/".jupyter"/"kernels"/"assessment"
    kernel_dir.mkdir(parents=True,exist_ok=True)
    (kernel_dir/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],
            "display_name":"Assessment venv","language":"python"}),encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(ROOT/".jupyter")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT/".jupyter"/"runtime")
    os.environ["IPYTHONDIR"] = str(ROOT/".jupyter"/"ipython")
    os.environ["MPLCONFIGDIR"] = str(ROOT/".matplotlib")
    notebook = nbformat.read(path,as_version=4)
    client = NotebookClient(notebook,timeout=180,kernel_name="assessment",resources={"metadata":{"path":str(ROOT)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook,path)
    print("Executed with actual outputs:",path)


if __name__=="__main__":
    main()
