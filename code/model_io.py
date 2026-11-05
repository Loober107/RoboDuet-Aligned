"""Model loading helpers.

MuJoCo's C++ layer cannot open paths containing non-ASCII characters on
Windows (this workspace lives under "...\\OneDrive\\<CJK>\\..."), so the XML is
read in Python and handed over as a string. Models that reference meshes resolve
meshdir relative to the working directory, so compilation happens with the
working directory temporarily switched to the model's folder.
"""
import contextlib
import os
import pathlib

import mujoco

ROOT = pathlib.Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "go1_arm.xml"


@contextlib.contextmanager
def _in(directory):
    old = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(old)


def load_model_path(path):
    """Load an MJCF that may reference meshes (meshdir relative to its folder)."""
    p = pathlib.Path(path)
    xml = p.read_text(encoding="utf-8")
    with _in(p.parent):
        return mujoco.MjModel.from_xml_string(xml)


def load_model(path=None):
    p = pathlib.Path(path) if path is not None else MODEL_PATH
    return load_model_path(p)
