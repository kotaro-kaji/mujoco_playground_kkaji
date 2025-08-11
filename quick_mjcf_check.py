# quick_mjcf_check.py
import os, traceback, mujoco
path = "mujoco_playground/_src/manipulation/my_leap_hand/xmls/ur5e_leap_menagerie_scene.xml"
print("Loading:", os.path.abspath(path))
try:
    mujoco.MjModel.from_xml_path(path)
    print("Loaded OK")
except Exception as e:
    print("ERROR:", repr(e))
    traceback.print_exc()