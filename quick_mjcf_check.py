# quick_mjcf_check.py
import os, traceback, mujoco
path = "mujoco_playground/_src/manipulation/my_leap_hand/xmls/leap_hand_6dof_ctrl_scene.xml"
print("Loading:", os.path.abspath(path))
try:
    mujoco.MjModel.from_xml_path(path)
    print("Loaded OK")
except Exception as e:
    print("ERROR:", repr(e))
    traceback.print_exc()