Project Memory for LLM

Scope
- This repository `mujoco_playground_kkaji` contains MuJoCo environments for dm_control suite, locomotion, and manipulation. Python package root: `mujoco_playground`.
- Notable submodules:
  - `mujoco_playground/_src/dm_control_suite/*`
  - `mujoco_playground/_src/locomotion/*`
  - `mujoco_playground/_src/manipulation/*`

LEAP Hand MJCF assets dependency
- File: `mujoco_playground/_src/manipulation/leap_hand/xmls/leap_rh_mjx.xml`.
- Inside `<asset>` section, many `<mesh>` entries reference files via relative path: `../../../../../mujoco_menagerie/leap_hand/assets/*.obj`.
- This assumes a sibling repository `mujoco_menagerie` exists next to the repo root. Expected folder layout:
  - `<parent>/mujoco_playground_kkaji/`
  - `<parent>/mujoco_menagerie/`
- If `mujoco_menagerie` is absent at that location, the MuJoCo loader will fail with an "Error opening file" for those OBJ meshes.

How to visualize/edit `leap_rh_mjx.xml`
1) Preferred: Place `mujoco_menagerie` as a sibling to this repo so the relative paths resolve. Required assets path: `mujoco_menagerie/leap_hand/assets/`.
2) Alternative: Copy the needed OBJ meshes into `mujoco_playground/_src/manipulation/leap_hand/xmls/meshes/` and change the `<mesh file>` paths in the XML to `meshes/<name>.obj`. Required file names include:
   - `palm_right.obj`, `base.obj`, `proximal.obj`, `medial.obj`, `distal.obj`, `tip.obj`, `thumb_base.obj`, `thumb_proximal.obj`, `thumb_distal.obj`, `thumb_tip.obj`.
   - Note: `leap_mount.obj` already exists in `xmls/meshes/`.

CLI viewing
- Quick check: `python -m mujoco.viewer --mjcf=mujoco_playground/_src/manipulation/leap_hand/xmls/leap_rh_mjx.xml`
- Ensure current working directory is the repo root so the XML's relative asset paths resolve correctly (if relying on sibling `mujoco_menagerie`).

Hot-reload workflow idea (optional helper script)
- A small Python watcher can reload the model on file change using `watchdog` and `mujoco.viewer`. Load via `mujoco.MjModel.from_xml_path` and reopen viewer on change.

Common pitfalls
- Using path `.../my_leap_hand/...` will not match the current tree; the canonical directory is `.../leap_hand/...`.
- Asset path errors will present like: `Error opening file '.../xmls/../../../../../mujoco_menagerie/... .obj'`.

Notes
- Other manipulation environments embed their assets locally (e.g., `franka_emika_panda/xmls/*`), but LEAP hand references the external menagerie assets by design.

