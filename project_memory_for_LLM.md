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


New scene: `leap_hand_6dof_ctrl_scene.xml` (my_leap_hand)
- File: `mujoco_playground/_src/manipulation/my_leap_hand/xmls/leap_hand_6dof_ctrl_scene.xml`
- Purpose: Add a 6-DoF controllable base that rigidly drives the LEAP hand via a weld, while reusing the standard LEAP hand and reorientation cube assets.
- Key details:
  - Compiler/option: `timestep=0.01`, Euler integrator, with modest solver iters; matches other LEAP scenes numerically.
  - Worldbody:
    - Body `hand_base` at `pos="0 0 0.6" quat="0 1 0 0"` with inertial specified.
    - Six joints on `hand_base`: slides `base_x/y/z` and hinges `base_yaw/pitch/roll` with ranges and damping (class `base6dof`).
    - `site base_mount_site` as a visual/reference site.
    - Visible floor geom `name="floor"` (checkerboard) with `contype=0` and an invisible physical ground `name="ground_physical"` with `contype=1`, `conaffinity=1`, friction `0.8 0.01 0.0001`.
    - Body `goal` is `mocap="true"` at `pos="0.325 0.17 0.0475"`, carrying a visual mesh and a physical box (group 3); used as a spatial reference.
  - Equality:
    - Weld: `<weld body1="hand_base" body2="leap_mount" solref="0.01 2" solimp="0.98 0.995 0.001" relpose="0 0 0 1 0 0 0"/>`.
    - Effect: The LEAP hand (`leap_mount` defined in `leap_rh_mjx.xml`) is rigidly attached to the 6-DoF base. Driving the base joints moves the entire hand.
  - Sensors present:
    - Cube: `cube_position`, `cube_orientation`, `cube_linvel`, `cube_angvel`, `cube_angacc`, `cube_upvector` (all frame sensors on `cube`).
    - Hand: `palm_position` (site `grasp_site`), and fingertips `th/if/mf/rf` tip positions relative to `grasp_site`.
    - Note: There is currently NO `cube_goal_orientation` or `goal_position` sensor defined in this scene.
  - Actuators:
    - Six `position` actuators for base joints: `act_base_x/y/z/yaw/pitch/roll` with `kp` and `forcerange` tuned for stability.
    - Finger actuators are provided by the included `leap_rh_mjx.xml` as usual. Total `nu` = 16 finger + 6 base = 22 (order: included file actuators first, then scene actuators).
  - Keyframe `home`:
    - Commented qpos order: base(6), `leap_mount_freejoint`(7), finger joints(16), `cube_freejoint`(7).
    - Provides `ctrl` targets (mostly zeros, one finger target `0.3`), and sets `mpos/mquat` for the `goal` body.

- Compatibility notes vs existing env code (`LeapHandEnv`, `CubeReorient`/`MyCubeReorient`):
  - Missing sensor: `cube_goal_orientation`. Existing reorient envs call `get_cube_goal_orientation` and will raise if the sensor is absent. If reusing those envs with this scene, add to `<sensor>`:
    - `<framequat name="cube_goal_orientation" objtype="body" objname="goal"/>`
  - Observation expectations: Existing envs assume fingertip and palm sensors exist (they do), and history buffers of size tied to 16 hand joints (unchanged because indexing is by `JOINT_NAMES`).
  - Action interface: Env code uses `self.mjx_model.nu` and delta-to-target control, so it can handle `nu=22`. If you intend base-only control for simple tasks, you can (in a new env) slice actions to 6 and keep finger `ctrl` fixed to the keyframe.
  - Goal position: If you want position-only tasks (cube-to-fixed-goal or palm-to-goal), consider adding `<framepos name="goal_position" objtype="body" objname="goal"/>`, or read the constant from the keyframe `mpos`.

- Intended simple-task usage:
  - Base-only reach: minimize ‖palm − cube‖ using present sensors; fingers held at default.
  - Push-X: reward cube +X progress; action = base 6D; no extra sensors needed.
  - Cube-to-fixed-goal: reward on ‖cube − goal_pos_const‖; can hardcode from keyframe `mpos` if not adding a `goal_position` sensor.

LEAP Hand (my_leap_hand) XML set overview
- Files (path: `mujoco_playground/_src/manipulation/my_leap_hand/xmls/`):
  - `leap_rh_mjx.xml`: Right-hand MJCF definition (model="leap_rh").
  - `reorientation_cube.xml`: Reorientation task cube (visual mesh + physical box with freejoint).
  - `my_scene_mjx_cube.xml`: Scene composition that includes the above two and adds floor, camera, goal body, sensors, and keyframe.

- Include relationships:
  - `my_scene_mjx_cube.xml` does `<include file="leap_rh_mjx.xml"/>` and `<include file="reorientation_cube.xml"/>`.

- `leap_rh_mjx.xml` key details:
  - Compiler/option: `angle="radian"`, `timestep=0.01`, `integrator=Euler`, `iterations=5`, `ls_iterations=8`, and `<flag eulerdamp="disable"/>`.
  - Custom numerics: `<numeric name="max_contact_points" data="30"/>`, `<numeric name="max_geom_pairs" data="12"/>` (used by higher-level code to configure contact constraints; not native engine options by themselves).
  - Assets:
    - Mesh `leap_mount` is local: `meshes/leap_mount.obj`.
    - All anatomical meshes reference menagerie: `../../../../../mujoco_menagerie/leap_hand/assets/{palm_right,base,proximal,medial,distal,tip,thumb_*}.obj`.
    - This requires a sibling repo layout with `mujoco_menagerie/leap_hand/assets/*` resolvable from the XML location.
  - Defaults (class `leap_rh`):
    - Collision geoms: `group=3`, `friction="0.2"`, `solref="0.02 1.5"`, `solimp="0.9 0.99 0.001"`.
    - Visual geoms (class `visual`): `group=2`, `type=mesh`, `contype=0`, `conaffinity=0`, `density=0`, default `material="black"`.
    - Fingertip proxy geoms (class `tip` and `thumb_tip`): defined as small boxes with higher friction `0.7 0.05 0.0002` and semi-transparent rgba, used for contact with objects.
    - Joint template: `axis="0 0 -1"`, `damping=0.2`, `armature=0.00149376`, `actuatorfrcrange="-0.2196 0.2196"`, `frictionloss=0.02`.
    - Joint ranges by class:
      - `mcp`: `-0.314 2.23`, `rot`: `-1.047 1.047`, `pip`: `-0.506 1.885`, `dip`: `-0.366 2.042`.
      - `thumb_cmc`: `-0.349 2.094`, `thumb_axl`: `-0.349 2.094`, `thumb_mcp`: `-0.47 2.443`, `thumb_ipl`: `-1.34 1.88`.
  - Contact filtering:
    - Extensive `<exclude>` pairs prevent `palm` and `leap_mount` from colliding with intermediate finger bodies, allowing contact predominantly at fingertips. Also prevents fingertip-totip collisions among base fingertip bodies.
  - Worldbody structure:
    - `light` targets `palm`. Root body `leap_mount` with visual mesh and two small box collision geoms; `site grasp_site` used as a grasp reference.
    - Hand kinematic tree: three fingers (`if`, `mf`, `rf`) each with bodies `{bs, px, md, ds}` and joints `{mcp, rot, pip, dip}`; thumb with bodies `{mp, bs, px, ds}` and joints `{cmc, axl, mcp, ipl}`.
    - Visual meshes for each phalanx (group 2, no contact) + multiple box geoms approximating collision volumes (group 3, contact enabled).
    - Sites: `if_tip`, `mf_tip`, `rf_tip`, `th_tip` on distal segments; used by sensors in the scene.
  - Actuators:
    - Position actuators for every joint (`*_mcp_act`, `*_rot_act`, `*_pip_act`, `*_dip_act`, `th_*_act`) using the corresponding default joint classes.

- `reorientation_cube.xml` key details:
  - Defaults: class `cube` sets `geom friction=".3 0.05"`, `conaffinity=2`, `condim=3`.
  - Assets: six-face cube textures for both colored and gray variants; 2D `dexcube` texture; `mesh name="cube_mesh"` loaded from `meshes/dex_cube.obj` with `scale=0.035`.
  - Worldbody: body `cube` at `pos="0.11 0.0 0.1"` with a `<freejoint name="cube_freejoint"/>`.
    - Visual-only mesh geom: `type=mesh`, `contype=0`, `conaffinity=0`, `density=0`, `group=2`.
    - Physical collision geom: `type=box`, `size=.035 .035 .035`, `mass=.108`, `group=3`.
    - `site cube_center` for convenience.

- `my_scene_mjx_cube.xml` key details:
  - Visual settings: headlight, azimuth/elevation, high shadow quality.
  - Assets: gradient skybox, checker ground texture/material.
  - Worldbody:
    - `camera name="side"` positioned to view the scene; `geom floor` is a plane at `z=-0.25` with `contype=2`/`conaffinity=2` (collides with group 2?).
    - Body `goal` is `mocap="true"` at `pos="0.325 0.17 0.0475"`:
      - Visual mesh geom (group 2, no contact) and a physical `box` geom `size=.035` and `mass=.108` (group 3). Used as a goal pose reference for the task.
  - Sensors:
    - Cube: `framepos/framequat/framelinvel/frameangvel/frameangacc/framezaxis` on body `cube`.
    - Hand: `framepos` for `palm_position` (site `grasp_site`) and fingertip positions `th/if/mf/rf` relative to `grasp_site` (via `reftype="site" refname="grasp_site"`).
    - Goal: `framequat` and `framezaxis` for body `goal`.
  - Keyframe `home`:
    - Provides initial `qpos` for 16 hand joints (IF/MF/RF: 4 each; TH: 4) followed by the cube freejoint `[x y z qw qx qy qz]`.
    - `ctrl` mirrors the 16 joint positions; `mpos/mquat` specify mocap target for `goal`.

Geom groups and contact semantics (as used across files)
- Group 2: visual-only geoms (`contype=0`, `conaffinity=0`), no collision.
- Group 3: collision-enabled geoms for hand and cubes.
- Floor uses `contype=2`/`conaffinity=2` to selectively collide with group 2/3 depending on engine settings; in practice, the cube and hand collision geoms (group 3) interact with the floor.

Operational implications
- Hand-object interaction is primarily through fingertip/tip proxy geoms with higher friction and constrained self-collision, improving stability of grasping and reducing spurious contacts with palm or mount.
- The cube uses a high-fidelity visual mesh plus a simple physical box for robust contact.
- External dependency on `mujoco_menagerie` is required for hand visual meshes; simulation still runs using box collision proxies even if meshes are missing (but visuals will be absent).

Recent edits (control of `leap_mount`)
- In `leap_rh_mjx.xml`, `leap_mount` is changed from fixed to free by adding `<freejoint name="leap_mount_freejoint"/>` under the `leap_mount` body.
- Added six body-actuated `motor` actuators to control spatial wrench on `leap_mount`:
  - Forces: `leap_mount_fx`, `leap_mount_fy`, `leap_mount_fz` with `ctrlrange="-50 50"`.
  - Torques: `leap_mount_tx`, `leap_mount_ty`, `leap_mount_tz` with `ctrlrange="-10 10"`.
- Total ctrl dims now: original 16 finger joints + 6 mount motors = 22.
- If 7D pose control (xyz + quaternion) is desired as control inputs, keep these 6 motors and implement a PD mapping from pose error to wrench; quaternion error (vector part) → torque, position error → force. Alternatively, consider a mocap + weld setup to drive `leap_mount` to a target pose via mocap states (not via `ctrl`).
