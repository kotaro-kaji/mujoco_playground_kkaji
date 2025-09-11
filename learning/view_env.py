"""Interactive environment viewer using the repo's dynamic XML pipeline.

This script loads an env via mujoco_playground.registry (so all assets and
XML includes are resolved internally), then launches a lightweight MuJoCo
viewer that steps the MJX environment and syncs the CPU renderer each frame.

Usage:
  python learning/view_env.py --env_name XArm7PushCube

Notes:
- No training is performed. Actions default to zeros unless specified.
- Uses env.dt for stepping; close the window to exit.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer as mj_viewer

from mujoco_playground import registry


def _np(a) -> np.ndarray:
  """Converts a JAX array to a NumPy array without copying if possible."""
  return np.array(a)


def run_viewer(
    env_name: str,
    seed: int = 0,
    action_mode: str = "zeros",
    action_scale: float = 0.0,
    fps_limit: Optional[float] = None,
):
  # Load env via the repo registry (uses dynamic XML + embedded assets).
  cfg = registry.get_default_config(env_name)
  env = registry.load(env_name, config=cfg)

  # Create a CPU-side model/data for rendering; keep MJX for physics stepping.
  mj_model: mujoco.MjModel = env.mj_model
  mj_data = mujoco.MjData(mj_model)

  # Reset env
  rng = jax.random.PRNGKey(seed)
  state = env.reset(rng)

  # Simple action generator
  def gen_action(step_idx: int):
    if action_mode == "zeros":
      return jp.zeros(env.action_size)
    elif action_mode == "constant":
      return jp.ones(env.action_size) * action_scale
    elif action_mode == "sine":
      t = step_idx * env.dt
      return jp.sin(t) * jp.ones(env.action_size) * action_scale
    elif action_mode == "random":
      key = jax.random.PRNGKey(step_idx)
      return (jax.random.uniform(key, (env.action_size,), minval=-1, maxval=1)
              * action_scale)
    else:
      return jp.zeros(env.action_size)

  # Launch viewer and loop.
  # Use passive mode so we control stepping + sync.
  with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    # FPS limiter (optional)
    dt = env.dt
    target_dt = 1.0 / fps_limit if fps_limit else dt
    step_idx = 0

    while viewer.is_running():
      # Generate a simple control and step MJX physics
      act = gen_action(step_idx)
      state = env.step(state, act)

      # Sync MJX state into CPU renderer
      mj_data.qpos = _np(state.data.qpos)
      mj_data.qvel = _np(state.data.qvel)
      if mj_model.nmocap:
        mj_data.mocap_pos = _np(state.data.mocap_pos)
        mj_data.mocap_quat = _np(state.data.mocap_quat)
      mujoco.mj_forward(mj_model, mj_data)

      # Optional overlay with simple info
      viewer.add_overlay(
          mj_viewer.Overlay.FastText,
          "Info",
          f"step={step_idx}  reward={float(state.reward): .3f}  done={int(state.done)}",
      )

      viewer.sync()
      step_idx += 1

      # Sleep a bit to avoid tight loop if dt is very small
      time.sleep(max(0.0, target_dt))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--env_name", type=str, default="XArm7PushCube")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
      "--action_mode",
      type=str,
      default="zeros",
      choices=["zeros", "constant", "sine", "random"],
  )
  parser.add_argument("--action_scale", type=float, default=0.2)
  parser.add_argument("--fps_limit", type=float, default=None)
  args = parser.parse_args()

  run_viewer(
      env_name=args.env_name,
      seed=args.seed,
      action_mode=args.action_mode,
      action_scale=args.action_scale,
      fps_limit=args.fps_limit,
  )


if __name__ == "__main__":
  main()

