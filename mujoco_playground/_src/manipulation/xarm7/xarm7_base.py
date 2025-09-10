"""Temporary XArm7 base class (Stage 0).

For Stage 0, inherit PandaRobotiqBase to keep behavior identical while
scaffolding a separate module namespace for XArm7.
"""

from typing import Any, Dict, Optional, Union

from etils import epath
from ml_collections import config_dict

from mujoco_playground._src.manipulation.franka_emika_panda_robotiq import (
    panda_robotiq,
)


def get_assets() -> Dict[str, bytes]:
  """Reuse Panda+Robotiq assets for Stage 0."""
  return panda_robotiq.get_assets()


class XArm7Base(panda_robotiq.PandaRobotiqBase):
  """Stage-0 XArm7 base, reusing Panda base implementation."""

  def __init__(
      self,
      config: config_dict.ConfigDict,
      xml_path: epath.Path,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, xml_path, config_overrides)

