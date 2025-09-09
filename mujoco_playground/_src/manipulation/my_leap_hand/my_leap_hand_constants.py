# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for leap hand."""

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "manipulation" / "my_leap_hand"
CUBE_XML = ROOT_PATH / "xmls" / "leap_hand_6dof_ctrl_scene.xml"

# 22は、base(6), finger joints(16)
NQ = 22 #23な気もしてきた。baseの姿勢をquatで取得している可能性あり。
NV = 22
NU = 22  # 実際の制御次元は手指16 + base6 = 22

JOINT_NAMES = [
    # index
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    # middle
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    # ring
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    # thumb
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
]

ACTUATOR_NAMES = [
    # index
    "if_mcp_act",
    "if_rot_act",
    "if_pip_act",
    "if_dip_act",
    # middle
    "mf_mcp_act",
    "mf_rot_act",
    "mf_pip_act",
    "mf_dip_act",
    # ring
    "rf_mcp_act",
    "rf_rot_act",
    "rf_pip_act",
    "rf_dip_act",
    # thumb
    "th_cmc_act",
    "th_axl_act",
    "th_mcp_act",
    "th_ipl_act",
]

FINGERTIP_NAMES = [
    "th_tip",
    "if_tip",
    "mf_tip",
    "rf_tip",
]

BASE_JOINTS_NAMES = [
    "base_x",
    "base_y",
    "base_z",
    "base_yaw",
    "base_pitch",
    "base_roll",
]