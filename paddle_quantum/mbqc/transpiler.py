# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

"""
此模块包含电路模型和 MBQC 测量模式的转义工具。
"""

from paddle_quantum.mbqc.qobject import Circuit
from paddle_quantum.mbqc.mcalculus import MCalculus

__all__ = [
    "transpile"
]


def transpile(circuit, track=False):
    r"""将输入的量子电路翻译为等价的测量模式。

    Args:
        circuit (Circuit): 量子电路，包含可能的测量部分
        track (bool): 是否显示翻译进度条的开关

    Returns:
        pattern (Pattern): 等价的测量模式
    """
    assert isinstance(circuit, Circuit), "'circuit' must be of type 'Circuit'."
    assert isinstance(track, bool), "'track' must be a 'bool'."
    mc = MCalculus()
    if track:
        mc.track_progress()
    mc.set_circuit(circuit)
    mc.standardize()
    mc.shift_signals()
    mc.optimize_by_row()
    pattern = mc.get_pattern()
    return pattern
