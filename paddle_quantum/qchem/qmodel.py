# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
用于构建量子电路层的基类
该模块依赖 paddlepaddle。
"""

from paddle import nn


class QModel(nn.Layer):
    r"""量子化学用量子线路的基类，任何自定义量子线路都需要继承自这个类。
    """
    def __init__(self, num_qubit: int):
        r"""构造函数

        Args:
            num_qubit (int): 量子比特数目
        """
        super().__init__(name_scope="QModel")
        self._n_qubit = num_qubit
        self._circuit = None

    @property
    def n_qubit(self):
        r"""量子比特数目
        """
        return self._n_qubit

    @property
    def circuit(self):
        r"""量子电路
        """
        if self._circuit is None:
            print("Circuit is not built, please run `forward` to build the circuit first.")
            return None
        else:
            return self._circuit
