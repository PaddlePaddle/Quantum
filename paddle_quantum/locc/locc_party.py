# !/usr/bin/env python3
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

r"""
The source file of the LoccParty class.
"""


class LoccParty(object):
    r"""An LOCC party.

    Args:
        num_qubits: Number of qubits of this party.
    """
    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.qubits = [None] * num_qubits

    def __setitem__(self, key, value):
        self.qubits[key] = value

    def __getitem__(self, item):
        return self.qubits[item]

    def __len__(self):
        return self.num_qubits
