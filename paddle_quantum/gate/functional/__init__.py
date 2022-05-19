# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The module that contains the functions of various quantum gates.
"""

from .single_qubit_gate import h
from .single_qubit_gate import s
from .single_qubit_gate import t
from .single_qubit_gate import x
from .single_qubit_gate import y
from .single_qubit_gate import z
from .single_qubit_gate import p
from .single_qubit_gate import rx
from .single_qubit_gate import ry
from .single_qubit_gate import rz
from .single_qubit_gate import u3
from .multi_qubit_gate import cnot
from .multi_qubit_gate import cx
from .multi_qubit_gate import cy
from .multi_qubit_gate import cz
from .multi_qubit_gate import swap
from .multi_qubit_gate import cp
from .multi_qubit_gate import crx
from .multi_qubit_gate import cry
from .multi_qubit_gate import crz
from .multi_qubit_gate import cu
from .multi_qubit_gate import rxx
from .multi_qubit_gate import ryy
from .multi_qubit_gate import rzz
from .multi_qubit_gate import ms
from .multi_qubit_gate import cswap
from .multi_qubit_gate import toffoli
from .multi_qubit_gate import universal_two_qubits
from .multi_qubit_gate import universal_three_qubits
from .multi_qubit_gate import oracle
