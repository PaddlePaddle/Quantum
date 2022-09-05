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
The module of the quantum gates.
"""

from . import functional
from .base import Gate, ParamGate
from .clifford import Clifford, compose_clifford_circuit
from .single_qubit_gate import H, S, T, X, Y, Z, P, RX, RY, RZ, U3
from .multi_qubit_gate import CNOT, CX, CY, CZ, SWAP
from .multi_qubit_gate import CP, CRX, CRY, CRZ, CU, RXX, RYY, RZZ
from .multi_qubit_gate import MS, CSWAP, Toffoli
from .multi_qubit_gate import UniversalTwoQubits, UniversalThreeQubits
from .custom import Oracle, ControlOracle
from .layer import SuperpositionLayer, WeakSuperpositionLayer
from .layer import LinearEntangledLayer, RealEntangledLayer, ComplexEntangledLayer
from .layer import RealBlockLayer, ComplexBlockLayer
from .layer import QAOALayer
from .encoding import BasisEncoding
from .encoding import AmplitudeEncoding
from .encoding import AngleEncoding
from .encoding import IQPEncoding
