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
量桨平台的量子化学模块
"""

from .qmodel import QModel
from . import ansatz
from .run import run_chem
from .qchem import *
import platform
import warnings

__all__ = [
    "run_chem",
    "QModel",
    "ansatz",
    "geometry",
    "get_molecular_data",
    "active_space",
    # forward compatible with original qchem module
    "fermionic_hamiltonian",
    "spin_hamiltonian"
]

if platform.system() == "Windows":
    warning_msg = ("Currently, Windows' users can't use 'hartree fock' ansatz "
                   "for ground state energy calculation in `run_chem`, "
                   "since it depends on pyscf, which is not available on Windows. "
                   "We will work it out in the near future, sorry for the inconvenience.")
    warnings.warn(message=warning_msg)
