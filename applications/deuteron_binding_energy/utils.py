# !/usr/bin/env python3
# Copyright (c) 2023 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

from typing import Optional
import logging
from paddle_quantum import Hamiltonian
from paddle_quantum.ansatz import Circuit
from openfermion import FermionOperator, jordan_wigner
import numpy as np

__all__ = ["DeuteronHamiltonian", "DeuteronUCC2", "DeuteronUCC3"]


class DeuteronHamiltonian(object):
    def __init__(self, omega: float, V0: float) -> None:
        self.omega = omega
        self.V0 = V0
        logging.info("\n#######################################\nDeuteron Hamiltonian\n#######################################")
        logging.info(f"hbar_omega: {omega:.5f}")
        logging.info(f"V0: {V0:.5f}")

    def get_hamiltonian(self, N: int) -> Hamiltonian:
        h = FermionOperator("0^ 0", self.V0)
        for i in range(N):
            h += 0.5*self.omega*(2*i+1.5)*FermionOperator(f"{i:d}^ {i:d}")
            if i < N-1:
                h += -0.5*self.omega*np.sqrt((i+1)*(i+1.5))*FermionOperator(f"{i+1:d}^ {i:d}")
            if i > 0:
                h += -0.5*self.omega*np.sqrt(i*(i+0.5))*FermionOperator(f"{i-1:d}^ {i:d}")
        h_qubit = jordan_wigner(h)
        return Hamiltonian.from_qubit_operator(h_qubit)


class DeuteronUCC2(Circuit):
    def __init__(self, theta: Optional[float] = None):
        num_qubits = 2
        super().__init__(num_qubits)

        self.x(0)
        self.ry(1)
        self.cx([1, 0])
        if isinstance(theta, float):
            self.update_param(theta)


class DeuteronUCC3(Circuit):
    def __init__(self, theta: Optional[np.array] = None):
        num_qubits = 3
        super().__init__(num_qubits)

        self.x(0)
        self.ry(1)
        self.ry(2)
        self.cx([2, 0])
        self.cx([0, 1])
        self.ry(1)
        self.cx([0, 1])
        self.cx([1, 0])
        if isinstance(theta, np.ndarray):
            self.update_param(theta)

if __name__ == "__main__":
    deuteron_h = DeuteronHamiltonian(7, -5.6865811)
    h1 = deuteron_h.get_hamiltonian(1)
    h2 = deuteron_h.get_hamiltonian(2)
    h3 = deuteron_h.get_hamiltonian(3)
