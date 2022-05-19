# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
main
"""

import scipy
from numpy import trace as np_trace
import paddle_quantum
from paddle_quantum.qinfo import pauli_str_to_matrix
from paddle_quantum.GIBBS.Paddle_GIBBS import Paddle_GIBBS


def main():
    # Generate Pauli string representing a specific Hamiltonian
    H = [[-1.0, 'z0,z1'], [-1.0, 'z1,z2'], [-1.0, 'z0,z2']]

    # Generate the matrix form of the Hamiltonian
    N_SYS_B = 3  # Number of qubits in subsystem B used to generate Gibbs state
    hamiltonian = pauli_str_to_matrix(H, N_SYS_B).numpy()

    # Generate the target Gibbs state rho
    beta = 1.5  # Set inverse temperature beta
    rho_G = scipy.linalg.expm(-1 * beta * hamiltonian) / np_trace(scipy.linalg.expm(-1 * beta * hamiltonian))

    # Convert to the data type supported by Paddle Quantum
    hamiltonian = hamiltonian
    rho_G = rho_G

    rho_B = Paddle_GIBBS(hamiltonian, rho_G)
    print(rho_B)


if __name__ == '__main__':
    main()
