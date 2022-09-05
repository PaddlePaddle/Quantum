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

"""
Using a H generator to define the H under WINDOWS
"""

from numpy import array, kron, trace
import scipy

__all__ = [
    "H_generator",
    "H2_generator",
]


def H_generator():
    r"""Generate a Hamiltonian with trivial descriptions
    
    Returns:
        Tuple: including following elements
            - H: the Hamiltonian
            - rho: density matrix
    """

    beta = 1
    sigma_I = array([[1, 0], [0, 1]])
    sigma_Z = array([[1, 0], [0, -1]])
    sigma_X = array([[0, 1], [1, 0]])
    sigma_Y = array([[0, -1j], [1j, 0]])
    # H = kron(kron(sigma_X, sigma_I), sigma_X) + 0.9 * kron(kron(sigma_Z, sigma_I), sigma_I) \
    #     - 0.3 * kron(kron(sigma_I, sigma_I), sigma_Y)
    H = 0.4 * kron(sigma_Z, sigma_I) + 0.4 * kron(
        sigma_I, sigma_Z) + 0.2 * kron(sigma_X, sigma_X)
    rho = scipy.linalg.expm(-1 * beta *
                            H) / trace(scipy.linalg.expm(-1 * beta * H))
    return H.astype('complex128'), rho.astype('complex128')


def H2_generator():
    r"""Generate a Hamiltonian with trivial descriptions
    
    Returns: 
        tuple contains
            - H: Hamiltonian, a list of Pauli string
            - N: the number of qubits
    """

    beta = 1
    sigma_I = array([[1, 0], [0, 1]])
    sigma_Z = array([[1, 0], [0, -1]])
    sigma_X = array([[0, 1], [1, 0]])
    sigma_Y = array([[0, -1j], [1j, 0]])
    # H = (-0.04207897647782276) * kron(kron(kron(sigma_I, sigma_I), sigma_I), sigma_I) \
    #     + (0.17771287465139946) * kron(kron(kron(sigma_Z, sigma_I), sigma_I), sigma_I) \
    #     + (0.1777128746513994) * kron(kron(kron(sigma_I, sigma_Z), sigma_I), sigma_I) \
    #     + (-0.24274280513140462) * kron(kron(kron(sigma_I, sigma_I), sigma_Z), sigma_I) \
    #     + (-0.24274280513140462) * kron(kron(kron(sigma_I, sigma_I), sigma_I), sigma_Z) \
    #     + (0.17059738328801052) * kron(kron(kron(sigma_Z, sigma_Z), sigma_I), sigma_I) \
    #     + (0.04475014401535161) * kron(kron(kron(sigma_Y, sigma_X), sigma_X), sigma_Y) \
    #     + (-0.04475014401535161) * kron(kron(kron(sigma_Y, sigma_Y), sigma_X), sigma_X) \
    #     + (-0.04475014401535161) * kron(kron(kron(sigma_X, sigma_X), sigma_Y), sigma_Y) \
    #     + (0.04475014401535161) * kron(kron(kron(sigma_X, sigma_Y), sigma_Y), sigma_X) \
    #     + (0.12293305056183798) * kron(kron(kron(sigma_Z, sigma_I), sigma_Z), sigma_I) \
    #     + (0.1676831945771896) * kron(kron(kron(sigma_Z, sigma_I), sigma_I), sigma_Z) \
    #     + (0.1676831945771896) * kron(kron(kron(sigma_I, sigma_Z), sigma_Z), sigma_I) \
    #     + (0.12293305056183798) * kron(kron(kron(sigma_I, sigma_Z), sigma_I), sigma_Z) \
    #     + (0.17627640804319591) * kron(kron(kron(sigma_I, sigma_I), sigma_Z), sigma_Z)
    H = [
        [-0.04207897647782277, 'i'],
        [0.17771287465139946, 'z0'],
        [0.1777128746513994, 'z1'],
        [-0.2427428051314046, 'z2'],
        [-0.24274280513140462, 'z3'],
        [0.17059738328801055, 'z0,z1'],
        [0.04475014401535163, 'y0,x1,x2,y3'],
        [-0.04475014401535163, 'y0,y1,x2,x3'],
        [-0.04475014401535163, 'x0,x1,y2,y3'],
        [0.04475014401535163, 'x0,y1,y2,x3'],
        [0.12293305056183797, 'z0,z2'],
        [0.1676831945771896, 'z0,z3'],
        [0.1676831945771896, 'z1,z2'],
        [0.12293305056183797, 'z1,z3'],
        [0.1762764080431959, 'z2,z3']
        ]
    # rho = scipy.linalg.expm(-1 * beta *
    #                         H) / trace(scipy.linalg.expm(-1 * beta * H))
    N = 4
    return H, N
