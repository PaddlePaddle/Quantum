# Copyright (c) 2020 Paddle Quantum Authors. All Rights Reserved.
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
    """
    Generate a Hamiltonian with trivial descriptions
    :return: a Hamiltonian, 'mat'
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
    return H.astype('complex64'), rho.astype('complex64')


def H2_generator():
    """
    Generate a Hamiltonian with trivial descriptions
    Returns: A Hamiltonian, 'mat'
    """

    beta = 1
    sigma_I = array([[1, 0], [0, 1]])
    sigma_Z = array([[1, 0], [0, -1]])
    sigma_X = array([[0, 1], [1, 0]])
    sigma_Y = array([[0, -1j], [1j, 0]])
    H = (-0.04207897647782276) * kron(kron(kron(sigma_I, sigma_I), sigma_I), sigma_I) \
        + (0.17771287465139946) * kron(kron(kron(sigma_Z, sigma_I), sigma_I), sigma_I) \
        + (0.1777128746513994) * kron(kron(kron(sigma_I, sigma_Z), sigma_I), sigma_I) \
        + (-0.24274280513140462) * kron(kron(kron(sigma_I, sigma_I), sigma_Z), sigma_I) \
        + (-0.24274280513140462) * kron(kron(kron(sigma_I, sigma_I), sigma_I), sigma_Z) \
        + (0.17059738328801052) * kron(kron(kron(sigma_Z, sigma_Z), sigma_I), sigma_I) \
        + (0.04475014401535161) * kron(kron(kron(sigma_Y, sigma_X), sigma_X), sigma_Y) \
        + (-0.04475014401535161) * kron(kron(kron(sigma_Y, sigma_Y), sigma_X), sigma_X) \
        + (-0.04475014401535161) * kron(kron(kron(sigma_X, sigma_X), sigma_Y), sigma_Y) \
        + (0.04475014401535161) * kron(kron(kron(sigma_X, sigma_Y), sigma_Y), sigma_X) \
        + (0.12293305056183798) * kron(kron(kron(sigma_Z, sigma_I), sigma_Z), sigma_I) \
        + (0.1676831945771896) * kron(kron(kron(sigma_Z, sigma_I), sigma_I), sigma_Z) \
        + (0.1676831945771896) * kron(kron(kron(sigma_I, sigma_Z), sigma_Z), sigma_I) \
        + (0.12293305056183798) * kron(kron(kron(sigma_I, sigma_Z), sigma_I), sigma_Z) \
        + (0.17627640804319591) * kron(kron(kron(sigma_I, sigma_I), sigma_Z), sigma_Z)
    rho = scipy.linalg.expm(-1 * beta *
                            H) / trace(scipy.linalg.expm(-1 * beta * H))
    N = 4
    return H.astype('complex64'), rho.astype('complex64'), N
