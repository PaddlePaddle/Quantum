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
HGenerator
"""

import numpy
import scipy
import scipy.stats

SEED = 13

__all__ = ["generate_rho_sigma", ]


def generate_rho_sigma():
    r"""Generate two quantum states with specific eigenvalues
    
    Returns: Tuple: including following elements
        - rho
        - sigma
    
    """
    
    numpy.random.seed(SEED)
    V = scipy.stats.unitary_group.rvs(4)  # Generate a random unitary matrix
    D = numpy.diag([0.5, 0.3, 0.1, 0.1])  # Input the spectrum of the target state rho
    V_H = V.conj().T
    rho = V @ D @ V_H  # Generate rho
    # print(rho)  # Print quantum state rho

    # Input the quantum state sigma
    sigma = numpy.diag([0.1, 0.2, 0.3, 0.4])
    return rho, sigma
