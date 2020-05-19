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
HGenerator
"""

from numpy import diag
import scipy

SEED = 1

__all__ = ["generate_rho_sigma", ]


def generate_rho_sigma():
    # V is a 4x4 random Unitary
    scipy.random.seed(SEED)
    V = scipy.stats.unitary_group.rvs(4)
    # generate rho
    D = diag([0.1, 0.2, 0.3, 0.4])
    sigma = diag([0.4, 0.3, 0.2, 0.1])
    V_H = V.conj().T
    rho = V @D @V_H

    return rho.astype('complex64'), sigma.astype('complex64')
