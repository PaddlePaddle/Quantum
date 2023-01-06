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
The utilities.
"""

from typing import Tuple, Optional
from itertools import product
import numpy as np

__all__ = ["orb2spinorb"]


def orb2spinorb(
    num_modes: int,
    single_ex_amps: Optional[np.ndarray] = None,
    double_ex_amps: Optional[np.ndarray] = None
) -> Tuple[np.ndarray]:
    r"""
    Transform molecular orbital integral into spin orbital integral, assume
    the quantum system is spin restricted.

    Args:
        num_modes: Number of molecular orbitals.
        single_ex_amps: One electron integral.
        double_ex_amps: Two electron integral.
    
    Return:
        The molecular integral in spin orbital form.
    """
    if isinstance(single_ex_amps, np.ndarray):
        assert single_ex_amps.shape == (num_modes, num_modes)
        single_ex_amps_so = np.zeros((2*num_modes, 2*num_modes))
        for p in range(num_modes):
            single_ex_amps_so[2*p, 2*p] = single_ex_amps[p, p]
            single_ex_amps_so[2*p+1, 2*p+1] = single_ex_amps[p, p]
            for q in range(p+1, num_modes):
                single_ex_amps_so[2*p, 2*q] = single_ex_amps[p, q]
                single_ex_amps_so[2*p+1, 2*q+1] = single_ex_amps[p, q]

                single_ex_amps_so[2*q, 2*p] = single_ex_amps[p, q]
                single_ex_amps_so[2*q+1, 2*p+1] = single_ex_amps[p, q]
    if isinstance(double_ex_amps, np.ndarray):
        assert double_ex_amps.shape == (num_modes, num_modes, num_modes, num_modes)
        double_ex_amps_so = np.zeros((2*num_modes, 2*num_modes, 2*num_modes, 2*num_modes))
        for p, r, s, q in product(range(num_modes), repeat=4):
            double_ex_amps_so[2*p, 2*r, 2*s, 2*q] = double_ex_amps[p, r, s, q]
            double_ex_amps_so[2*p+1, 2*r, 2*s, 2*q+1] = double_ex_amps[p, r, s, q]
            double_ex_amps_so[2*p, 2*r+1, 2*s+1, 2*q] = double_ex_amps[p, r, s, q]
            double_ex_amps_so[2*p+1, 2*r+1, 2*s+1, 2*q+1] = double_ex_amps[p, r, s, q]
    
    if isinstance(single_ex_amps, np.ndarray) and isinstance(double_ex_amps, np.ndarray):
        return single_ex_amps_so, double_ex_amps_so
    elif isinstance(single_ex_amps, np.ndarray):
        return single_ex_amps_so
    elif isinstance(double_ex_amps, np.ndarray):
        return double_ex_amps_so
    else:
        raise ValueError("One of the `single_ex_amps` and `double_ex_amps` should be an np.ndarray.")


if __name__ == "__main__":
    import numpy as np
    from openfermion import InteractionOperator
    from openfermion.utils import is_hermitian

    # build symmetric matrix
    num_modes = 2
    A = np.random.randn(num_modes, num_modes)
    A = A + A.T
    B = np.random.randn(num_modes, num_modes, num_modes, num_modes)
    B = B + np.transpose(B, (3, 2, 1, 0))

    def test_array():
        P, Q = orb2spinorb(num_modes, A, B)
        np.testing.assert_array_almost_equal(P, P.T)
        np.testing.assert_array_almost_equal(A, P[::2, ::2])
        np.testing.assert_array_almost_equal(A, P[1:2*num_modes:2, 1:2*num_modes:2])
        np.testing.assert_array_almost_equal(Q, np.transpose(Q, (3, 2, 1, 0)))
        np.testing.assert_array_almost_equal(Q[::2, ::2, ::2, ::2], B)
        np.testing.assert_array_almost_equal(Q[1:2*num_modes:2, ::2, ::2, 1:2*num_modes:2], B)

        V = InteractionOperator(0.0, P, Q)
        assert is_hermitian(V)
    
    test_array()