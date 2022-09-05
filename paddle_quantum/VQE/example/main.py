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

import platform
from paddle_quantum.VQE.Paddle_VQE import Paddle_VQE
from paddle_quantum.VQE.benchmark import benchmark_result
from paddle_quantum.VQE.chemistrysub import H2_generator


if __name__ == '__main__':
    #Main Learning network using dynamic graph
    # Read data from built-in function or xyz file depending on OS
    sysStr = platform.system()

    if sysStr == 'Windows':
        #  Windows does not support SCF, using H2_generator instead
        print('Molecule data will be read from built-in function')
        hamiltonian, N = H2_generator()
        print('Read Process Finished')

    elif sysStr in ('Linux', 'Darwin'):
        # for linux only
        from paddle_quantum.VQE.chemistrygen import read_calc_H
        # Hamiltonian and cnot module preparing, must be executed under Linux
        # Read the H2 molecule data
        print('Molecule data will be read from h2.xyz')
        hamiltonian, N = read_calc_H(geo_fn='h2.xyz')
        print('Read Process Finished')

    else:
        print("Don't support this OS.")

    Paddle_VQE(hamiltonian, N)
    benchmark_result()
