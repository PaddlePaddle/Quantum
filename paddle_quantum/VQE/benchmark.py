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
benchmark the result
"""


import platform
import matplotlib.pyplot as plt
import numpy
from paddle_quantum.qinfo import pauli_str_to_matrix
from paddle_quantum.VQE.chemistrysub import H2_generator

__all__ = [
    "benchmark_result",
]


def benchmark_result():
    # Read H and calc using numpy
    sysStr = platform.system()
    if sysStr == 'Windows':
        #  Windows does not support SCF, using H2_generator instead
        print('Molecule data will be read from built-in function')
        Hamiltonian, N = H2_generator()
        print('Read Process Finished')
    elif sysStr == 'Linux' or sysStr == 'Darwin':
        # for linux only
        from paddle_quantum.VQE.chemistrygen import read_calc_H
        # Hamiltonian and cnot module preparing, must be executed under Linux
        # Read the H2 molecule data
        print('Molecule data will be read from h2.xyz')
        Hamiltonian, N = read_calc_H(geo_fn='h2.xyz')
        print('Read Process Finished')
    else:
        print("Don't support this os.")

    result = numpy.load('./output/summary_data.npz')

    eig_val, eig_state = numpy.linalg.eig(pauli_str_to_matrix(Hamiltonian, N))
    min_eig_H = numpy.min(eig_val.real)
    min_loss = numpy.ones([len(result['iter'])]) * min_eig_H

    plt.figure(1)
    func1, = plt.plot(result['iter'], result['energy'], alpha=0.7, marker='', linestyle="-", color='r')
    func_min, = plt.plot(result['iter'], min_loss, alpha=0.7, marker='', linestyle=":", color='b')
    plt.xlabel('Number of iteration')
    plt.ylabel('Energy (Ha)')

    plt.legend(handles=[
        func1,
        func_min
    ],
        labels=[
            r'$\left\langle {\psi \left( {\theta } \right)} '
            r'\right|H\left| {\psi \left( {\theta } \right)} \right\rangle $',
            'Ground-state energy',
        ], loc='best')

    # plt.savefig("vqe.png", bbox_inches='tight', dpi=300)
    plt.show()
