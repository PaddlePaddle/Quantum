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
benchmark the result
"""

import platform

import matplotlib.pyplot as plt
import numpy
from paddle_quantum.VQE.chemistrysub import H2_generator


def benchmark_result():
    """
    benchmark using numpy
    """

    # Read H and calc using numpy
    sysStr = platform.system()
    if sysStr == 'Windows':
        #  Windows does not support SCF, using H2_generator instead
        print('Molecule data will be read from built-in function')
        _H, _, _ = H2_generator()
        print('Read Process Finished')
    elif sysStr == 'Linux' or sysStr == 'Darwin':
        # for linux only
        from paddle_quantum.VQE.chemistrygen import read_calc_H
        # Harmiltonian and cnot module preparing, must be executed under Linux
        # Read the H2 molecule data
        print('Molecule data will be read from h2.xyz')
        _H, _, _ = read_calc_H(geo_fn='h2.xyz')
        print('Read Process Finished')
    else:
        print("Don't support this os.")

    # plot
    x1 = numpy.load('./output/summary_data.npz')

    eig_val, eig_state = numpy.linalg.eig(_H)
    min_eig_H = numpy.min(eig_val)
    min_loss = numpy.ones([len(x1['iter'])]) * min_eig_H

    plt.figure(1)
    func1, = plt.plot(
        x1['iter'],
        x1['energy'],
        alpha=0.7,
        marker='',
        linestyle="--",
        color='m')
    func_min, = plt.plot(
        x1['iter'], min_loss, alpha=0.7, marker='', linestyle=":", color='b')
    plt.xlabel('Number of iteration')
    plt.ylabel('Energy (Ha)')

    plt.legend(
        handles=[func1, func_min],
        labels=[
            r'$\left\langle {\psi \left( {\bf{\theta }} \right)} '
            r'\right|H\left| {\psi \left( {\bf{\theta }} \right)} \right\rangle $',
            'Minimum energy',
        ],
        loc='best')

    # output the picture
    plt.show()


def main():
    """
    Call the real benchmark function
    """

    benchmark_result()


if __name__ == '__main__':
    main()
