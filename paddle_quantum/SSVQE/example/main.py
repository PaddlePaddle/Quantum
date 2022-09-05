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

import numpy
from paddle_quantum.SSVQE.HGenerator import H_generator
from paddle_quantum.SSVQE.Paddle_SSVQE import Paddle_SSVQE
   


if __name__ == '__main__':
    N = 2
    H = H_generator(N)

    loss_components = Paddle_SSVQE(H)

    print('The estimated ground state energy is: ', loss_components[0].numpy())
    print('The theoretical ground state energy: ', numpy.linalg.eigh(H)[0][0])

    print('The estimated 1st excited state energy is: ', loss_components[1].numpy())
    print('The theoretical 1st excited state energy: ', numpy.linalg.eigh(H)[0][1])

    print('The estimated 2nd excited state energy is: ', loss_components[2].numpy())
    print('The theoretical 2nd excited state energy: ', numpy.linalg.eigh(H)[0][2])

    print('The estimated 3rd excited state energy is: ', loss_components[3].numpy())
    print('The theoretical 3rd excited state energy: ', numpy.linalg.eigh(H)[0][3])