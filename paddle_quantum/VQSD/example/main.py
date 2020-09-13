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
Main
"""

import numpy

from paddle_quantum.VQSD.HGenerator import generate_rho_sigma
from paddle_quantum.VQSD.Paddle_VQSD import Paddle_VQSD


def main():
    """
    main
    """
    D = [0.5, 0.3, 0.1, 0.1]

    rho, sigma = generate_rho_sigma()

    rho_tilde_np = Paddle_VQSD(rho, sigma)

    print("The estimated spectrum is:", numpy.real(numpy.diag(rho_tilde_np)))
    print('The target spectrum is:', D)


if __name__ == '__main__':
    main()
