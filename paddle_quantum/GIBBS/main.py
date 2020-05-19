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
main
"""

from paddle_quantum.GIBBS.HGenerator import H_generator
from paddle_quantum.GIBBS.Paddle_GIBBS import Paddle_GIBBS


def main():
    # gibbs Hamiltonian preparing
    hamiltonian, rho = H_generator()
    rho_B = Paddle_GIBBS(hamiltonian, rho)
    print(rho_B)


if __name__ == '__main__':
    main()
