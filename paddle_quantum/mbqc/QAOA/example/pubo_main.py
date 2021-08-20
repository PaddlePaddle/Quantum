# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
PUBO main
"""

from paddle_quantum.mbqc.QAOA.pubo import mbqc_pubo, brute_force_search, is_poly_valid, random_poly


def main():
    r"""PUBO 主函数。

    """
    # Randomly generate an objective function
    var_num = 4
    polynomial = random_poly(var_num)

    # A standard form of input polynomial x_1 + x_2 - x_3 + x_1 * x_2 - x_1 * x_2 * x_3
    # should be {'x_1': 1, 'x_2':1, 'x_3':-1, 'x_1,x_2': 1, 'x_1,x_2,x_3': -1}
    # var_num = 3
    # poly_dict = {'x_1': 1, 'x_2':1, 'x_3':-1, 'x_1,x_2': 1, 'x_1,x_2,x_3': -1}
    # polynomial = [n, func_dict]
    print("The input polynomial is: ", polynomial)

    is_poly_valid(polynomial)

    # Do the training and obtain the result
    mbqc_result = mbqc_pubo(
        OBJ_POLY=polynomial,  # Objective Function
        DEPTH=6,  # QAOA Depth
        SEED=1024,  # Plant Seed
        LR=0.1,  # Learning Rate
        ITR=120,  # Training Iters
        EPOCH=1  # Epoch Times
    )

    print("Optimal solution by MBQC: " + str(mbqc_result[0]))
    print("Optimal value by MBQC: " + str(mbqc_result[1]))

    # Compute the optimal result by a brute-force method and print the result
    brute_result = brute_force_search(polynomial)
    print("Optimal solution by brute force search: " + str(brute_result[0]))
    print("Optimal value by brute force search: " + str(brute_result[1]))
    print("Difference of optimal values from MBQC and brute force search: " +
          str(mbqc_result[1] - brute_result[1]))


if __name__ == '__main__':
    main()
