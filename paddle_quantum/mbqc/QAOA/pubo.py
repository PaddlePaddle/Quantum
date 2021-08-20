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
Use MBQC-QAOA algorithm to solve PUBO problem and compare with the solution by brute-force search
"""

from numpy import random
from time import perf_counter
from sympy import symbols
from copy import deepcopy
from paddle import seed, optimizer
from paddle_quantum.mbqc.QAOA.qaoa import get_solution_string, MBQC_QAOA_Net

__all__ = [
    "random_poly",
    "is_poly_valid",
    "dict_to_symbol",
    "brute_force_search",
    "mbqc_pubo"
]


def random_poly(var_num):
    r"""随机生成一个多项式函数。

    Args:
        var_num (int): 多项式变量个数

    Returns:
        list: 列表的第一个元素为变量个数，第二个元素为各单项式构成的字典

    代码示例:

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.pubo import random_poly
        polynomial = random_poly(3)
        print("The random polynomial is: \n", polynomial)

    ::

        The random polynomial is:
        [3, {'cons': 0.8634473818565477, 'x_3': 0.10521510553668978, 'x_2': 0.7519767805645575,
        'x_2,x_3': -0.20826424036741054, 'x_1': -0.2795543099640111, 'x_1,x_3': -0.06628925930094798,
        'x_1,x_2': -0.6094165475879592, 'x_1,x_2,x_3': 0.175938331955921}]
    """
    # Random a constant item in the function
    poly_dict = {'cons': random.rand()}
    # Random the other items
    for bit in range(1, 2 ** var_num):
        item = bin(bit)[2:].zfill(var_num)
        var_str = []
        for j in range(1, var_num + 1):
            if int(item[j - 1]) == 1:
                var_str.append('x_' + str(j))
        var_str = ",".join(var_str)
        poly_dict[var_str] = random.rand() - random.rand()
    poly = [var_num, poly_dict]
    # Return polynomial
    return poly


def is_poly_valid(poly):
    r"""检查用户输入的多项式是否符合规范。

    Args:
        poly (list): 列表第一个元素为变量个数，第二个元素为用户输入的字典类型的多项式

    Note:
        为了代码规范，我们要求用户输入的多项式满足下列要求：1. 单项式中每个变量指数最多为一，2. 多项式变量个数与用户输入的变量个数一致，
        3.变量为连续编号的 ``x_1,...,x_n``，另外，常熟项的键为 `cons`，多项式字典的键值不能重复，否则后面的条目会覆盖之前的条目。

    Returns:
        list: 列表的第一个元素为变量个数，第二个元素为各单项式构成的字典

    代码示例:

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.pubo import random_poly, is_poly_valid
        polynomial = random_poly(3)
        print("The random polynomial is: \n", polynomial)
        is_poly_valid(polynomial)

    ::

        The random polynomial is:
         [3, {'cons': 0.3166178352704635, 'x_3': -0.30850205546468723, 'x_2': 0.1938147859698406,
         'x_2,x_3': 0.5722646368431439, 'x_1': 0.03709620256724111, 'x_1,x_3': 0.3273727561299321,
         'x_1,x_2': -0.4544731299546062, 'x_1,x_2,x_3': -0.1406736501192053}]
        The polynomial is valid.
    """
    # Check the validity of the input polynomial
    # Take a copy of the polynomial,
    # as we do not want to change the polynomial by the latter pop action of the 'cons' term
    poly_copy = deepcopy(poly)
    var_num = poly_copy[0]
    poly_dict = poly_copy[1]

    # Remove the cons term first and then check the variables
    # If there is no 'cons' term, it will do nothing
    poly_dict.pop('cons', None)
    # Obtain the dict keys
    keys_list = list(poly_dict.keys())

    # Extract all the input variables
    input_vars_list = []
    for key in keys_list:
        key_sep = key.split(',')
        # We do not allow input like {'x_1,x_1': 2}
        if len(list(set(key_sep))) != len(key_sep):
            print("The input polynomial contains at least one not valid monomial:" + str(key) +
                  ". Each key of input polynomial dictionary should only consist of different variables.")
            raise KeyError(key)
        input_vars_list += key_sep

    # Check the number of input variables
    input_vars_set = list(set(input_vars_list))
    if len(input_vars_set) != var_num:
        input_vars_set.sort()
        print("The polynomial variables are: " + str(input_vars_set) +
              ", but the expected number of input variables is: " + str(var_num) + ".")
        raise ValueError("the number of input variables ", var_num, " is not correct.")

    # Check the subscript of the variables
    std_vars_list = ['x_' + str(i) for i in range(1, var_num + 1)]
    input_diff_std = list(set(input_vars_set).difference(std_vars_list))
    if len(input_diff_std) != 0:
        input_vars_set.sort()
        print("The polynomial variables are: " + str(input_vars_set) +
              ", but the expected variables are: " + str(std_vars_list) + ".")
        raise ValueError("the subscript of input variable does not range from 1 to " + str(var_num) + ".")

    # If the input polynomial is a valid one
    print("The polynomial is valid.")


def dict_to_symbol(poly):
    r"""将用户输入的多项式字典处理成符号多项式。

    用户输入为以 ``x_i`` 为自变量的目标函数，第一步需要对目标函数进行处理，处理成 sympy 所接受的符号多项式，便于后续的变量替换和计算。

    Args:
        poly (list): 列表第一个元素为变量个数，第二个元素为用户输入的字典类型的多项式

    Returns:
        list: 列表第一个元素为变量个数，第二个元素为符号化的多项式

    代码示例:

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.pubo import dict_to_symbol
        var_num = 4
        poly_dict = {"cons": 0.5, 'x_1': 2, 'x_1,x_2': -2, 'x_2,x_3': -2, 'x_3,x_4': -2, 'x_4,x_1': -2}
        polynomial = [var_num, poly_dict]
        new_poly = dict_to_symbol(polynomial)
        print("The symbolized polynomial is: \n", new_poly)

    ::

        The symbolized polynomial is:  [4, -2*x_1*x_2 - 2*x_1*x_4 + 2*x_1 - 2*x_2*x_3 - 2*x_3*x_4 + 0.5]
    """
    var_num, poly_dict = poly

    # Transform the dict to a symbolized function
    poly_symbol = 0
    for key in poly_dict:
        value = poly_dict[key]
        if key == 'cons':
            poly_symbol += value
        else:
            key_sep = key.split(',')
            sym_var = 1
            for var in key_sep:
                sym_var *= symbols(var)
            poly_symbol += value * sym_var
    new_poly = [var_num, poly_symbol]

    return new_poly


def brute_force_search(poly):
    r"""用遍历的算法在解空间里暴力搜索 PUBO 问题的解，作为标准答案和其他算法的结果作比较。

    Args:
        poly (list): 列表第一个元素为变量个数，第二个元素为单项式构成的字典

    Returns:
        list: 列表第一个元素为 PUBO 问题的解，第二个元素为对应的目标函数的值

    代码示例:

    .. code-block:: python

        from paddle_quantum.mbqc.QAOA.pubo import brute_force_search
        n = 4
        func_dict = {"cons":0.5, 'x_1': 2, 'x_1,x_2': -2, 'x_2,x_3': -2, 'x_3,x_4': -2, 'x_4,x_1':-2}
        polynomial = [n, func_dict]
        opt = brute_force_search(polynomial)
        print("The optimal solution by brute force search is: ", opt[0])
        print("The optimal value by brute force search is: ", opt[1])

    ::

        The optimal solution by brute force search is: 1000
        The optimal value by brute force search is:  2.50000000000000
    """
    # Transform the dict of objective function to a symbolic function
    var_num, poly_symbol = dict_to_symbol(poly)
    feasible_values = []
    feasible_set = []

    # Scan the solution space
    for bit in range(2 ** var_num):
        feasible_solution = bin(bit)[2:].zfill(var_num)
        feasible_set += [feasible_solution]
        relation = {symbols('x_' + str(j + 1)): int(feasible_solution[j]) for j in range(var_num)}
        feasible_values += [poly_symbol.evalf(subs=relation)]
    opt_value = max(feasible_values)
    opt_solution = feasible_set[feasible_values.index(opt_value)]
    opt = [opt_solution, opt_value]

    return opt


def mbqc_pubo(OBJ_POLY, DEPTH, SEED, LR, ITR, EPOCH, SHOTS=1024):
    r"""定义 MBQC 模型下的 PUBO 主函数。

    选择 Adams 优化器，梯度下降算法最小化目标函数。

    Args:
        OBJ_POLY (list): 输入为以 x 为自变量的目标函数，列表第一个元素为变量个数，第二个元素为单项式构成的字典
        DEPTH (int): QAOA 算法的深度
        SEED (int): paddle 的训练种子
        LR (float): 学习率
        ITR (int): 单次轮回的迭代次数
        EPOCH (int): 轮回数
        SHOTS (int): 获得最终比特串时设定的测量次数

    Returns:
        list: 列表第一个元素为求得的最优解，第二个元素为对应的目标函数的值

    代码示例:

    .. code-block:: python

        from pubo import mbqc_pubo
        n = 4
        poly_dict = {'x_1': 2, 'x_2': 2, 'x_3': 2, 'x_4': 2, 'x_1,x_2': -2, 'x_2,x_3': -2, 'x_3,x_4': -2, 'x_4,x_1': -2}
        polynomial = [n, poly_dict]
        mbqc_opt = mbqc_pubo(OBJ_POLY=polynomial, DEPTH=4, SEED=1024, LR=0.1, ITR=120, EPOCH=1, shots=1024)
        print("The optimal solution by MBQC is: ", mbqc_opt[0])
        print("The optimal value by MBQC is: ", mbqc_opt[1])

    ::

        QAOA Ansatz depth is: 4
        iter: 10   loss_MBQC: -3.8231
        iter: 20   loss_MBQC: -3.9038
        iter: 30   loss_MBQC: -3.9840
        iter: 40   loss_MBQC: -3.9970
        iter: 50   loss_MBQC: -3.9990
        iter: 60   loss_MBQC: -3.9993
        iter: 70   loss_MBQC: -3.9997
        iter: 80   loss_MBQC: -3.9999
        iter: 90   loss_MBQC: -4.0000
        iter: 100   loss_MBQC: -4.0000
        iter: 110   loss_MBQC: -4.0000
        iter: 120   loss_MBQC: -4.0000
        MBQC running time is:  16.864049434661865
        Optimal parameter gamma:  [3.15639021 0.23177807 4.99173672 0.69199477]
        Optimal parameter beta:  [0.13486116 2.22551912 5.10371187 2.4004731 ]
        The optimal solution by MBQC is: 0101
        The optimal value by MBQC is:  4.00000000000000
    """
    obj_poly = dict_to_symbol(OBJ_POLY)
    var_num, poly_symbol = obj_poly

    # Initialize
    print("QAOA Ansatz depth is:", DEPTH)

    # Initialize MBQC - PUBO optimization net
    start_time = perf_counter()

    seed(SEED)
    mbqc_net = MBQC_QAOA_Net(DEPTH)
    # Choose Adams optimizer (or SGD optimizer)
    opt = optimizer.Adam(learning_rate=LR, parameters=mbqc_net.parameters())
    # opt = optimizer.SGD(learning_rate = LR, parameters = mbqc_net.parameters())

    # Start training
    for epoch in range(EPOCH):
        # Update parameters for each iter
        for itr in range(1, ITR + 1):
            # Train with mbqc_net and return the loss
            loss, state_out = mbqc_net(poly=obj_poly)
            # Propagate loss backwards and optimize the parameters
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
            if itr % 10 == 0:
                print("iter:", itr, "  loss_MBQC:", "%.4f" % loss.numpy())

    end_time = perf_counter()
    print("MBQC running time is: ", end_time - start_time)

    # Print the optimization parameters
    print("Optimal parameter gamma: ", mbqc_net.gamma.numpy())
    print("Optimal parameter beta: ", mbqc_net.beta.numpy())

    solu_str = get_solution_string(state_out, SHOTS)

    # Evaluate the corresponding value
    relation = {symbols('x_' + str(j + 1)): int(solu_str[j]) for j in range(var_num)}
    value = poly_symbol.evalf(subs=relation)
    # Return the solution and its corresponding value
    opt = [solu_str, value]

    return opt
