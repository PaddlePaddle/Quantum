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
Newton-CG optimizer
"""

from scipy import optimize
from .custom_optimizer import CustomOptimizer


class NewtonCG(CustomOptimizer):
    r"""Newton-CG Optimizer
    
    继承 ``CustomOptimizer`` 类，使用 SciPy 里 Newton conjugate gradient 的方法优化。此优化器需要说明计算梯度的方法。
    
    Attributes:
        cir (UAnsatz): 带可训练参数的量子电路
        hamiltonian (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
        shots (int): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值
        grad_func_name (string): 用来计算梯度的函数的函数名，可以选择 'linear_comb'、'finite_diff' 或 'param_shift'，默认为 ``None``
        delta (float): 差分法中的 delta，默认为 0.01
    """

    def __init__(self, cir, hamiltonian, shots, grad_func_name=None, delta=0.01):
        r"""``NewtonCG`` 的构造函数。

        Args:
            cir (UAnsatz): 带可训练参数的量子电路
            hamiltonian (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
            shots (int): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值
            grad_func_name (string): 用来计算梯度的函数的函数名，可以选择 'linear_comb'、'finite_diff' 或 'param_shift'，默认为 ``None``
            delta (float): 差分法中的 delta，默认为 0.01
        """
        super().__init__(cir, hamiltonian, shots, grad_func_name, delta)

    def minimize(self, iterations):
        r"""最小化给定的损失函数。

        Args:
            iterations (int): 迭代的次数
        """
        opt_res = optimize.minimize(
            self.loss_func,
            self.cir.get_param().numpy(),
            args=(self.cir, self.hamiltonian, self.shots),
            method='Newton-CG',
            jac=self.grad_func,
            options={'maxiter': iterations},
            callback=lambda xk: print('loss: ', self.loss_func(xk, self.cir, self.hamiltonian, self.shots))
        )
        print(opt_res.message)
