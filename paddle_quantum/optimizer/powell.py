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
Powell optimizer
"""

from scipy import optimize
from .custom_optimizer import CustomOptimizer


class Powell(CustomOptimizer):
    r"""Powell Optimizer
    
    继承 ``CustomOptimizer`` 类，使用 SciPy 里 Powell 方法优化。该方法不需要传入计算 gradient 的方式。

    Attributes:
        cir (UAnsatz): 带可训练参数的量子电路
        hamiltonian (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
        shots (int): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值
    """

    def __init__(self, cir, hamiltonian, shots):
        r"""``Powell`` 的构造函数。

        Args:
            cir (UAnsatz): 带可训练参数的量子电路
            hamiltonian (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
            shots (int): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值

        """
        super().__init__(cir, hamiltonian, shots)
        
    def minimize(self, iterations):
        r"""最小化给定的损失函数。

        Args:
            iterations (int): 迭代的次数
        """
        opt_res = optimize.minimize(
            self.loss_func,
            self.cir.get_param().numpy(),
            args=(self.cir, self.hamiltonian, self.shots),
            method='Powell',
            options={'maxiter': iterations},
            callback=lambda xk: print('loss: ', self.loss_func(xk, self.cir, self.hamiltonian, self.shots))
        )
        print(opt_res.message)
