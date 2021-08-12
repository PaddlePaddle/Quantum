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
Custom optimizer
"""

from abc import ABC, abstractmethod

class CustomOptimizer(ABC):
    r"""所有 SciPy 优化器的基类。

    定义了在用 SciPy 优化器优化时所需的基本功能，如计算期望值和计算梯度的函数。

    Attributes:
        cir (UAnsatz): 带可训练参数的量子电路
        hamiltonian (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
        shots (int): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值
        grad_func_name (string, optional): 用来计算梯度的函数的函数名，可以选择 'linear_comb'、'finite_diff' 或 'param_shift'。
            只有特定需要梯度的 optimizer 如 ``ConjugateGradient`` 需要，默认为 ``None``
        delta (float, optional): 差分法中的 delta，默认为 0.01
    """
   

    def __init__(self, cir, hamiltonian, shots, grad_func_name=None, delta=0.01):
        r"""``CustomOptimizer`` 的构造函数。

        Args:
            cir (UAnsatz): 带可训练参数的量子电路
            hamiltonian (list or Hamiltonian): 记录哈密顿量信息的列表或 Hamiltonian 类的对象
            shots (int): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值
            grad_func_name (string, optional): 用来计算梯度的函数的函数名，可以选择 'linear_comb'、'finite_diff' 或 'param_shift'。
                只有特定需要梯度的 optimizer 如 ``ConjugateGradient`` 需要，默认为 ``None``
            delta (float, optional): 差分法中的 delta，默认为 0.01
        """
        self.cir = cir
        self.grad_func_name = grad_func_name
        self.hamiltonian = hamiltonian
        self.shots = shots
        self.delta = delta
        self.loss_func = self._get_expec_val_scipy

        self.grad_func = None
        if self.grad_func_name == 'linear_comb':
            self.grad_func = self._linear_combinations_gradient_scipy
        elif self.grad_func_name == 'finite_diff':
            self.grad_func = self._finite_difference_gradient_scipy
        elif self.grad_func_name == 'param_shift':
            self.grad_func = self._param_shift_gradient_scipy
        else:
            assert self.grad_func_name == None, \
                "grad_func_name should be None or one of 'linear_comb', 'finite_diff', 'param_shift'"

    def _get_expec_val_scipy(self, theta, cir, H, shots=0):
        r"""计算关于哈密顿量 H 的期望的理论值。

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        cir.update_param(theta)
        return cir.expecval(H, shots).numpy()
    
    def _linear_combinations_gradient_scipy(self, theta, cir, H, shots):
        r"""用 linear combination 的方法计算参数的梯度。

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        grad = cir.linear_combinations_gradient(H, shots)
        return grad
    
    def _finite_difference_gradient_scipy(self, theta, cir, H, shots):
        r"""用差分法计算参数的梯度。

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        grad = cir.finite_difference_gradient(H, self.delta, shots)
        return grad
    
    def _param_shift_gradient_scipy(self, theta, cir, H, shots):
        r"""用 parameter shift 的方法计算参数的梯度。

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        grad = cir.param_shift_gradient(H, shots)
        return grad

    @abstractmethod
    def minimize(self, iterations):
        r"""最小化给定的损失函数。

        Args:
            iterations (int): 迭代的次数
        """
        raise NotImplementedError
