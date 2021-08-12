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
ExpecVal Class
"""

import paddle
from paddle.autograd import PyLayer

__all__ = [
    "ExpecVal"
]


class ExpecVal(PyLayer):
    r"""PaddlePaddle 自定义 Python 算子，用来计算量子电路输出的量子态关于可观测量 H 的期望值。
    """

    @staticmethod
    def forward(ctx, cir, theta, grad_func, hamiltonian, delta=None, shots=0):
        r"""前向函数。

        Hint:
            如果想输入的可观测量的矩阵为 :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I` 。则 ``H`` 的 ``list`` 形式为 ``[[0.7, 'Z0, X1'], [0.2, 'Z1']]`` 。

        Args:
            cir (UAnsatz): 目标量子电路
            theta (paddle.Tensor): 量子电路中的需要被优化的参数
            grad_func (string): 用于计算梯度的函数名，应为 ``'finite_diff'`` 或 ``'param_shift'``
            hamiltonian (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
            delta (float): 差分法中需要用到的 delta，默认为 ``None``
            shots (int, optional): 表示测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值

        Returns:
            paddle.Tensor: 量子电路输出的量子态关于可观测量 H 的期望值

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.expecval import ExpecVal

            N = 2
            D = 2
            theta = paddle.uniform(shape=[N * D], dtype='float64', min=0.0, max=np.pi * 2)
            theta.stop_gradient = False
            cir = UAnsatz(N)
            cir.real_entangled_layer(theta, D)
            cir.run_state_vector()

            H = [[1.0, 'z0,z1']]
            delta = 0.01
            shots = 0

            z = ExpecVal.apply(cir, cir.get_param(), 'finite_diff', H, delta, shots)
            print(z)

        ::

            Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=False,
                   [0.61836319])
        """
        assert grad_func in {'finite_diff', 'param_shift'}, "grad_func must be one of 'finite_diff' or 'param_shift'"
        # Pass grad_func, cir, theta, delta, shots, and Hamiltonian into the backward function by adding temporary attributes
        ctx.grad_func = grad_func
        ctx.cir = cir
        ctx.theta = theta
        ctx.delta = delta
        ctx.shots = shots
        ctx.Hamiltonian = hamiltonian

        # Compute the expectation value
        cir.update_param(theta)
        expec_val = cir.expecval(ctx.Hamiltonian, shots)

        return expec_val

    @staticmethod
    def backward(ctx, dy):
        r"""反向函数。

        Args:
            dy (paddle.Tensor): 前向函数输出的期望值的梯度

        Returns:
            paddle.Tensor: 前向函数中输入的参数 ``theta`` 的梯度

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.expecval import ExpecVal

            N = 2
            D = 2
            theta = paddle.uniform(shape=[N * D], dtype='float64', min=0.0, max=np.pi * 2)
            theta.stop_gradient = False
            cir = UAnsatz(N)
            cir.real_entangled_layer(theta, D)
            cir.run_state_vector()

            H = [[1.0, 'z0,z1']]
            delta = 0.01
            shots = 0

            z = ExpecVal.apply(cir, cir.get_param(), 'finite_diff', H, delta, shots)
            temp = paddle.square(z)
            temp.backward()
        """
        # Get expec_func, grad_func, theta, delta, and args
        cir = ctx.cir
        grad_func = ctx.grad_func
        delta = ctx.delta
        shots = ctx.shots
        Hamiltonian = ctx.Hamiltonian

        # Compute the gradient
        if grad_func == "finite_diff":
            assert delta is not None, "Finite difference gradient requires an input 'delta'"
            grad = dy * cir.finite_difference_gradient(Hamiltonian, delta, shots)
        else:
            grad = dy * cir.param_shift_gradient(Hamiltonian, shots)
        grad.stop_gradient = False

        return paddle.reshape(grad, ctx.theta.shape)
