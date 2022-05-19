# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The source file of the class for the special quantum operator.
"""

import random
import paddle
import paddle_quantum
from ..base import Operator
from typing import Union, Iterable


class ResetState(Operator):
    r"""The class to reset the quantum state. It will be implemented soon.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **kwargs):
        r"""The forward function.

        Returns:
            NotImplemented.
        """
        return NotImplemented


class PartialState(Operator):
    r"""The class to obtain the partial quantum state. It will be implemented soon.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *inputs, **kwargs):
        r"""The forward function.

        Returns:
            NotImplemented.
        """
        return NotImplemented


class Collapse(Operator):
    r"""The class to compute the collapse of the quantum state.

    Args:
        measure_basis: The basis of the measurement. The quantum state will collapse to the corresponding eigenstate.

    Raises:
        NotImplementedError: If the basis of measurement is not z. Other bases will be implemented soon.
    """
    def __init__(self, measure_basis: Union[Iterable[paddle.Tensor], str]):
        super().__init__()
        self.measure_basis = []
        if measure_basis == 'z' or measure_basis == 'computational_basis':
            basis0 = paddle.to_tensor([[1.0, 0], [0, 0]])
            basis1 = paddle.to_tensor([[0.0, 0], [0, 1]])
            self.measure_basis.append(basis0)
            self.measure_basis.append(basis1)
        else:
            raise NotImplementedError

    def forward(self, state: paddle_quantum.State, desired_result: Union[int, str]) -> paddle_quantum.State:
        r"""Compute the collapse of the input state.

        Args:
            state: The input state, which will be collapsed.
            desired_result: The desired result you want to collapse.

        Raises:
            NotImplementedError: Currently we just support the z basis.

        Returns:
            The collapsed quantum state.
        """
        if self.backend == paddle_quantum.Backend.StateVector:
            if desired_result == 'random':
                prob_list = []
                idx_list = list(range(0, len(self.measure_basis)))
                for idx in idx_list:
                    measure_op = self.measure_basis[idx]
                    state = paddle.unsqueeze(state.data, axis=1)
                    _prob = paddle.matmul(measure_op, state.data)
                    prob = paddle.matmul(paddle.conj(paddle.t(_prob)), _prob).item()
                    prob_list.append(prob)
                    desired_result = random.choices(idx_list, prob_list)
            measure_op = self.measure_basis[desired_result]
            state = paddle.unsqueeze(state.data, axis=1)
            _prob = paddle.matmul(measure_op, state.data)
            prob = paddle.matmul(paddle.conj(paddle.t(_prob)), _prob)
            prob = paddle.reshape(prob, [1])
            state = paddle.matmul(measure_op, state) / paddle.sqrt(prob)
            measured_state = paddle_quantum.State(state, backend=self.backend)
        elif self.backend == paddle_quantum.Backend.DensityMatrix:
            if desired_result == 'random':
                prob_list = []
                idx_list = list(range(0, len(self.measure_basis)))
                for idx in idx_list:
                    measure_op = self.measure_basis[idx]
                    state = paddle.unsqueeze(state.data, axis=1)
                    measure_op_dagger = paddle.conj(paddle.t(measure_op))
                    prob = paddle.trace(paddle.matmul(paddle.matmul(measure_op_dagger, measure_op), state)).item()
                    prob_list.append(prob)
                    desired_result = random.choices(idx_list, prob_list)
            measure_op = self.measure_basis[desired_result]
            state = state.data
            measure_op_dagger = paddle.conj(paddle.t(measure_op))
            prob = paddle.trace(paddle.matmul(paddle.matmul(measure_op_dagger, measure_op), state))
            state = paddle.matmul(paddle.matmul(measure_op, state), measure_op_dagger) / prob
            measured_state = paddle_quantum.State(state)
        else:
            raise NotImplementedError
        return measured_state
