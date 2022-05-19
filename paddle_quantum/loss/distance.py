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
The source file of the class for the distance.
"""

import paddle
import paddle_quantum


class TraceDistance(paddle_quantum.Operator):
    r"""The class of the loss function to compute the trace distance.

    This interface can make you using the trace distance as the loss function.

    Args:
        target_state: The target state to be used to compute the trace distance.

    """
    def __init__(self, target_state: paddle_quantum.State):
        super().__init__()
        self.target_state = target_state

    def forward(self, state: paddle_quantum.State) -> paddle.Tensor:
        r"""Compute the trace distance between the input state and the target state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the trace distance with the target state.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The trace distance between the input state and the target state.
        """
        if self.backend == paddle_quantum.Backend.StateVector:
            state = state.data
            state = paddle.unsqueeze(state, axis=1)
            target_state = paddle.unsqueeze(self.target_state.data, axis=1)
            target_state_dagger = paddle.conj(paddle.t(target_state))
            fidelity = paddle.abs(paddle.matmul(target_state_dagger, state))
            distance = paddle.sqrt(1 - paddle.square(fidelity))
            distance = paddle.reshape(distance, [1])
        elif self.backend == paddle_quantum.Backend.DensityMatrix:
            _matrix = state.data - self.target_state.data
            _matrix = paddle.matmul(paddle.conj(paddle.t(_matrix)), _matrix)
            eigenvalues = paddle.linalg.eigvalsh(_matrix).abs()
            distance = paddle.sum(paddle.sqrt(eigenvalues)) * 0.5
        else:
            raise NotImplementedError
        return distance


class StateFidelity(paddle_quantum.Operator):
    r"""The class of the loss function to compute the state fidelity.

    This interface can make you using the state fidelity as the loss function.

    Args:
        target_state: The target state to be used to compute the state fidelity.
    """
    def __init__(self, target_state: paddle_quantum.State):
        super().__init__()
        self.target_state = target_state

    def forward(self, state: paddle_quantum.State) -> paddle.Tensor:
        r"""Compute the state fidelity between the input state and the target state.

        The value computed by this function can be used as a loss function to optimize.

        Args:
            state: The input state which will be used to compute the state fidelity with the target state.

        Raises:
            NotImplementedError: The backend is wrong or not supported.

        Returns:
            The state fidelity between the input state and the target state.
        """
        if self.backend == paddle_quantum.Backend.StateVector:
            state = state.data
            state = paddle.unsqueeze(state, axis=1)
            target_state = paddle.unsqueeze(self.target_state.data, axis=1)
            target_state_dagger = paddle.conj(paddle.t(target_state))
            fidelity = paddle.abs(paddle.matmul(target_state_dagger, state))
            fidelity = paddle.reshape(fidelity, [1])
        elif self.backend == paddle_quantum.Backend.DensityMatrix:
            target_state = self.target_state.data
            state = state.data
            target_state_eig_vals, target_state_eig_vec = paddle.linalg.eigh(target_state)
            target_state_sqrt = paddle.matmul(
                paddle.matmul(target_state_eig_vec, paddle.diag(paddle.sqrt(paddle.abs(target_state_eig_vals)))),
                paddle.conj(paddle.t(target_state_eig_vec))
            )
            _matrix = paddle.matmul(target_state_sqrt, paddle.matmul(state, target_state_sqrt))
            eig_vals = paddle.linalg.eigvalsh(_matrix)
            fidelity = paddle.sum(paddle.sqrt(paddle.abs(eig_vals)))
        else:
            raise NotImplementedError
        return fidelity
