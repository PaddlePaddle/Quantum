# !/usr/bin/env python3
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

r"""
The source file of the LoccState class.
"""

import math
import paddle
import paddle_quantum
from ..intrinsic import _get_float_dtype
from typing import Optional


class LoccState(paddle_quantum.State):
    r"""An LOCC state in LOCCNet.

    We care not only about the quantum state in LOCC, but also about the probability of getting it and
    the corresponding measurement result. Therefore, this class contains three attributes:
    quantum state ``data``, the probability of getting this state ``prob``, and the measurement result leading
    to this state, namely ``measured_result``.

    Args:
        data: Matrix form of the quantum state of this ``LoccState``. Defaults to ``None``.
        prob: Probability of getting this quantum state.. Defaults to ``None``.
        measured_result: Measurement result leading to this quantum state. Defaults to ``None``.
        num_qubits: Total number of qubits. Defaults to ``None``.
        backend: Backend of Paddle Quantum. Defaults to ``None``.
        dtype: Type of data. Defaults to ``None``.
    """

    def __init__(
        self,
        data: paddle.Tensor = None,
        prob: paddle.Tensor = None,
        measured_result: str = None,
        num_qubits: Optional[int] = None,
        backend: Optional[paddle_quantum.Backend] = None,
        dtype: Optional[str] = None,
    ):
        if data is None and prob is None and measured_result is None:
            self.data = paddle.to_tensor([1], dtype=paddle_quantum.get_dtype())
            self.prob = paddle.to_tensor(
                [1], dtype=_get_float_dtype(paddle_quantum.get_dtype())
            )
            self.measured_result = ""
            self.num_qubits = 0
        else:
            self.data = data
            self.prob = prob
            self.measured_result = measured_result
            if num_qubits is None:
                self.num_qubits = int(math.log2(data.shape[-1]))
            else:
                self.num_qubits = num_qubits
        backend = paddle_quantum.get_backend() if backend is None else backend
        assert backend == paddle_quantum.Backend.DensityMatrix
        self.backend = backend
        self.dtype = dtype if dtype is not None else paddle_quantum.get_dtype()

    def clone(self) -> "LoccState":
        r"""Create a copy of the current object.

        Returns:
            A copy of the current object.
        """
        return LoccState(
            self.data,
            self.prob,
            self.measured_result,
            self.num_qubits,
            self.backend,
            self.dtype,
        )

    def __getitem__(self, item):
        if item == 0:
            return self.data
        if item == 1:
            return self.prob
        if item == 2:
            return self.measured_result
        raise ValueError("too many values to unpack (expected 3)")

    def __repr__(self):
        return (
            f"state: {self.data.numpy()}\n"
            f"prob: {self.prob.numpy()[0]}\n"
            f"measured_result: {self.measured_result}"
        )

    def __str__(self):
        return (
            f"state: {self.data.numpy()}\n"
            f"prob: {self.prob.numpy()[0]}\n"
            f"measured_result: {self.measured_result}"
        )


LoccStatus = LoccState
