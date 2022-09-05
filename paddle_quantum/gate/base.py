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
The source file of the basic class for the quantum gates.
"""

import paddle
import paddle_quantum
from typing import Union, List, Iterable
from ..intrinsic import _get_float_dtype
from math import pi


class Gate(paddle_quantum.Operator):
    r"""Base class for quantum gates.

    Args:
        depth: Number of layers. Defaults to 1.
        backend: Backend on which the gates are executed. Defaults to None.
        dtype: Type of data. Defaults to None.
        name_scope: Prefix name used by the layer to name parameters. If prefix is "my_layer", parameter name in
            MyLayer can be "my_layer_0.w_n", where "w" is the parameter base name and "n" is an unique suffix
            auto-generated. If ``None``, prefix name will be snake cased class name. Defaults to ``None``.
    """
    def __init__(
            self, depth: int = 1, backend: paddle_quantum.Backend = None, dtype: str = None, name_scope: str = None
    ):
        super().__init__(backend, dtype, name_scope)
        self.depth = depth
        self.gate_name = None

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, paddle_quantum.Operator):
            if value.backend is None:
                value.backend = paddle_quantum.get_backend() if self.backend is None else self.backend
            if value.dtype is None:
                value.dtype = paddle_quantum.get_dtype() if self.dtype is None else self.dtype

    def gate_history_generation(self) -> None:
        r""" determine self.gate_history
        
        """
        gate_history = []
        for _ in range(0, self.depth):
            for qubit_idx in self.qubits_idx:
                gate_info = {'gate': self.gate_name, 'which_qubits': qubit_idx, 'theta': None}
                gate_history.append(gate_info)
        self.gate_history = gate_history


class ParamGate(Gate):
    r""" Base class for quantum parameterized gates
    
    """
    def theta_generation(self, param: Union[paddle.Tensor, float, List[float]], param_shape: List[int]) -> None:
        """ determine self.theta, and create parameter if necessary

        Args:
            param: input theta
            param_shape: shape for theta
            
        Note:
            in the following cases ``param`` will be transformed to a parameter:
                - ``param`` is None
            in the following cases ``param`` will be added to the parameter list:
                - ``param`` is ParamBase
            in the following cases ``param`` will keep unchange:
                - ``param`` is a Tensor but not a parameter
                - ``param`` is a float or list of floats
            
        """
        
        float_dtype = _get_float_dtype(self.dtype)
        
        if param is None:
            theta = self.create_parameter(
                shape=param_shape, dtype=float_dtype,
                default_initializer=paddle.nn.initializer.Uniform(low=0, high=2 * pi)
            )
            self.add_parameter('theta', theta)

        elif isinstance(param, paddle.fluid.framework.ParamBase):
            assert param.shape == param_shape, "received: " + str(param.shape) + " expect: " + str(param_shape)
            self.add_parameter('theta', param)
            
        elif isinstance(param, paddle.Tensor):
            param = param.reshape(param_shape)
            self.theta = param
            
        elif isinstance(param, float):
            self.theta = paddle.ones(param_shape, dtype=float_dtype) * param
        
        else: # when param is a list of float
            self.theta = paddle.to_tensor(param, dtype=float_dtype).reshape(param_shape)

    def gate_history_generation(self) -> None:
        r""" determine self.gate_history when gate is parameterized
        
        """
        gate_history = []
        for depth_idx in range(0, self.depth):
            for idx, qubit_idx in enumerate(self.qubits_idx):
                if self.param_sharing:
                    param = self.theta[depth_idx]
                else:
                    param = self.theta[depth_idx][idx]
                gate_info = {'gate': self.gate_name, 'which_qubits': qubit_idx, 'theta': param}
                gate_history.append(gate_info)
        self.gate_history = gate_history
