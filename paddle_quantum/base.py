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
The basic function of the paddle quantum.
"""

import paddle
import paddle_quantum
from typing import Union, Optional

DEFAULT_DEVICE = 'cpu'
DEFAULT_SIMULATOR = paddle_quantum.Backend.StateVector
DEFAULT_DTYPE = 'complex64'


def set_device(device: 'str') -> None:
    r"""Set the device to save the tensor.

    Args:
        device: The name of the device.
    """  
    global DEFAULT_DEVICE
    DEFAULT_DEVICE = device
    paddle.set_device(device)


def get_device() -> 'str':
    r"""Get the current device to save the tensor.

    Returns:
        The name of the current device.
    """
    return DEFAULT_DEVICE


def set_backend(backend: Union[str, paddle_quantum.Backend]) -> None:
    r"""Set the backend implementation of paddle quantum.

    Args:
        backend: The name of the backend.
    """
    global DEFAULT_SIMULATOR
    if isinstance(backend, str):
        DEFAULT_SIMULATOR = paddle_quantum.Backend(backend)
    else:
        DEFAULT_SIMULATOR = backend


def get_backend() -> paddle_quantum.Backend:
    r"""Get the current backend of paddle quantum.

    Returns:
        The name of currently used backend.
    """
    return DEFAULT_SIMULATOR


def set_dtype(dtype: 'str') -> None:
    r"""Set the data type .

    Args:
        dtype: The dtype can be ``complex64`` and ``complex128``. 

    Raises:
        ValueError: The dtype should be complex64 or complex128.
    """
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype
    if dtype == 'complex64':
        paddle.set_default_dtype('float32')
    elif dtype == 'complex128':
        paddle.set_default_dtype('float64')
    else:
        raise ValueError("The dtype should be complex64 or complex128.")


def get_dtype() -> 'str':
    r"""Return currently used data type.

    Returns:
        Currently used data type.
    """
    return DEFAULT_DTYPE


class Operator(paddle.nn.Layer):
    r"""The basic class to implement the quantum operation.

    Args:
        backend: The backend implementation of the operator.
            Defaults to ``None``, which means to use the default backend implementation.
        dtype: The data type of the operator.
            Defaults to ``None``, which means to use the default data type.
        name_scope: Prefix name used by the operator to name parameters. Defaults to ``None``.
    """
    def __init__(self, backend: Optional[paddle_quantum.Backend] = None, dtype: Optional[str] = None,
                 name_scope: Optional[str] = None):
        if dtype is None:
            super().__init__(name_scope)
        else:
            super().__init__(name_scope, dtype)
        self.dtype = dtype if dtype is not None else get_dtype()

        if backend == paddle_quantum.Backend.StateVector:
            self.backend = backend
        elif backend == paddle_quantum.Backend.DensityMatrix:
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()

    def to(self, backend: Optional[paddle_quantum.Backend] = None, device: Optional[str] = None,
           dtype: Optional[str] = None, blocking: Optional[str] = None):
        super().to(device, dtype, blocking)
        if backend is not None:
            self.backend = backend
            for sub_layer in self.children():
                sub_layer.backend = backend

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError
