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
The source file of the basic class for the quantum channels.
"""

from typing import Any, Optional, List

import paddle
from ..base import Operator, get_backend, get_dtype
from ..backend import Backend


class Channel(Operator):
    r"""Basic class for quantum channels.

    Args:
        backend: Backend on which the channel is executed. Defaults to ``None``.
        dtype: Type of data. Defaults to ``None``.
        name_scope: Prefix name used by the layer to name parameters. If prefix is "my_layer", parameter name in
            MyLayer can be "my_layer_0.w_n", where "w" is the parameter base name and "n" is an unique suffix
            auto-generated. If ``None``, prefix name will be snake cased class name. Defaults to ``None``.
    """
    def __init__(
            self, backend: Backend = None, dtype: str = None, name_scope: str = None
    ) -> None:
        super().__init__(backend, dtype, name_scope)

    def to(self, backend: Optional[str] = None, device: Optional[str] = None, 
           dtype: Optional[str] = None, blocking: Optional[str] = None) -> None:
        super().to(device, dtype, blocking)
        # TODO: to be implemented
        raise NotImplementedError

    def forward(self, *inputs: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Channel):
            if value.backend is None:
                value.backend = get_backend() if self.backend is None else self.backend
            if value.dtype is None:
                value.dtype = get_dtype() if self.dtype is None else self.dtype

    @property
    def choi_repr(self) -> paddle.Tensor:
        r"""Choi representation of a channel
        
        Returns:
            a tensor with shape :math:`[d_\text{out}^2, d_\text{in}^2]`, where :math:`d_\text{in/out}` is the input/output 
            dimension of this channel
        
        Raises:
            Cannot return the Choi representation of a general channel: use ChoiRepr instead to specify your channel
        
        """
        raise NotImplementedError(
            "Cannot return the Choi representation of a general channel: use ChoiRepr instead to specify your channel")
    
    @property
    def kraus_repr(self) -> List[paddle.Tensor]:
        r"""Kraus representation of a channel
        
        Returns:
            a list of tensors with shape :math:`[d_\text{out}, d_\text{in}]`, where :math:`d_\text{in/out}` is the input/output 
            dimension of this channel
        
        Raises:
            Cannot return the Kraus representation of a general channel: use KrausRepr instead to specify your channel
        
        """
        raise NotImplementedError(
            "Cannot return the Kraus representation of a general channel: use KrausRepr instead to specify your channel")
    
    @property
    def stinespring_repr(self) -> paddle.Tensor:
        r"""Stinespring representation of a channel
        
        Returns:
            a tensor with shape :math:`[r * d_\text{out}, d_\text{in}]`, where :math:`r` is the rank of this channel and 
            :math:`d_\text{in/out}` is the input/output dimension of this channel
        
        Raises:
            Cannot return the Stinespring representation of a general channel: use StinespringRepr instead to specify your channel
        
        """
        raise NotImplementedError(
            "Cannot return the Stinespring representation of a general channel: use StinespringRepr instead to specify your channel")
