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

import copy
import math
import matplotlib
import warnings
from typing import Union, Iterable, Any, List, Callable

import paddle
import paddle_quantum as pq
from .functional.base import simulation
from .functional.visual import _base_gate_display, _base_param_gate_display
from ..backend import Backend
from ..base import get_backend, get_dtype, Operator
from ..channel import Channel
from ..channel.base import _kraus_to_choi, _kraus_to_stinespring
from ..intrinsic import _get_float_dtype, _format_param_shape, _format_qubits_idx


class Gate(Channel):
    r"""Base class for quantum gates.

    Args:
        matrix: the matrix of this gate. Defaults to ``None`` i.e. not specified.
        qubits_idx: indices of the qubits on which this gate acts on. Defaults to ``None``.
            i.e. list(range(num_acted_qubits)).
        depth: Number of layers. Defaults to ``1``.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit. 
        Defaults to ``None``.
        num_qubits: total number of qubits. Defaults to ``None``.
        check_legality: whether check the completeness of the matrix if provided. Defaults to ``True``.
        num_acted_qubits: the number of qubits that this gate acts on. Defaults to ``None``.
        backend: Backend on which the gates are executed. Defaults to ``None``.
        dtype: Type of data. Defaults to ``None``.
        name_scope: Prefix name used by the layer to name parameters. If prefix is "my_layer", parameter name in
        MyLayer can be "my_layer_0.w_n", where "w" is the parameter base name and "n" is an unique suffix
        auto-generated. If None, prefix name will be snake cased class name. Defaults to ``None``.
    """

    def __init__(
            self, matrix: paddle.Tensor = None, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None,
            depth: int = 1, gate_info: dict = None, num_qubits: int = None,
            check_legality: bool = True, num_acted_qubits: int = None,
            backend: pq.Backend = None, dtype: str = None, name_scope: str = None
    ):
        Operator.__init__(self, backend=backend,
                          dtype=dtype, name_scope=name_scope)
        self.type_repr = 'Gate'
        assert matrix is not None or num_acted_qubits is not None, (
            "Received None for matrix and num_acted_qubits: "
            "either one of them must be specified to initialize a Gate instance.")
        if matrix is not None:
            num_acted_qubits = self.__gate_init(matrix, check_legality)
        if qubits_idx is None:
            self.qubits_idx = [0] if num_acted_qubits == 1 else [
                list(range(num_acted_qubits))]
        else:
            self.qubits_idx = _format_qubits_idx(
                qubits_idx, num_qubits, num_acted_qubits)
        self.num_acted_qubits = num_acted_qubits
        self.num_qubits = num_qubits
        self.depth = depth
        default_gate_info = {
            'gatename': 'O',
            'texname': r'$O$',
            'plot_width': 0.9,
        }
        if gate_info is not None:
            default_gate_info.update(gate_info)
        self.gate_info = default_gate_info
        self.__matrix = matrix if matrix is None else matrix.cast(self.dtype)

    def __gate_init(self, gate_matrix: paddle.Tensor, check_legality: bool) -> int:
        r"""Initialize channel for type_repr as ``'Gate'``
        """
        complex_dtype = self.dtype
        gate_matrix = gate_matrix.cast(complex_dtype)

        if not isinstance(gate_matrix, paddle.Tensor):
            raise TypeError(
                f"Unexpected data type for quantum gate: expected paddle.Tensor, received {type(gate_matrix)}")

        dimension = gate_matrix.shape[0]

        if check_legality:
            err = paddle.norm(
                paddle.abs(gate_matrix @ paddle.conj(gate_matrix.T) -
                           paddle.cast(paddle.eye(dimension), complex_dtype))
            ).item()
            if err > min(1e-6 * dimension, 0.01):
                warnings.warn(
                    f"\nThe input gate matrix may not be a unitary: norm(U * U^d - I) = {err}.", UserWarning)
        num_acted_qubits = int(math.log2(dimension))

        self.__kraus_repr = [gate_matrix]
        return num_acted_qubits

    @property
    def matrix(self) -> paddle.Tensor:
        r"""Unitary matrix of this gate

        Raises:
            ValueError: Need to specify the matrix form in this Gate instance.

        """
        if self.__matrix is None:
            raise ValueError(
                "Need to specify the matrix form in this Gate instance.")
        return self.__matrix

    @property
    def choi_repr(self) -> paddle.Tensor:
        return _kraus_to_choi(self.kraus_repr)

    @property
    def kraus_repr(self) -> List[paddle.Tensor]:
        return [self.matrix]

    @property
    def stinespring_repr(self) -> paddle.Tensor:
        return _kraus_to_stinespring(self.kraus_repr)

    def gate_history_generation(self) -> None:
        r""" determine self.gate_history

        """
        gate_history = []
        for _ in range(self.depth):
            for qubit_idx in self.qubits_idx:
                gate_info = {
                    'gate': self.gate_info['gatename'], 'which_qubits': qubit_idx, 'theta': None}
                gate_history.append(gate_info)
        self.gate_history = gate_history

    def set_gate_info(self, **kwargs: Any) -> None:
        r'''the interface to set `self.gate_info`

        Args:
            kwargs: parameters to set `self.gate_info`
        '''
        self.gate_info.update(kwargs)

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float) -> float:
        r'''The display function called by circuit instance when plotting.

        Args:
            ax: the ``matplotlib.axes.Axes`` instance
            x: the start horizontal position

        Returns:
            the total width occupied

        Note:
            Users could overload this function for custom display.
        '''
        return _base_gate_display(self, ax, x)

    def _single_qubit_combine_with_threshold6(self, state: pq.State, matrices: List[paddle.Tensor]) -> pq.State:
        r"""
        Combines single-qubit gates in a circuit with 6-qubit threshold and updates the state accordingly.

        Args:
            state: The current state, represented as a `pq.State` object.
            matrices: A list of `paddle.Tensor` objects representing the single-qubit gates to be combined.

        Returns:
            The updated state after the single-qubit gates have been combined, represented as a `pq.State` object.
        """
        threshold_qubits = 6
        tensor_times = len(self.qubits_idx) // threshold_qubits
        tensor_left = len(self.qubits_idx) % threshold_qubits
        for threshold_idx in range(tensor_times):
            idx_st = threshold_idx * threshold_qubits
            idx_end = (threshold_idx + 1) * threshold_qubits
            state = simulation(
                state, matrices[idx_st:idx_end], self.qubits_idx[idx_st:idx_end])
        if tensor_left > 0:
            idx_st = tensor_times * threshold_qubits
            state = simulation(
                state, matrices[idx_st:], self.qubits_idx[idx_st:])
        return state

    def forward(self, state: pq.State) -> pq.State:
        for _ in range(self.depth):
            matrices = [self.matrix for _ in self.qubits_idx]

            if self.num_acted_qubits == 1:
                state = self._single_qubit_combine_with_threshold6(
                    state=state, matrices=matrices)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = simulation(
                        state, matrices[param_idx:param_idx + 1], qubit_idx)
        return state


class ParamGate(Gate):
    r"""Base class for quantum parameterized gates.

    Args:
        generator: function that generates the unitary matrix of this gate.
        param: input parameters of quantum parameterized gates. Defaults to ``None`` i.e. randomized.
        qubits_idx: indices of the qubits on which this gate acts on. Defaults to ``None``.
            i.e. list(range(num_acted_qubits)).
        depth: number of layers. Defaults to ``1``.
        num_acted_param: the number of parameters required for a single operation.
        param_sharing: whether all operations are shared by the same parameter set.
        gate_info: information of this gate that will be placed into the gate history or plotted by a Circuit.
            Defaults to ``None``.
        num_qubits: total number of qubits. Defaults to ``None``.
        check_legality: whether check the completeness of the matrix if provided. Defaults to ``True``.
        num_acted_qubits: the number of qubits that this gate acts on.  Defaults to ``None``.
        backend: Backend on which the gates are executed. Defaults to ``None``.
        dtype: Type of data. Defaults to ``None``.
        name_scope: Prefix name used by the layer to name parameters. If prefix is "my_layer", parameter name in
        MyLayer can be "my_layer_0.w_n", where "w" is the parameter base name and "n" is an unique suffix
        auto-generated. If None, prefix name will be snake cased class name. Defaults to ``None``.
    """

    def __init__(
            self, generator: Callable[[paddle.Tensor], paddle.Tensor],
            param: Union[paddle.Tensor, float, List[float]] = None,
            depth: int = 1, num_acted_param: int = 1, param_sharing: bool = False,
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int] = None, gate_info: dict = None,
            num_qubits: int = None, check_legality: bool = True, num_acted_qubits: int = None,
            backend: pq.Backend = None, dtype: str = None, name_scope: str = None
    ):
        # if num_acted_qubits is unknown, generate an example matrix to run Gate.__init__
        ex_matrix = generator(paddle.randn(
            [num_acted_param], dtype)) if num_acted_qubits is None else None

        super().__init__(ex_matrix, qubits_idx, depth, gate_info, num_qubits, check_legality, num_acted_qubits,
                         backend=backend, dtype=dtype, name_scope=name_scope)

        param_shape = _format_param_shape(
            self.depth, self.qubits_idx, num_acted_param, param_sharing)
        self.param_sharing = param_sharing
        self.theta_generation(param, param_shape)
        self.__generator = generator

    @property
    def matrix(self) -> Union[paddle.Tensor, List[paddle.Tensor]]:
        matrices_list = []
        for depth_idx in range(self.depth):
            if self.param_sharing:
                matrices_list.append(self.__generator(self.theta[depth_idx]))
            else:
                matrices_list.extend(
                    self.__generator(self.theta[depth_idx, param_idx])
                    for param_idx, _ in enumerate(self.qubits_idx)
                )
        return matrices_list[0] if len(matrices_list) == 1 else matrices_list
    
    @property
    def kraus_repr(self) -> List[paddle.Tensor]:
        matrix = self.matrix
        assert not isinstance(matrix, List), \
            f"Cannot generate Kraus representation for multiple matrices: received number of matrices {len(matrix)}"
        return [matrix]

    def theta_generation(self, param: Union[paddle.Tensor, float, List[float]], param_shape: List[int]) -> None:
        r""" determine self.theta, and create parameter if necessary

        Args:
            param: input theta
            param_shape: shape for theta

        Note:
            In the following cases ``param`` will be transformed to a parameter:
                - ``param`` is ``None``
            In the following cases ``param`` will be added to the parameter list:
                - ``param`` is a ParamBase
            In the following cases ``param`` will keep unchanged:
                - ``param`` is a Tensor but not a ParamBase
                - ``param`` is a float or a list of floats
        """
        float_dtype = _get_float_dtype(self.dtype)
        
        if param is None:
            theta = self.create_parameter(
                shape=param_shape, dtype=float_dtype,
                default_initializer=paddle.nn.initializer.Uniform(
                    low=0, high=2 * math.pi)
            )
            self.add_parameter('theta', theta)
        elif isinstance(param, paddle.fluid.framework.ParamBase):
            assert param.shape == param_shape, \
                f"Shape assertion failed for input parameter: receive {str(param.shape)}, expect {param_shape}"
            assert param.dtype == (paddle.float32 if float_dtype == 'float32' else paddle.float64), \
                f"Dtype assertion failed for input parameter: receive {param.dtype}, expect {float_dtype}"
            self.add_parameter('theta', param)
        elif isinstance(param, paddle.Tensor):
            param = param.reshape(param_shape).cast(float_dtype)
            self.theta = param
        elif isinstance(param, (int, float)):
            self.theta = paddle.ones(param_shape, dtype=float_dtype) * param
        else:  # when param is a list of float
            self.theta = paddle.to_tensor(
                param, dtype=float_dtype).reshape(param_shape)

    def gate_history_generation(self) -> None:
        r""" determine self.gate_history when gate is parameterized
        """
        gate_history = []
        for depth_idx in range(self.depth):
            for idx, qubit_idx in enumerate(self.qubits_idx):
                if self.param_sharing:
                    param = self.theta[depth_idx]
                else:
                    param = self.theta[depth_idx][idx]
                gate_info = {
                    'gate': self.gate_info['gatename'], 'which_qubits': qubit_idx, 'theta': param}
                gate_history.append(gate_info)
        self.gate_history = gate_history

    def display_in_circuit(self, ax: matplotlib.axes.Axes, x: float, ) -> float:
        r'''The display function called by circuit instance when plotting.

        Args:
            ax: the ``matplotlib.axes.Axes`` instance
            x: the start horizontal position

        Returns:
            the total width occupied

        Note:
            Users could overload this function for custom display.
        '''
        return _base_param_gate_display(self, ax, x)

    def forward(self, state: pq.State) -> pq.State:
        for depth_idx in range(self.depth):
            param_matrices = []
            if self.param_sharing:
                param_matrix = self.__generator(self.theta[depth_idx])
                param_matrix = paddle.cast(param_matrix, self.dtype)
                param_matrices = [param_matrix for _ in self.qubits_idx]
            else:
                for param_idx in range(len(self.qubits_idx)):
                    param_matrix = self.__generator(
                        self.theta[depth_idx, param_idx])
                    param_matrix = paddle.cast(param_matrix, self.dtype)
                    param_matrices.append(param_matrix)
            if self.num_acted_qubits == 1:
                state = self._single_qubit_combine_with_threshold6(
                    state=state, matrices=param_matrices)
            else:
                for param_idx, qubit_idx in enumerate(self.qubits_idx):
                    state = simulation(
                        state, param_matrices[param_idx:param_idx + 1], qubit_idx)
        return state
