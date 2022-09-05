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
The source file of the Circuit class.
"""

import warnings
import paddle
from .container import Sequential
from ..gate import Gate, H, S, T, X, Y, Z, P, RX, RY, RZ, U3
from ..gate import CNOT, CX, CY, CZ, SWAP
from ..gate import CP, CRX, CRY, CRZ, CU, RXX, RYY, RZZ
from ..gate import MS, CSWAP, Toffoli
from ..gate import UniversalTwoQubits, UniversalThreeQubits
from ..gate import Oracle, ControlOracle
from ..gate import SuperpositionLayer, WeakSuperpositionLayer, LinearEntangledLayer
from ..gate import RealBlockLayer, RealEntangledLayer, ComplexBlockLayer, ComplexEntangledLayer
from ..gate import QAOALayer
from ..gate import AmplitudeEncoding
from ..channel import BitFlip, PhaseFlip, BitPhaseFlip, AmplitudeDamping, GeneralizedAmplitudeDamping, PhaseDamping
from ..channel import Depolarizing, PauliChannel, ResetChannel, ThermalRelaxation, KrausRepr
from ..intrinsic import _get_float_dtype
from ..state import zero_state
from ..operator import Collapse
from typing import Union, Iterable, Optional, Dict, List, Tuple
from paddle_quantum import State, get_backend, get_dtype, Backend
from math import pi
import numpy as np


class Circuit(Sequential):
    r"""Quantum circuit.

    Args:
        num_qubits: Number of qubits. Defaults to None.
    """
    
    def __init__(self, num_qubits: Optional[int] = None):
        super().__init__()
        self.__num_qubits = num_qubits
        
        # whether the circuit is a dynamic quantum circuit
        self.__isdynamic = True if num_qubits is None else False
        
        # alias for ccx
        self.toffoli = self.ccx

    @property
    def num_qubits(self) -> int:
        r"""Number of qubits.
        """
        return self.__num_qubits

    @property
    def isdynamic(self) -> bool:
        r"""Whether the circuit is dynamic
        """
        return self.__dynamic
    
    @num_qubits.setter
    def num_qubits(self, value: int) -> None:
        assert isinstance(value, int)
        self.__num_qubits = value
    
    @property    
    def param(self) -> paddle.Tensor:
        r"""Flattened parameters in the circuit.
        """
        if len(self.parameters()) == 0:
            return []
        return paddle.concat([paddle.flatten(param) for param in self.parameters()])
    
    @property
    def grad(self) -> np.ndarray:
        r"""Gradients with respect to the flattened parameters.
        """
        grad_list = []
        for param in self.parameters():
            assert param.grad is not None, 'the gradient is None, run the backward first before calling this property,' + \
                ' otherwise check where the gradient chain is broken'
            grad_list.append(paddle.flatten(param.grad))
        return paddle.concat(grad_list).numpy()
    
    @property
    def depth(self) -> int:
        r"""(current) Depth of this Circuit
        """
        qubit_depth = [len(qubit_gates) for qubit_gates in self.qubit_history]
        return max(qubit_depth)
    
    def update_param(self, theta: Union[paddle.Tensor, np.ndarray, float], idx: int = None) -> None:
        r"""Replace parameters of all/one layer(s) by ``theta``.

        Args:
            theta: New parameters
            idx: Index of replacement. Defaults to None, referring to all layers.
        """
        if not isinstance(theta, paddle.Tensor):
            theta = paddle.to_tensor(theta, dtype='float32')
        theta = paddle.flatten(theta)
        
        backend_dtype = _get_float_dtype(get_dtype())
        if backend_dtype != 'float32':
            warnings.warn(
                f"\ndtype of parameters will be float32 instead of {backend_dtype}", UserWarning)
        
        if idx is None:
            assert self.param.shape == theta.shape, "the shape of input parameters is not correct"
            for layer in self.sublayers():
                for name, _ in layer.named_parameters():
                    param = getattr(layer, name)
                    num_param = int(paddle.numel(param))

                    param = paddle.create_parameter(
                        shape=param.shape,
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(theta[:num_param].reshape(param.shape)),
                    )

                    setattr(layer, 'theta', param)
                    if num_param == theta.shape[0]:
                        return
                    theta = theta[num_param:]
        elif isinstance(idx, int):
            assert idx < len(self.sublayers()), "the index is out of range, expect below " + str(len(self.sublayers()))
            layer = self.sublayers()[idx]
            assert theta.shape == paddle.concat([paddle.flatten(param) for param in layer.parameters()]).shape, \
                "the shape of input parameters is not correct,"
                
            for name, _ in layer.named_parameters():
                param = getattr(layer, name)
                num_param = int(paddle.numel(param))

                param = paddle.create_parameter(
                    shape=param.shape,
                    dtype='float32',
                    default_initializer=paddle.nn.initializer.Assign(theta[:num_param].reshape(param.shape)),
                )

                setattr(layer, 'theta', param)
                if num_param == theta.shape[0]:
                    return
                theta = theta[num_param:]
        else:
            raise ValueError("idx must be an integer or None")
        
    def transfer_static(self) -> None:
        r""" set ``stop_gradient`` of all parameters of the circuit as ``True``
        
        """
        for layer in self.sublayers():
            for name, _ in layer.named_parameters():
                param = getattr(layer, name)
                param.stop_gradient = True 
                setattr(layer, 'theta', param)
                
    def randomize_param(self, low: float = 0, high: Optional[float] = 2 * pi) -> None:
        r"""Randomize parameters of the circuit in a range from low to high.
        
        Args:
            low: Lower bound.
            high: Upper bound.
        """
        for layer in self.sublayers():
            for name, _ in layer.named_parameters():
                param = getattr(layer, name)

                param = paddle.create_parameter(
                    shape=param.shape,
                    dtype=param.dtype,
                    default_initializer=paddle.nn.initializer.Uniform(low=low, high=high),
                )
                setattr(layer, 'theta', param)

    def __num_qubits_update(self, qubits_idx: Union[Iterable[int], int, str]) -> None:
        r"""Update ``self.num_qubits`` according to ``qubits_idx``, or report error.
        
        Args:
            qubits_idx: Input qubit indices of a quantum gate.
        """
        num_qubits = self.__num_qubits
        if isinstance(qubits_idx, str):
            assert num_qubits is not None, "The qubit idx cannot be full when the number of qubits is unknown"
            return

        if isinstance(qubits_idx, Iterable):
            max_idx = np.max(qubits_idx)
        else:
            max_idx = qubits_idx

        if num_qubits is None:
            self.__num_qubits = max_idx + 1
            return

        assert max_idx + 1 <= num_qubits or self.__isdynamic, \
            f"The circuit is not a dynamic quantum circuit. Invalid input qubit idx: {max_idx} num_qubit: {self.__num_qubits}"
        self.__num_qubits = int(max(max_idx + 1, num_qubits))

    def h(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1
    ) -> None:
        r"""Add single-qubit Hadamard gates.

        The matrix form of such a gate is:

        .. math::

            H = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1&1\\
                    1&-1
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(H(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def s(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1
    ) -> None:
        r"""Add single-qubit S gates.

        The matrix form of such a gate is:

        .. math::

            S =
                \begin{bmatrix}
                    1&0\\
                    0&i
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            S(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def t(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1
    ) -> None:
        r"""Add single-qubit T gates.

        The matrix form of such a gate is:

        .. math::

            T = \begin{bmatrix}
                    1&0\\
                    0&e^\frac{i\pi}{4}
                \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            T(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def x(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1
    ) -> None:
        r"""Add single-qubit X gates.

        The matrix form of such a gate is:

        .. math::
           X = \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            X(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def y(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1
    ) -> None:
        r"""Add single-qubit Y gates.

        The matrix form of such a gate is:

        .. math::

            Y = \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            Y(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def z(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1
    ) -> None:
        r"""Add single-qubit Z gates.

        The matrix form of such a gate is:

        .. math::

            Z = \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            Z(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def p(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add single-qubit P gates.

        The matrix form of such a gate is:

        .. math::

            P(\theta) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            P(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth, param, param_sharing))

    def rx(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add single-qubit rotation gates about the x-axis.

        The matrix form of such a gate is:

        .. math::

            R_X(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(RX(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                       depth, param, param_sharing))

    def ry(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add single-qubit rotation gates about the y-axis.

        The matrix form of such a gate is:

        .. math::

            R_Y(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(RY(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                       depth, param, param_sharing))

    def rz(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: Optional[int] = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add single-qubit rotation gates about the z-axis.

        The matrix form of such a gate is:

        .. math::

            R_Z(\theta) = \begin{bmatrix}
                e^{-i\frac{\theta}{2}} & 0 \\
                0 & e^{i\frac{\theta}{2}}
            \end{bmatrix}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(RZ(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                       depth, param, param_sharing))

    def u3(
            self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, Iterable[float]] = None, param_sharing: bool = False
    ) -> None:
        r"""Add single-qubit rotation gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                U_3(\theta, \phi, \lambda) =
                    \begin{bmatrix}
                        \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                        e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                    \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(U3(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                       depth, param, param_sharing))

    def cnot(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add CNOT gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CNOT} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            CNOT(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def cx(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Same as cnot.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            CX(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def cy(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add controlled Y gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CY} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & -1j \\
                    0 & 0 & 1j & 0
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            CY(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def cz(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add controlled Z gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CZ} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & -1
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            CZ(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def swap(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add SWAP gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{SWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            SWAP(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def cp(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled P gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CP}(\theta) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\theta}
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(CP(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                       depth, param, param_sharing))

    def crx(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled rotation gates about the x-axis.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CR_X} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                    0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(CRX(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                        depth, param, param_sharing))

    def cry(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled rotation gates about the y-axis.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CR_Y} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Y\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                    0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(CRY(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                        depth, param, param_sharing))

    def crz(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled rotation gates about the z-axis.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CR_Z} &= |0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Z\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
                    0 & 0 & 0 & e^{i\frac{\theta}{2}}
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(CRZ(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                        depth, param, param_sharing))

    def cu(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add controlled single-qubit rotation gates.

        For a 2-qubit quantum circuit, when `qubits_idx` is `[0, 1]`, the matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CU}
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac\theta2 &-e^{i\lambda}\sin\frac\theta2 \\
                    0 & 0 & e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(CU(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                       depth, param, param_sharing))

    def rxx(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add RXX gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{R_{XX}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
                    \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(RXX(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                        depth, param, param_sharing))

    def ryy(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add RYY gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{R_{YY}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
                    \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(RYY(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                        depth, param, param_sharing))

    def rzz(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add RZZ gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{R_{ZZ}}(\theta) =
                    \begin{bmatrix}
                        e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                        0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                        0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                        0 & 0 & 0 & e^{-i\frac{\theta}{2}}
                    \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(RZZ(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                        depth, param, param_sharing))

    def ms(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add Mølmer-Sørensen (MS) gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{MS} = \mathit{R_{XX}}(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                    \begin{bmatrix}
                        1 & 0 & 0 & i \\
                        0 & 1 & i & 0 \\
                        0 & i & 1 & 0 \\
                        i & 0 & 0 & 1
                    \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            MS(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def cswap(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add CSWAP (Fredkin) gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                \mathit{CSWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            CSWAP(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def ccx(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add CCX gates.

        The matrix form of such a gate is:

        .. math::

            \begin{align}
                    \mathit{CCX} = \begin{bmatrix}
                        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                    \end{bmatrix}
                \end{align}

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            Toffoli(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def universal_two_qubits(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add universal two-qubit gates. One of such a gate requires 15 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(UniversalTwoQubits(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                                       depth, param, param_sharing))

    def universal_three_qubits(
            self, qubits_idx: Union[Iterable[int], str] = 'cycle', num_qubits: int = None, depth: int = 1,
            param: Union[paddle.Tensor, float] = None, param_sharing: bool = False
    ) -> None:
        r"""Add universal three-qubit gates. One of such a gate requires 81 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'cycle'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            param: Parameters of the gates. Defaults to None.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to False.

        Raises:
            ValueError: The ``param`` must be paddle.Tensor or float.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(UniversalThreeQubits(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                                         depth, param, param_sharing))

    def oracle(
            self, oracle: paddle.tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
            num_qubits: int = None, depth: int = 1, gate_name: str = 'O'
    ) -> None:
        """Add an oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            qubits_idx: Indices of the qubits on which the gates are applied.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            gate_name: name of this oracle
        """
        self.__num_qubits_update(qubits_idx)
        self.append(Oracle(oracle, qubits_idx, 
                           self.num_qubits if num_qubits is None else num_qubits, depth, gate_name))

    def control_oracle(
            self, oracle: paddle.Tensor, qubits_idx: Union[Iterable[Iterable[int]], Iterable[int]],
            num_qubits: int = None, depth: int = 1, gate_name: str = 'cO'
    ) -> None:
        """Add a controlled oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            qubits_idx: Indices of the qubits on which the gates are applied.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
            gate_name: name of this oracle
        """
        self.__num_qubits_update(qubits_idx)
        self.append(ControlOracle(oracle, qubits_idx, 
                                  self.num_qubits if num_qubits is None else num_qubits, depth, gate_name))
        
    def collapse(self, qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None,
                 desired_result: Union[int, str] = None, if_print: bool = False,
                 measure_basis: Union[Iterable[paddle.Tensor], str] = 'z') -> None:
        r"""
        Args:
            qubits_idx: list of qubits to be collapsed. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            desired_result: The desired result you want to collapse. Defaults to ``None`` meaning randomly choose one.
            if_print: whether print the information about the collapsed state. Defaults to ``False``.
            measure_basis: The basis of the measurement. The quantum state will collapse to the corresponding eigenstate.

        Raises:
            NotImplementedError: If the basis of measurement is not z. Other bases will be implemented in future.
            TypeError: cannot get probability of state when the backend is unitary_matrix.
            
        Note:
            When desired_result is `None`, Collapse does not support gradient calculation
        """
        self.__num_qubits_update(qubits_idx)
        self.append(Collapse(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, 
                             desired_result, if_print, measure_basis))

    def superposition_layer(
            self, qubits_idx: Iterable[int] = 'full', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add layers of Hadamard gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            SuperpositionLayer(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))
        
    def weak_superposition_layer(
            self, qubits_idx: Iterable[int] = 'full', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add layers of Ry gates with a rotation angle :math:`\pi/4`.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            WeakSuperpositionLayer(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))
        
    def linear_entangled_layer(
        self, qubits_idx: Iterable[int] = 'full', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add linear entangled layers consisting of Ry gates, Rz gates, and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            LinearEntangledLayer(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def real_entangled_layer(
        self, qubits_idx: Iterable[int] = 'full', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add strongly entangled layers consisting of Ry gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            RealEntangledLayer(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def complex_entangled_layer(
        self, qubits_idx: Iterable[int] = 'full', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add strongly entangled layers consisting of single-qubit rotation gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            ComplexEntangledLayer(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def real_block_layer(
        self, qubits_idx: Iterable[int] = 'full', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add weakly entangled layers consisting of Ry gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            RealBlockLayer(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))
    
    def complex_block_layer(
        self, qubits_idx: Iterable[int] = 'full', num_qubits: int = None, depth: int = 1
    ) -> None:
        r"""Add weakly entangled layers consisting of single-qubit rotation gates and CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
            depth: Number of layers. Defaults to 1.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(
            ComplexBlockLayer(qubits_idx, self.num_qubits if num_qubits is None else num_qubits, depth))

    def bit_flip(
        self, prob: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add bit flip channels.

        Args:
            prob: Probability of a bit flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(BitFlip(prob, qubits_idx, 
                            self.num_qubits if num_qubits is None else num_qubits))
        
    def phase_flip(
        self, prob: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add phase flip channels.

        Args:
            prob: Probability of a phase flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(PhaseFlip(prob, qubits_idx, 
                              self.num_qubits if num_qubits is None else num_qubits))    
        
    def bit_phase_flip(
        self, prob: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add bit phase flip channels.

        Args:
            prob: Probability of a bit phase flip.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(BitPhaseFlip(prob, qubits_idx, 
                                 self.num_qubits if num_qubits is None else num_qubits)) 
    
    def amplitude_damping(
        self, gamma: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add amplitude damping channels.

        Args:
            gamma: Damping probability.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(AmplitudeDamping(gamma, qubits_idx, 
                                     self.num_qubits if num_qubits is None else num_qubits)) 
    
    #TODO: change bug    
    def generalized_amplitude_damping(
        self, gamma: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add generalized amplitude damping channels.

        Args:
            gamma: Damping probability.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(GeneralizedAmplitudeDamping(gamma, qubits_idx, 
                                                self.num_qubits if num_qubits is None else num_qubits)) 
        
    def phase_damping(
        self, gamma: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add phase damping channels.

        Args:
            gamma: Parameter of the phase damping channel.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(PhaseDamping(gamma, qubits_idx, 
                                 self.num_qubits if num_qubits is None else num_qubits)) 

    def depolarizing(
        self, prob: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add depolarizing channels.

        Args:
            prob: Parameter of the depolarizing channel.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(Depolarizing(prob, qubits_idx, 
                                 self.num_qubits if num_qubits is None else num_qubits)) 

    def pauli_channel(
        self, prob: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add Pauli channels.

        Args:
            prob: Probabilities corresponding to the Pauli X, Y, and Z operators.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(PauliChannel(prob, qubits_idx, 
                                 self.num_qubits if num_qubits is None else num_qubits)) 
        
    def reset_channel(
        self, prob: Union[paddle.Tensor, float], qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add reset channels.

        Args:
            prob: Probabilities of resetting to :math:`|0\rangle` and to :math:`|1\rangle`.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(ResetChannel(prob, qubits_idx, 
                                 self.num_qubits if num_qubits is None else num_qubits)) 

    def thermal_relaxation(
        self, const_t: Union[paddle.Tensor, Iterable[float]], exec_time: Union[paddle.Tensor, float], 
        qubits_idx: Union[Iterable[int], int, str] = 'full', num_qubits: int = None
    ) -> None:
        r"""Add thermal relaxation channels.

        Args:
            const_t: :math:`T_1` and :math:`T_2` relaxation time in microseconds.
            exec_time: Quantum gate execution time in the process of relaxation in nanoseconds.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(ThermalRelaxation(const_t, exec_time, qubits_idx, 
                                      self.num_qubits if num_qubits is None else num_qubits))

    def kraus_repr(
            self, kraus_oper: Iterable[paddle.Tensor],
            qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
            num_qubits: int = None
    ) -> None:
        r"""Add custom channels in the Kraus representation.

        Args:
            kraus_oper: Kraus operators of this channel.
            qubits_idx: Indices of the qubits on which the channels are applied. Defaults to 'full'.
            num_qubits: Total number of qubits. Defaults to None.
        """
        self.__num_qubits_update(qubits_idx)
        self.append(KrausRepr(kraus_oper, qubits_idx, 
                              self.num_qubits if num_qubits is None else num_qubits))

    def qaoa_layer(self, edges: Iterable, nodes: Iterable, depth: Optional[int] = 1) -> None:
        # TODO: see qaoa layer in layer.py
        self.__num_qubits_update(edges)
        self.__num_qubits_update(nodes)
        self.append(QAOALayer(edges, nodes, depth))

    def unitary_matrix(self, num_qubits: Optional[int] = None) -> paddle.Tensor:
        r"""Get the unitary matrix form of the circuit.

        Args:
            num_qubits: Total number of qubits. Defaults to None.

        Returns:
            Unitary matrix form of the circuit.
        """
        if num_qubits is None:
            num_qubits = self.__num_qubits
        else:
            assert num_qubits >= self.__num_qubits
        
        backend = get_backend()
        self.to(backend=Backend.UnitaryMatrix)
        unitary = State(paddle.eye(2 ** num_qubits).cast(get_dtype()), 
                        backend=Backend.UnitaryMatrix)
        for layer in self._sub_layers.values():
            unitary = layer(unitary)
        self.to(backend=backend)
        return unitary.data

    @property
    def gate_history(self) -> List[Dict[str, Union[str, List[int], paddle.Tensor]]]:
        r""" list of gates information of circuit

        Returns:
            history of quantum gates of circuit
            
        """
        gate_history = []
        for gate in self.sublayers():
            if gate.gate_name is None:
                raise NotImplementedError(f"Gate {type(gate)} has no gate name and hence cannot be recorded into history.")
            else:
                gate.gate_history_generation()
                gate_history.extend(gate.gate_history)
        return gate_history

    def __count_history(self, history):
        # Record length of each section
        length = [5]
        n = self.__num_qubits
        # Record current section number for every qubit
        qubit = [0] * n
        # Number of sections
        qubit_max = max(qubit)
        # Record section number for each gate
        gate = []

        for current_gate in history:
            # Single-qubit gates with no params to print
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u', 'sdg', 'tdg'}:
                curr_qubit = current_gate['which_qubits']
                gate.append(qubit[curr_qubit])
                qubit[curr_qubit] = qubit[curr_qubit] + 1
                # A new section is added
                if qubit[curr_qubit] > qubit_max:
                    length.append(5)
                    qubit_max = qubit[curr_qubit]
            # Gates with params to print
            elif current_gate['gate'] in {'p', 'rx', 'ry', 'rz'}:
                curr_qubit = current_gate['which_qubits']
                gate.append(qubit[curr_qubit])
                if length[qubit[curr_qubit]] == 5:
                    length[qubit[curr_qubit]] = 13
                qubit[curr_qubit] = qubit[curr_qubit] + 1
                if qubit[curr_qubit] > qubit_max:
                    length.append(5)
                    qubit_max = qubit[curr_qubit]
            # Two-qubit gates or Three-qubit gates
            elif (
                    current_gate['gate'] in {
                'cnot', 'swap', 'rxx', 'ryy', 'rzz', 'ms',
                'cy', 'cz', 'cu', 'cp', 'crx', 'cry', 'crz'} or
                    current_gate['gate'] in {'cswap', 'ccx'}
            ):
                a = max(current_gate['which_qubits'])
                b = min(current_gate['which_qubits'])
                ind = max(qubit[b: a + 1])
                gate.append(ind)
                if length[ind] < 13 and current_gate['gate'] in {'rxx', 'ryy', 'rzz',
                                                                 'cp', 'crx', 'cry', 'crz'}:
                    length[ind] = 13
                for j in range(b, a + 1):
                    qubit[j] = ind + 1
                if ind + 1 > qubit_max:
                    length.append(5)
                    qubit_max = ind + 1

        return length, gate
    
    @property
    def qubit_history(self) -> List[List[Tuple[Dict[str, Union[str, List[int], paddle.Tensor]], int]]]:
        r""" gate information on each qubit
        
        Returns:
            list of gate history on each qubit
        
        Note:
            The entry ``qubit_history[i][j][0/1]`` returns the gate information / gate index of the j-th gate
            applied on the i-th qubit.
        """
        history_qubit = []
        for i in range(self.num_qubits):
            history_i = []
            history_qubit.append(history_i)
        for idx, i in enumerate(self.gate_history):
            qubits = i["which_qubits"]
            if not isinstance(qubits, Iterable):
                history_qubit[qubits].append([i, idx])
            else:
                for j in qubits:
                    history_qubit[j].append([i, idx])
        return history_qubit

    def __str__(self) -> str:
        history = self.gate_history
        num_qubits = self.__num_qubits
        length, gate = self.__count_history(history)
        # Ignore the unused section
        total_length = sum(length) - 5

        print_list = [['-' if i % 2 == 0 else ' '] * 
                      total_length for i in range(num_qubits * 2)]

        for i, current_gate in enumerate(history):
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u'}:
                # Calculate starting position ind of current gate
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'] * 2][ind + length[sec] // 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'sdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'tdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'p', 'rx', 'ry', 'rz'}:
                sec = gate[i]
                ind = sum(length[:sec])
                line = current_gate['which_qubits'] * 2
                # param = self.__param[current_gate['theta'][2 if current_gate['gate'] == 'rz' else 0]]
                param = current_gate['theta']
                if current_gate['gate'] == 'p':
                    print_list[line][ind + 2] = 'P'
                    print_list[line][ind + 3] = ' '
                else:
                    print_list[line][ind + 2] = 'R'
                    print_list[line][ind + 3] = current_gate['gate'][1]
                print_list[line][ind + 4] = '('
                print_list[line][ind + 5: ind + 10] = format(float(param.numpy()), '.3f')[:5]
                print_list[line][ind + 10] = ')'
            # Two-qubit gates
            elif current_gate['gate'] in {'cnot', 'swap', 'rxx', 'ryy', 'rzz', 'ms', 'cz', 'cy',
                                          'cu', 'crx', 'cry', 'crz'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit = current_gate['which_qubits'][0]
                tqubit = current_gate['which_qubits'][1]
                if current_gate['gate'] in {'cnot', 'swap', 'cy', 'cz', 'cu'}:
                    print_list[cqubit * 2][ind + length[sec] // 2] = \
                        '*' if current_gate['gate'] in {'cnot', 'cy', 'cz', 'cu'} else 'x'
                    print_list[tqubit * 2][ind + length[sec] // 2] = \
                        'x' if current_gate['gate'] in {'swap', 'cnot'} else current_gate['gate'][1]
                elif current_gate['gate'] == 'ms':
                    for qubit in {cqubit, tqubit}:
                        print_list[qubit * 2][ind + length[sec] // 2 - 1] = 'M'
                        print_list[qubit * 2][ind + length[sec] // 2] = '_'
                        print_list[qubit * 2][ind + length[sec] // 2 + 1] = 'S'
                elif current_gate['gate'] in {'rxx', 'ryy', 'rzz'}:
                    # param = self.__param[current_gate['theta'][0]]
                    param = current_gate['theta']
                    for line in {cqubit * 2, tqubit * 2}:
                        print_list[line][ind + 2] = 'R'
                        print_list[line][ind + 3: ind + 5] = current_gate['gate'][1:3].lower()
                        print_list[line][ind + 5] = '('
                        print_list[line][ind + 6: ind + 10] = format(float(param.numpy()), '.2f')[:4]
                        print_list[line][ind + 10] = ')'
                elif current_gate['gate'] in {'crx', 'cry', 'crz'}:
                    # param = self.__param[current_gate['theta'][2 if current_gate['gate'] == 'crz' else 0]]
                    param = current_gate['theta']
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit * 2][ind + 2] = 'R'
                    print_list[tqubit * 2][ind + 3] = current_gate['gate'][2]
                    print_list[tqubit * 2][ind + 4] = '('
                    print_list[tqubit * 2][ind + 5: ind + 10] = format(float(param.numpy()), '.3f')[:5]
                    print_list[tqubit * 2][ind + 10] = ')'
                start_line = min(cqubit, tqubit)
                end_line = max(cqubit, tqubit)
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
            # Three-qubit gates
            elif current_gate['gate'] in {'cswap'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit = current_gate['which_qubits'][0]
                tqubit1 = current_gate['which_qubits'][1]
                tqubit2 = current_gate['which_qubits'][2]
                start_line = min(current_gate['which_qubits'])
                end_line = max(current_gate['which_qubits'])
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
                if current_gate['gate'] in {'cswap'}:
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit1 * 2][ind + length[sec] // 2] = 'x'
                    print_list[tqubit2 * 2][ind + length[sec] // 2] = 'x'
            elif current_gate['gate'] in {'ccx'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit1 = current_gate['which_qubits'][0]
                cqubit2 = current_gate['which_qubits'][1]
                tqubit = current_gate['which_qubits'][2]
                start_line = min(current_gate['which_qubits'])
                end_line = max(current_gate['which_qubits'])
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
                if current_gate['gate'] in {'ccx'}:
                    print_list[cqubit1 * 2][ind + length[sec] // 2] = '*'
                    print_list[cqubit2 * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit * 2][ind + length[sec] // 2] = 'X'
            else:
                raise NotImplementedError(f"Not support to print the gate {current_gate['gate']}.")

        print_list = list(map(''.join, print_list))
        return_str = '\n'.join(print_list)

        return return_str
    
    def forward(self, state: Optional[State] = None) -> State:
        r""" forward the input
        
        Args:
            state: initial state
            
        Returns:
            output quantum state
        
        """
        assert self.__num_qubits is not None, "Information about num_qubits is required before running the circuit"
        
        if state is None:
            state = zero_state(self.__num_qubits, self.backend, self.dtype)
        else:
            assert self.__num_qubits == state.num_qubits, \
                f"num_qubits does not agree: expected {self.__num_qubits}, received {state.num_qubits}"
            
        return super().forward(state)
