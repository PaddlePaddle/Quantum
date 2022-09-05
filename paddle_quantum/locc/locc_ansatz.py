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
The source file of the LoccAnsatz class.
"""

import collections
from matplotlib import docstring
import paddle
import paddle_quantum
from ..gate import H, S, T, X, Y, Z, P, RX, RY, RZ, U3
from ..gate import CNOT, CX, CY, CZ, SWAP
from ..gate import CP, CRX, CRY, CRZ, CU, RXX, RYY, RZZ
from ..gate import MS, CSWAP, Toffoli
from ..gate import UniversalTwoQubits, UniversalThreeQubits
from ..gate import Oracle, ControlOracle
from .locc_party import LoccParty
from .locc_state import LoccState
from paddle_quantum import Operator
from typing import Iterable, Union, Optional, List


class LoccAnsatz(paddle_quantum.ansatz.Circuit):
    r"""Inherit the ``Circuit`` class. The purpose is to build a circuit template for an LOCC task.

    In an LOCC task, each party can only perform quantum operations on their own qubits. So we only allow local quantum gates to be added to each party's qubits.

    Args:
        party: The owner of this circuit.
    """

    def __init__(self, party: LoccParty):
        super().__init__()
        self.party = party
        self.num_local_qubits = len(self.party)

    def __transform_qubits_idx(self, oper):
        if hasattr(oper, "qubits_idx"):
            if isinstance(oper.qubits_idx[0], Iterable):
                oper.qubits_idx = [
                    [self.party[qubit_idx] for qubit_idx in qubits_idx]
                    for qubits_idx in oper.qubits_idx
                ]
            else:
                oper.qubits_idx = [
                    self.party[qubit_idx] for qubit_idx in oper.qubits_idx
                ]

    def append(self, operator: Union[Iterable, paddle_quantum.Operator]) -> None:
        r"""Append an operator.

        Args:
            operator: operator with a name or just an operator.
        """
        if isinstance(operator, Iterable):
            name, oper = operator
            self.__transform_qubits_idx(oper)
            self.add_sublayer(name, oper)
        elif isinstance(operator, paddle_quantum.Operator):
            self.__transform_qubits_idx(operator)
            idx = len(self._sub_layers)
            self.add_sublayer(str(idx), operator)

    def extend(self, operators: List[Operator]) -> None:
        r"""Append a list of operators.

        Args:
            operators: List of operators.
        """
        if len(operators) > 0 and isinstance(operators[0], (list, tuple)):
            for name, oper in operators:
                self.__transform_qubits_idx(oper)
                self.add_sublayer(name, oper)
        else:
            origin_len = len(self._sub_layers)
            for idx, oper in enumerate(operators):
                self.__transform_qubits_idx(oper)
                self.add_sublayer(str(idx + origin_len), oper)

    def insert(self, index: int, operator: Operator) -> None:
        r"""Insert an operator at index ``index``.

        Args:
            index: Index to be inserted.
            operator: An operator.
        """
        new_operators = collections.OrderedDict()
        for idx, name in enumerate(self._sub_layers):
            if idx < index:
                new_operators[name] = self._sub_layers[name]
            elif idx == index:
                if isinstance(operator, (list, tuple)):
                    name, oper = operator
                    self.__transform_qubits_idx(oper)
                    new_operators[name] = oper
                elif isinstance(operator, paddle_quantum.Operator):
                    self.__transform_qubits_idx(operator)
                    new_operators[str(index)] = operator
            elif name.isdigit():
                new_operators[str(int(name) + 1)] = self._sub_layers[name]
            else:
                new_operators[name] = self._sub_layers[name]
        self._sub_layers = new_operators

    def pop(self, operator: Operator) -> None:
        r"""Remove the matched operator.

        Args:
            operator: Matched with which the operator will be popped.
        """
        new_operators = collections.OrderedDict()
        behind_operator = False
        for idx, name in enumerate(self._sub_layers):
            if operator is self._sub_layers[name]:
                behind_operator = True
            elif not behind_operator:
                new_operators[name] = self._sub_layers[name]
            elif name.isdigit():
                new_operators[str(int(name) - 1)] = self._sub_layers[name]
            else:
                new_operators[name] = self._sub_layers[name]
        self._sub_layers = new_operators

    def forward(self, state: LoccState) -> LoccState:
        r"""Forward the input.

        Args:
            state: Initial state.

        Returns:
            Output state.
        """
        for layer in self._sub_layers.values():
            state = layer(state)
        return state

    def h(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
    ) -> None:
        r"""Add single-qubit Hadamard gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = H(qubits_idx, num_qubits, depth)
        self.append(oper)

    def s(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
    ) -> None:
        r"""Add single-qubit S gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = S(qubits_idx, num_qubits, depth)
        self.append(oper)

    def t(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
    ) -> None:
        r"""Add single-qubit T gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = T(qubits_idx, num_qubits, depth)
        self.append(oper)

    def x(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
    ) -> None:
        r"""Add single-qubit X gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = X(qubits_idx, num_qubits, depth)
        self.append(oper)

    def y(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
    ) -> None:
        r"""Add single-qubit Y gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = Y(qubits_idx, num_qubits, depth)
        self.append(oper)

    def z(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
    ) -> None:
        r"""Add single-qubit Z gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = Z(qubits_idx, num_qubits, depth)
        self.append(oper)

    def p(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add single-qubit P gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = P(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def rx(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add single-qubit rotation gates about the x-axis.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = RX(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def ry(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add single-qubit rotation gates about the y-axis.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = RY(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def rz(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: Optional[int] = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add single-qubit rotation gates about the z-axis.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = RZ(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def u3(
        self,
        qubits_idx: Union[Iterable, int, str] = "full",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, Iterable[float]] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add single-qubit rotation gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'full'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = U3(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def cnot(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add CNOT gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CNOT(qubits_idx, num_qubits, depth)
        self.append(oper)

    def cx(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Same as cnot.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CX(qubits_idx, num_qubits, depth)
        self.append(oper)

    def cy(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add controlled Y gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CY(qubits_idx, num_qubits, depth)
        self.append(oper)

    def cz(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add controlled Z gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CZ(qubits_idx, num_qubits, depth)
        self.append(oper)

    def swap(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add SWAP gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = SWAP(qubits_idx, num_qubits, depth)
        self.append(oper)

    def cp(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add controlled P gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CP(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def crx(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add controlled rotation gates about the x-axis.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CRX(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def cry(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add controlled rotation gates about the y-axis.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CRY(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def crz(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add controlled rotation gates about the z-axis.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CRZ(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def cu(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add controlled single-qubit rotation gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CU(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def rxx(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add RXX gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = RXX(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def ryy(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add RYY gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = RYY(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def rzz(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add RZZ gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = RZZ(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def ms(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add Mølmer-Sørensen (MS) gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = MS(qubits_idx, num_qubits, depth)
        self.append(oper)

    def cswap(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add CSWAP (Fredkin) gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CSWAP(qubits_idx, num_qubits, depth)
        self.append(oper)

    def ccx(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add CCX gates.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = CSWAP(qubits_idx, num_qubits, depth)
        self.append(oper)
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = Toffoli(qubits_idx, num_qubits, depth)
        self.append(oper)

    def universal_two_qubits(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add universal two-qubit gates. One of such a gate requires 15 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = UniversalTwoQubits(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def universal_three_qubits(
        self,
        qubits_idx: Union[Iterable, str] = "cycle",
        num_qubits: int = None,
        depth: int = 1,
        param: Union[paddle.Tensor, float] = None,
        param_sharing: bool = False,
    ) -> None:
        r"""Add universal three-qubit gates. One of such a gate requires 81 parameters.

        Args:
            qubits_idx: Indices of the qubits on which the gates are applied. Defaults to ``'cycle'``.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
            param: Parameters of the gates. Defaults to ``None``.
            param_sharing: Whether gates in the same layer share a parameter. Defaults to ``False``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = UniversalThreeQubits(qubits_idx, num_qubits, depth, param, param_sharing)
        self.append(oper)

    def oracle(
        self,
        oracle: paddle.Tensor,
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int],
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        r"""Add an oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            qubits_idx: Indices of the qubits on which the gates are applied.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = Oracle(oracle, qubits_idx, num_qubits, depth)
        self.append(oper)

    def control_oracle(
        self,
        oracle: paddle.Tensor,
        # num_control_qubits: int, controlled_value: 'str',
        qubits_idx: Union[Iterable[Iterable[int]], Iterable[int]],
        num_qubits: int = None,
        depth: int = 1,
    ) -> None:
        """Add a controlled oracle gate.

        Args:
            oracle: Unitary oracle to be implemented.
            qubits_idx: Indices of the qubits on which the gates are applied.
            num_qubits: Total number of qubits. Defaults to ``None``.
            depth: Number of layers. Defaults to ``1``.
        """
        if num_qubits is None:
            num_qubits = self.num_local_qubits
        oper = ControlOracle(oracle, qubits_idx, num_qubits, depth)
        self.append(oper)
