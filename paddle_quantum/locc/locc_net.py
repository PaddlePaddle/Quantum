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
The source file of the LoccNet class.
"""

import math
import numpy as np
import functools
import paddle
import paddle_quantum
from .locc_party import LoccParty
from .locc_state import LoccState
from .locc_ansatz import LoccAnsatz
from typing import Optional, Union, Iterable, List


class LoccNet(paddle.nn.Layer):
    r"""Used to design LOCC protocols and perform training or verification."""

    def __init__(self):
        super().__init__()
        self.parties_by_number = []
        self.parties_by_name = {}
        self.init_status = LoccState()

    def set_init_state(self, state: paddle_quantum.State, qubits_idx: Iterable):
        r"""Initialize the LoccState of LoccNet.

        Args:
            state: Matrix form of the input quantum state.
            qubits_idx: Indices of the qubits corresponding to the input quantum state. It should be a ``tuple`` of
                ``(party_id, qubit_id)``, or a ``list`` of it.

        Raises:
            ValueError: Party's ID should be ``str`` or ``int``.
        """
        qubits_idx = list(qubits_idx)
        temp_len = int(math.log2(math.sqrt(self.init_status.data.size)))
        self.init_status.data = paddle.kron(self.init_status.data, state.data)
        self.init_status.num_qubits += state.num_qubits
        for idx, (party_id, qubit_id) in enumerate(qubits_idx):
            if isinstance(party_id, str):
                self.parties_by_name[party_id][qubit_id] = temp_len + idx
            elif isinstance(party_id, int):
                self.parties_by_number[party_id][qubit_id] = temp_len + idx
            else:
                raise ValueError

    def __partial_trace(self, rho_AB: paddle.Tensor, dim1: int, dim2: int, A_or_B: int):
        r"""TODO To be checked."""
        if A_or_B == 2:
            dim1, dim2 = dim2, dim1
        idty_np = np.identity(dim2)
        idty_B = paddle.to_tensor(idty_np, dtype=rho_AB.dtype)
        zero_np = np.zeros([dim2, dim2])
        res = paddle.to_tensor(zero_np, dtype=rho_AB.dtype)
        for dim_j in range(dim1):
            row_top = paddle.zeros([1, dim_j])
            row_mid = paddle.ones([1, 1])
            row_bot = paddle.zeros([1, dim1 - dim_j - 1])
            bra_j = paddle.concat([row_top, row_mid, row_bot], axis=1)
            if A_or_B == 1:
                row_tmp = paddle.kron(bra_j, idty_B)
                row_tmp = paddle.cast(row_tmp, rho_AB.dtype)
                row_tmp_conj = paddle.conj(row_tmp)
                res = paddle.add(
                    res,
                    paddle.matmul(
                        paddle.matmul(row_tmp, rho_AB),
                        paddle.transpose(row_tmp_conj, perm=[1, 0]),
                    ),
                )
            if A_or_B == 2:
                row_tmp = paddle.kron(idty_B, bra_j)
                row_tmp = paddle.cast(row_tmp, rho_AB.dtype)
                row_tmp_conj = paddle.conj(row_tmp)
                res = paddle.add(
                    res,
                    paddle.matmul(
                        paddle.matmul(row_tmp, rho_AB),
                        paddle.transpose(row_tmp_conj, perm=[1, 0]),
                    ),
                )
        return res

    def partial_state(
        self,
        state: Union[List[LoccState], LoccState],
        qubits_idx: Iterable,
        is_desired: bool = True,
    ) -> Union[List[LoccState], LoccState]:
        r"""Get the quantum state of the qubits of interest.

        Args:
            state: Input LOCC state.
            qubits_idx: Indices of the qubits of interest. It should be a ``tuple`` of ``(party_id, qubit_id)``,
                or a ``list`` of it.
            is_desired: If ``True``, return the partial quantum state with the respect to the given qubits;
                if ``False``, return the partial quantum state with the respect to the remaining qubits. Defaults to ``True``.

        Raises:
            ValueError: Party's ID should be ``str`` or ``int``.
            ValueError: The ``state`` should be ``LoccState`` or a ``list`` of it.

        Returns:
            LOCC state after obtaining partial quantum state.
        """
        qubits_idx = list(qubits_idx)
        qubits_list = []
        for party_id, qubit_id in qubits_idx:
            if isinstance(party_id, str):
                qubits_list.append(self.parties_by_name[party_id][qubit_id])
            elif isinstance(party_id, int):
                qubits_list.append(self.parties_by_number[party_id][qubit_id])
            else:
                raise ValueError
        m = len(qubits_list)
        if isinstance(state, LoccState):
            n = state.num_qubits
        elif isinstance(state, list):
            n = state[0].num_qubits
        else:
            raise ValueError("can't recognize the input status")

        assert max(qubits_list) <= n, "qubit index out of range"
        origin_seq = list(range(0, n))
        target_seq = [idx for idx in origin_seq if idx not in qubits_list]
        target_seq = qubits_list + target_seq

        swapped = [False] * n
        swap_list = []
        for idx in range(0, n):
            if not swapped[idx]:
                next_idx = idx
                swapped[next_idx] = True
                while not swapped[target_seq[next_idx]]:
                    swapped[target_seq[next_idx]] = True
                    swap_list.append((next_idx, target_seq[next_idx]))
                    next_idx = target_seq[next_idx]
        
        cir = paddle_quantum.ansatz.Circuit(n)
        for a, b in swap_list:
            cir.swap([a, b])

        if isinstance(state, LoccState):
            state = cir(state)
            if is_desired:
                state_data = self.__partial_trace(state.data, 2**m, 2 ** (n - m), 2)
            else:
                state_data = self.__partial_trace(state.data, 2**m, 2 ** (n - m), 1)
            new_state = state.clone()
            new_state.data = state_data
            new_state.num_qubits = int(math.log2(state_data.shape[-1]))
        elif isinstance(state, list):
            new_state = []
            for each_state in state:
                each_state = cir(each_state)
                if is_desired:
                    state_data = self.__partial_trace(
                        each_state.data, 2**m, 2 ** (n - m), 2
                    )
                else:
                    state_data = self.__partial_trace(
                        each_state.data, 2**m, 2 ** (n - m), 1
                    )
                _state = each_state.clone()
                _state.data = state_data
                _state.num_qubits = int(math.log2(state_data.shape[-1]))
                new_state.append(_state)
        else:
            # TODO: seems unnecessary
            raise ValueError("can't recognize the input status")

        return new_state

    def reset_state(
        self,
        status: Union[List[LoccState], LoccState],
        state: paddle_quantum.State,
        which_qubits: Iterable,
    ) -> Union[List[LoccState], LoccState]:
        r"""Reset the quantum state of the qubits of interest.

        Args:
            status:  LOCC state before resetting.
            state: Matrix form of the input quantum state.
            which_qubits: Indices of the qubits to be reset. It should be a ``tuple`` of ``(party_id, qubit_id)``,
                or a ``list`` of it.

        Raises:
            ValueError: Party's ID should be ``str`` or ``int``.
            ValueError: The ``state`` should be ``LoccState`` or a ``list`` of it.

        Returns:
           LOCC state after resetting the state of part of the qubits.
        """
        # TODO: which_qubits -> qubits_idx?
        if isinstance(which_qubits, tuple):
            which_qubits = [which_qubits]
        qubits_list = []
        for party_id, qubit_id in which_qubits:
            if isinstance(party_id, str):
                qubits_list.append(self.parties_by_name[party_id][qubit_id])
            elif isinstance(party_id, int):
                qubits_list.append(self.parties_by_number[party_id][qubit_id])
            else:
                raise ValueError
        m = len(qubits_list)
        if isinstance(status, LoccState):
            n = status.num_qubits
        elif isinstance(status, list):
            n = status[0].num_qubits
        else:
            raise ValueError("can't recognize the input status")
        assert max(qubits_list) <= n, "qubit index out of range"

        origin_seq = list(range(0, n))
        target_seq = [idx for idx in origin_seq if idx not in qubits_list]
        target_seq = qubits_list + target_seq

        swapped = [False] * n
        swap_list = []
        for idx in range(0, n):
            if not swapped[idx]:
                next_idx = idx
                swapped[next_idx] = True
                while not swapped[target_seq[next_idx]]:
                    swapped[target_seq[next_idx]] = True
                    swap_list.append((next_idx, target_seq[next_idx]))
                    next_idx = target_seq[next_idx]

        cir0 = paddle_quantum.ansatz.Circuit(n)
        for a, b in swap_list:
            cir0.swap([a, b])

        cir1 = paddle_quantum.ansatz.Circuit(n)
        swap_list.reverse()
        for a, b in swap_list:
            cir1.swap([a, b])

        if isinstance(status, LoccState):
            _state = cir0(status)
            state_data = self.__partial_trace(_state.data, 2**m, 2 ** (n - m), 1)
            state_data = paddle.kron(state.data, state_data)
            _state = _state.clone()
            _state.data = state_data
            _state = cir1(_state)
            new_status = _state
        elif isinstance(status, list):
            new_status = []
            for each_status in status:
                _state = cir0(each_status)
                state_data = self.__partial_trace(_state.data, 2**m, 2 ** (n - m), 1)
                state_data = paddle.kron(state.data, state_data)
                _state = _state.clone()
                _state.data = state_data
                _state = cir1(_state)
                _state = _state.clone()
                new_status.append(_state)
        else:
            # TODO: seems unnecessary
            raise ValueError("can't recognize the input status")

        return new_status

    def add_new_party(
        self, qubits_number: int, party_name: Optional[str] = None
    ) -> Union[int, str]:
        r"""Add a new LOCC party.

        Args:
            qubits_number: Number of qubits of the party.
            party_name: Name of the party. Defaults to ``None``.

        Note:
            You can use a string or a number as a party's ID. If a string is preferred,
            you can set ``party_name``; if a number is preferred, then you don't need to set ``party_name``
            and the party's index number will be automatically assigned.

        Raises:
            ValueError: The ``party_name`` should be ``str``.

        Returns:
            ID of the party.
        """
        party_id = None
        if party_name is None:
            party_id = party_name
            party_name = str(len(self.parties_by_name))
        elif isinstance(party_name, str) is False:
            raise ValueError
        if party_id is None:
            party_id = len(self.parties_by_name)

        new_party = LoccParty(qubits_number)
        self.parties_by_name[party_name] = new_party
        self.parties_by_number.append(new_party)

        return party_id

    def create_ansatz(self, party_id: Union[int, str]) -> LoccAnsatz:
        r"""Create a new local ansatz.

        Args:
            party_id: Party's ID.

        Raises:
            ValueError: Party's ID should be ``str`` or ``int``.

        Returns:
            Created local ansatz.
        """
        if isinstance(party_id, int):
            return LoccAnsatz(self.parties_by_number[party_id])
        elif isinstance(party_id, str):
            return LoccAnsatz(self.parties_by_name[party_id])
        else:
            raise ValueError

    def __measure_parameterized(
        self,
        state_data: paddle.Tensor,
        which_qubits: Iterable,
        result_desired: str,
        theta: paddle.Tensor,
    ):
        r"""TODO Do parameterized measurement。

        Args:
            state_data (Tensor): The input quantum state
            which_qubits (list): The indices of qubits to be measured
            result_desired (str): The desired result
            theta (Tensor): The parameters of measurement

        Returns:
            Tensor: The quantum state collapsed after measurement
            Tensor: The probability of collapsing to the resulting state
            str: The result of measurement
        """
        n = self.get_num_qubits()
        assert len(which_qubits) == len(
            result_desired
        ), "the length of qubits wanted to be measured and the result desired should be same"
        op_list = [paddle.to_tensor(np.eye(2), dtype=paddle_quantum.get_dtype())] * n
        for idx in range(0, len(which_qubits)):
            i = which_qubits[idx]
            ele = result_desired[idx]
            if int(ele) == 0:
                basis0 = paddle.to_tensor(
                    np.array([[1, 0], [0, 0]]), dtype=paddle_quantum.get_dtype()
                )
                basis1 = paddle.to_tensor(
                    np.array([[0, 0], [0, 1]]), dtype=paddle_quantum.get_dtype()
                )
                rho0 = paddle.multiply(basis0, paddle.cos(theta[idx]))
                rho1 = paddle.multiply(basis1, paddle.sin(theta[idx]))
                rho = paddle.add(rho0, rho1)
                op_list[i] = rho
            elif int(ele) == 1:
                # rho = diag(concat([cos(theta[idx]), sin(theta[idx])]))
                # rho = paddle.to_tensor(rho, zeros((2, 2), dtype="float64"))
                basis0 = paddle.to_tensor(
                    np.array([[1, 0], [0, 0]]), dtype=paddle_quantum.get_dtype()
                )
                basis1 = paddle.to_tensor(
                    np.array([[0, 0], [0, 1]]), dtype=paddle_quantum.get_dtype()
                )
                rho0 = paddle.multiply(basis0, paddle.sin(theta[idx]))
                rho1 = paddle.multiply(basis1, paddle.cos(theta[idx]))
                rho = paddle.add(rho0, rho1)
                op_list[i] = rho
            else:
                print("cannot recognize the result_desired.")
            # rho = paddle.to_tensor(ones((2, 2), dtype="float64"), zeros((2, 2), dtype="float64"))
        measure_operator = paddle.to_tensor(op_list[0])
        if n > 1:
            for idx in range(1, len(op_list)):
                measure_operator = paddle.kron(measure_operator, op_list[idx])
        state_measured = paddle.matmul(
            paddle.matmul(measure_operator, state_data),
            paddle_quantum.linalg.dagger(measure_operator),
        )
        prob = paddle.real(
            paddle.trace(
                paddle.matmul(
                    paddle.matmul(
                        paddle_quantum.linalg.dagger(measure_operator), measure_operator
                    ),
                    state_data,
                )
            )
        )
        state_measured = paddle.divide(state_measured, prob)
        return state_measured, prob, result_desired

    def __measure_parameterless(
        self, state: paddle.Tensor, which_qubits: Iterable, result_desired: str
    ):
        r"""TODO Do 01 measurement。

        Args:
            state (Tensor): The input quantum state
            which_qubits (list): The indices of qubits to be measured
            result_desired (str): The desired result

        Returns:
            Tensor: The quantum state after measurement
            Tensor: The probability of collapsing to the resulting state
            str: The result of measurement
        """
        n = self.get_num_qubits()
        assert len(which_qubits) == len(
            result_desired
        ), "the length of qubits wanted to be measured and the result desired should be same"
        op_list = [np.eye(2)] * n
        for i, ele in zip(which_qubits, result_desired):
            k = int(ele)
            rho = np.zeros((2, 2))
            rho[int(k), int(k)] = 1
            op_list[i] = rho
        if n > 1:
            measure_operator = paddle.to_tensor(
                functools.reduce(np.kron, op_list), dtype=paddle_quantum.get_dtype()
            )
        else:
            measure_operator = paddle.to_tensor(
                op_list[0], dtype=paddle_quantum.get_dtype()
            )
        state_measured = paddle.matmul(
            paddle.matmul(measure_operator, state),
            paddle_quantum.linalg.dagger(measure_operator),
        )
        prob = paddle.real(
            paddle.trace(
                paddle.matmul(
                    paddle.matmul(
                        paddle_quantum.linalg.dagger(measure_operator), measure_operator
                    ),
                    state,
                )
            )
        )
        state_measured = paddle.divide(state_measured, prob)
        return state_measured, prob, result_desired

    def measure(
        self,
        status: Union[List[LoccState], LoccState],
        which_qubits: Iterable,
        results_desired: Union[List[str], str],
        theta: paddle.Tensor = None,
    ) -> Union[List[LoccState], LoccState]:
        r"""Perform 0-1 measurement or parameterized measurement on an LOCC state.

        Args:
            status: LOCC state to be measured.
            which_qubits: Indices of the qubits to be measured.
            results_desired: Expected measurement outcomes.
            theta: Parameters of measurement. Defaults to ``None``, which means 0-1 measurement.

        Raises:
            ValueError: The ``results_desired`` should be ``str`` or a ``list`` of it.
            ValueError: Party's ID should be ``str`` or ``int``.
            ValueError: The ``status`` should be ``LoccState`` or a ``list`` of it.

        Returns:
            LOCC state after measurement.
        """
        # TODO: names of status, which_qubits
        if isinstance(which_qubits, tuple):
            which_qubits = [which_qubits]
        if isinstance(results_desired, str):
            results_desired = [results_desired]
        elif not isinstance(results_desired, list):
            raise ValueError("cannot recognize the input of results_desired")

        qubits_list = []
        for party_id, qubit_id in which_qubits:
            if isinstance(party_id, int):
                qubits_list.append((self.parties_by_number[party_id][qubit_id]))
            elif isinstance(party_id, str):
                qubits_list.append((self.parties_by_name[party_id][qubit_id]))
            else:
                raise ValueError

        if isinstance(status, LoccState):
            existing_result = status.measured_result
            prior_prob = status.prob
            new_status = []
            for result_desired in results_desired:
                if theta is None:
                    result_measured = self.__measure_parameterless(
                        status.data, qubits_list, result_desired
                    )
                else:
                    result_measured = self.__measure_parameterized(
                        status.data, qubits_list, result_desired, theta
                    )
                state_data, prob, res = result_measured
                _state = status.clone()
                _state.data = state_data
                _state.num_qubits = int(math.log2(state_data.shape[-1]))
                _state.prob = prior_prob * prob
                _state.measured_result = existing_result + res
                new_status.append(_state)
            if len(new_status) == 1:
                new_status = new_status[0]
        elif isinstance(status, list):
            new_status = []
            for each_status in status:
                existing_result = each_status.measured_result
                prior_prob = each_status.prob
                for result_desired in results_desired:
                    if theta is None:
                        result_measured = self.__measure_parameterless(
                            each_status.state, qubits_list, result_desired
                        )
                    else:
                        result_measured = self.__measure_parameterized(
                            each_status.state, qubits_list, result_desired, theta
                        )
                    state_data, prob, res = result_measured
                    _state = each_status.clone()
                    _state.data = state_data
                    _state.num_qubits = int(math.log2(state_data.shape[-1]))
                    _state.prob = prior_prob * prob
                    _state.measured_result = existing_result + res
                    new_status.append(_state)
        else:
            raise ValueError("can't recognize the input status")

        return new_status

    def get_num_qubits(self) -> int:
        r"""Get the number of the qubits in this LOCCNet.

        Returns:
            The number of qubits in LOCCNet.
        """
        num_qubits = 0
        for party in self.parties_by_number:
            num_qubits += party.num_qubits
        return num_qubits
