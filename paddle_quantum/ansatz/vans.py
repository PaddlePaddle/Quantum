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
The source file of the variable ansatz.
"""

import numpy as np
from sympy import *
import paddle
from paddle_quantum.ansatz import Circuit
from paddle_quantum.gate import RX, RZ, CNOT
from paddle_quantum.linalg import is_unitary
from typing import Callable, List, Any, Optional


class Inserter:
    r"""Class for block insertion for the circuit.
    """
    @classmethod
    def insert_identities(cls, cir: Circuit, insert_rate: float, epsilon: float) -> Circuit:
        r"""Insert identity blocks to the current circuit, according to the insert rate.

        Args:
            cir: Quantum circuit to be simplified.
            insert_rate: Rate of number of inserted blocks.
            epsilon: Range of random initialization of parameters.

        Returns:
            Inserted circuit.
        """
        num_id = int(np.random.exponential(scale=insert_rate)) + 1
        theta = paddle.uniform(
            shape=[num_id, 6], dtype="float32", min=-epsilon, max=epsilon,)
        for i in range(num_id):
            cir = cls.__place_identity_block(cir, theta[i])
        return cir

    @classmethod
    def __place_identity_block(cls, cir: Circuit, theta: paddle.Tensor) -> Circuit:
        r"""Insert a set of gates to the circuit.

        Args:
            cir: Quantum circuit.
            theta: Parameters for inserted gates.

        Returns:
            Output circuit.
        """
        # count the number of single- and two-qubit gates in the circuit
        count_current = cls.__count_qubit_gates(cir)
        # get the inverse, use it as the probability
        inv_count_current = np.array(
            [1 / i for i in count_current])  # as probability
        total_num = (cir.num_qubits ** 2 + cir.num_qubits) // 2
        # get the int, which indicate which set to insert
        p = inv_count_current / inv_count_current.sum()
        index = np.random.choice(range(total_num), p=p.ravel())

        qubit_history = cir.qubit_history
        if index < cir.num_qubits:
            # get the position to insert in the circuit
            cir_qubit_history_i = qubit_history[index]
            if len(cir_qubit_history_i) == 0:
                idx_qubit = 0
                insert_ind = 0
            else:
                idx_qubit = np.random.randint(0, len(cir_qubit_history_i))
                insert_ind = cir_qubit_history_i[idx_qubit][1]
            # add one qubit gates rz_rx_rz to the circuit
            cir.insert(insert_ind, RZ([index], param=theta[0]))
            cir.insert(insert_ind + 1, RX([index], param=theta[1]))
            cir.insert(insert_ind + 2, RZ([index], param=theta[2]))
        else:
            # add two qubit gates to the circuit
            # obtain which two qubits to insert
            curr = cir.num_qubits
            count = 0
            while index - curr >= 0:
                index -= curr
                curr -= 1
                count += 1
            qubit_i = count - 1
            qubit_j = qubit_i + index + 1

            # get insertion position
            cir_qubit_history_i = qubit_history[qubit_i]
            cir_qubit_history_j = qubit_history[qubit_j]
            if len(cir_qubit_history_i) + len(cir_qubit_history_j) == 0:
                idx_qubit = 0
                insert_ind = 0
            elif len(cir_qubit_history_i) == 0 and len(cir_qubit_history_j) == 1:
                idx_qubit = 0
                insert_ind = cir_qubit_history_j[0][1]
            elif len(cir_qubit_history_i) == 1 and len(cir_qubit_history_j) == 0:
                idx_qubit = 0
                insert_ind = cir_qubit_history_i[0][1]
            else:
                idx_qubit = np.random.randint(
                    0, len(cir_qubit_history_i) + len(cir_qubit_history_j) - 1
                )
                if idx_qubit < len(cir_qubit_history_i):
                    insert_ind = cir_qubit_history_i[idx_qubit][1]
                else:
                    insert_ind = cir_qubit_history_j[idx_qubit -
                                                     len(cir_qubit_history_i)][1]
            # add gates
            cir.insert(insert_ind, CNOT([qubit_i, qubit_j]))
            cir.insert(insert_ind + 1, RZ([qubit_i], param=theta[0]))
            cir.insert(insert_ind + 2, RX([qubit_i], param=theta[1]))
            cir.insert(insert_ind + 3, RZ([qubit_i], param=theta[2]))
            cir.insert(insert_ind + 4, RX([qubit_j], param=theta[3]))
            cir.insert(insert_ind + 5, RZ([qubit_j], param=theta[4]))
            cir.insert(insert_ind + 6, RX([qubit_j], param=theta[5]))
            cir.insert(insert_ind + 7, CNOT([qubit_i, qubit_j]))
        return cir

    @classmethod
    def __count_qubit_gates(cls, cir: Circuit) -> np.ndarray:
        r"""Count the number of single-qubit and two-qubit gates.

        Args:
            cir: Quantum circuit.

        Returns:
            List with information of number of different gates.
        """
        # The previous n nums count rotation gates at each qubit, and the next (n^2-n)/2 nums count cnot gates.
        count_gates = np.zeros((cir.num_qubits ** 2 + cir.num_qubits) // 2)
        # initialize to 0.1 in case there's no gates, which will lead to divide by zero error
        count_gates = [0.1 for _ in count_gates]
        history = cir.gate_history
        for gate_info in history:
            qubits_idx = gate_info["which_qubits"]
            if gate_info["gate"] == "rz" or gate_info["gate"] == "rx":
                qubit_ind = qubits_idx
                count_gates[qubit_ind] += 1
            elif gate_info["gate"] == "cnot":
                qubit_i = min(qubits_idx[0], qubits_idx[1])
                qubit_j = max(qubits_idx[0], qubits_idx[1])
                idx = (2 * cir.num_qubits - qubit_i) * \
                    (qubit_i + 1) // 2 + qubit_j - qubit_i - 1
                count_gates[idx] += 1
        return count_gates


class Simplifier:
    r"""Class for circuit simplification.
    """
    @classmethod
    def __check_cnot_init(cls, cir: Circuit) -> bool:
        r"""For rule 1, check if there are CNOTs in the front of the circuit.

        Args:
            cir: Quantum circuit.

        Returns:
            Determine whether there are CNOTs in the front of the circuit.
        """
        count = 0
        qubit_history = cir.qubit_history
        for i in range(cir.num_qubits):
            history_i = qubit_history[i]
            if not history_i:
                continue
            if history_i[0][0]["gate"] == "cnot":
                cnot_qubits = history_i[0][0]["which_qubits"]
                # find the other qubit
                for j in cnot_qubits:
                    if j != i:
                        # check the CNOT is also in the front for the other qubit
                        if (
                            qubit_history[j][0][0]["gate"] == "cnot"
                            and qubit_history[j][0][0]["which_qubits"]
                            == cnot_qubits
                        ):
                            count += 1
        if count == 0:
            return True
        else:
            return False

    @classmethod
    def __check_consec_cnot(cls, cir: Circuit) -> bool:
        r"""For rule 2, check if there are consecutive CNOTs on the same qubits.

        Args:
            cir: Quantum circuit.

        Returns:
            Determine whether there are consecutive CNOTs on the same qubits.
        """
        count = 0
        qubit_history = cir.qubit_history
        for i in range(cir.num_qubits):
            history_i = qubit_history[i]
            if not history_i:
                continue
            for j in range(len(history_i) - 1):
                # find consecutive cnots on one qubit
                if (
                    history_i[j][0]["gate"] == "cnot"
                    and history_i[j + 1][0] == history_i[j][0]
                ):
                    cnot_qubits = history_i[j][0]["which_qubits"]
                    # get the other qubit
                    k = list(set(cnot_qubits).difference(set([i])))[0]
                    # check if the found consecutive cnots are also consecutive on the other qubit
                    history_k = qubit_history[k]
                    idx_k = history_k.index(history_i[j])
                    if history_k[idx_k + 1] == history_i[j + 1]:
                        count += 1
        if count == 0:
            return True
        else:
            return False

    @classmethod
    def __check_rz_init(cls, cir: Circuit) -> bool:
        r"""For rule 3, check if there are rz in the front of the circuit.

        Args:
            cir: Quantum circuit.

        Returns:
            Determine whether there are rz in the front of the circuit.
        """
        count = 0
        qubit_history = cir.qubit_history
        for i in range(cir.num_qubits):
            history_i = qubit_history[i]
            if not history_i:
                continue
            if history_i[0][0]["gate"] == "z" or history_i[0][0]["gate"] == "rz":
                count += 1
        if count == 0:
            return True
        else:
            return False

    @classmethod
    def __check_repeated_rotations(cls, cir: Circuit) -> bool:
        r"""For rule 4, check if there are consecutive rx's or rz's.

        Args:
            cir: Quantum circuit.

        Returns:
            Determine whether there are consecutive rx's or rz's.
        """
        count = 0
        cir_qubit_history = cir.qubit_history
        for i in range(cir.num_qubits):
            history_i = cir_qubit_history[i]
            if not history_i:
                continue
            for j in range(len(history_i) - 1):
                if (
                    history_i[j][0]["gate"] == "rx"
                    and history_i[j + 1][0]["gate"] == "rx"
                ) or (
                    history_i[j][0]["gate"] == "rz"
                    and history_i[j + 1][0]["gate"] == "rz"
                ):
                    count += 1
        if count == 0:
            return True
        else:
            return False

    @classmethod
    def __check_4_consec_rotations(cls, cir: Circuit) -> bool:
        r"""For rule 5, check if there are consecutive rotations on one qubits.

        Args:
            cir: Quantum circuit.

        Returns:
            Determine whether there are consecutive rotations on one qubits e.g., rx_rz_rx_rz & rz_rx_rz_rx.
        """
        count = 0
        cir_qubit_history = cir.qubit_history
        for i in range(cir.num_qubits):
            history_i = cir_qubit_history[i]
            if not history_i:
                continue
            for j in range(len(history_i) - 3):
                if (
                    history_i[j][0]["gate"] == "rx"
                    and history_i[j + 1][0]["gate"] == "rz"
                    and history_i[j + 2][0]["gate"] == "rx"
                    and history_i[j + 3][0]["gate"] == "rz"
                ) or (
                    history_i[j][0]["gate"] == "rz"
                    and history_i[j + 1][0]["gate"] == "rx"
                    and history_i[j + 2][0]["gate"] == "rz"
                    and history_i[j + 3][0]["gate"] == "rx"
                ):
                    count += 1
        if count == 0:
            return True
        else:
            return False

    @classmethod
    def __check_rz_cnot_rz_rx_cnot_rx(cls, cir: Circuit) -> bool:
        r"""For rule 6, check if there are rz_cnot_rz or rx_cnot_rx.

        Args:
            cir: Quantum circuit.

        Returns:
            Determine whether there are rz_cnot_rz(control) or rx_cnot_rx(target).
        """
        count = 0
        qubit_history = cir.qubit_history
        for i in range(cir.num_qubits):
            history_i = qubit_history[i]
            if not history_i:
                continue
            for j in range(len(history_i) - 2):
                # find rz_cnot_rz with rz on the control qubit or rx_cnot_rx with rx on the target qubit
                if (
                    history_i[j][0]["gate"] == "rz"
                    and history_i[j + 1][0]["gate"] == "cnot"
                    and history_i[j + 2][0]["gate"] == "rz"
                    and history_i[j + 1][0]["which_qubits"][0] == i
                ) or (
                    history_i[j][0]["gate"] == "rx"
                    and history_i[j + 1][0]["gate"] == "cnot"
                    and history_i[j + 2][0]["gate"] == "rx"
                    and history_i[j + 1][0]["which_qubits"][1] == i
                ):
                    count += 1
        if count == 0:
            return True
        else:
            return False

    @classmethod
    def __rule_1(cls, cir: Circuit) -> Circuit:
        r"""Simplify circuit when circuit has a CNOT just after :math:`|0\rangle` initialization.

        Args:
            cir: Quantum circuit.

        Returns:
            Simplified circuit.
        """
        while not cls.__check_cnot_init(cir):
            for i in range(cir.num_qubits):
                qubit_history = cir.qubit_history
                history_i = qubit_history[i]
                if not history_i:
                    continue
                if history_i[0][0]["gate"] == "cnot":
                    cnot_qubits = history_i[0][0]["which_qubits"]
                    # find the other qubit
                    for j in cnot_qubits:
                        if j != i:
                            # check the CNOT is also in the front for the other qubit
                            if (
                                qubit_history[j][0][0]["gate"] == "cnot"
                                and qubit_history[j][0][0]["which_qubits"]
                                == cnot_qubits
                            ):
                                # delete the gate
                                cir.pop(qubit_history[j][0][1])
                        qubit_history = cir.qubit_history
                        history_i = cir.qubit_history[i]
        return cir

    @classmethod
    def __rule_2(cls, cir: Circuit) -> Circuit:
        r"""Simplify circuit when circuit has 2 consecutive and equal CNOTs compile to identity.

        Args:
            cir: Quantum circuit.

        Returns:
            Simplified circuit.

        """
        while not cls.__check_consec_cnot(cir):
            for i in range(cir.num_qubits):
                qubit_history = cir.qubit_history
                history_i = qubit_history[i]
                if not history_i:
                    continue
                for j in range(len(history_i) - 1):
                    if j >= len(history_i) - 1:
                        break
                    # find consecutive cnots on one qubit
                    if (
                        history_i[j][0]["gate"] == "cnot"
                        and history_i[j + 1][0] == history_i[j][0]
                    ):
                        cnot_qubits = history_i[j][0]["which_qubits"]
                        # get the other qubit
                        k = list(set(cnot_qubits).difference(set([i])))[0]
                        # check if the found consecutive cnots are also consecutive on the other qubit
                        history_k = qubit_history[k]
                        idx_k = history_k.index(history_i[j])
                        if history_k[idx_k + 1] == history_i[j + 1]:
                            # delete found consecutive cnots from circuit history
                            cir.pop(history_i[j][1])
                            cir.pop(history_i[j][1])
                    qubit_history = cir.qubit_history
                    history_i = cir.qubit_history[i]
        return cir

    @classmethod
    def __rule_3(cls, cir: Circuit) -> Circuit:
        r"""Simplify circuit when circuit has rotations around z axis after initialization, which leaves invariant <H>.

        Args:
            cir: Quantum circuit.

        Returns:
            Simplified circuit.

        """
        while not cls.__check_rz_init(cir):
            for i in range(cir.num_qubits):
                history_i = cir.qubit_history[i]
                if not history_i:
                    continue
                if history_i[0][0]["gate"] == "z" or history_i[0][0]["gate"] == "rz":
                    # delete from history
                    cir.pop(history_i[0][1])
        return cir

    @classmethod
    def __rule_4(cls, cir: Circuit) -> Circuit:
        r"""Simplify circuit when circuit has repeated rotations (rz_rz -> r_z & rx_rx -> r_x).

        Args:
            cir: Quantum circuit.

        Returns:
            Simplified circuit.
        """
        while not cls.__check_repeated_rotations(cir):
            for i in range(cir.num_qubits):
                history_i = cir.qubit_history[i]
                if not history_i:
                    continue
                for j in range(len(history_i) - 1):
                    if j == len(history_i) - 1:
                        break
                    # find consecutive rx/rz's
                    gates_name = [history_i[j + k][0]["gate"]
                                  for k in range(2)]
                    if (gates_name[0] == "rx" and gates_name[1] == "rx") or \
                            (gates_name[0] == "rz" and gates_name[1] == "rz"):
                        # get indexes of gates
                        idx = [history_i[j + k][1] - k for k in range(2)]
                        theta = history_i[j][0]["theta"] + \
                            history_i[j + 1][0]["theta"]

                        # delete two rx/rz
                        cir.pop(idx[0])
                        cir.pop(idx[1])

                        # add values together for a new rx/rz
                        cir.insert(idx[0], RX(
                            i, param=theta) if gates_name[0] == "rx" else RZ(i, param=theta))

                    # update qubit_history of ith qubit
                    history_i = cir.qubit_history[i]
        return cir

    @classmethod
    def __calculate_new_angle(cls, theta: List[paddle.Tensor], xzxz: bool) -> paddle.Tensor:
        r"""Calculate three new angles according to four old angles.

        Args:
            theta: Four angles correspoding to rx, rz, rx, rz or rz, rx, rz, rx.
            xzxz: Determine whether it is rx, rz, rx, rz or rz, rx, rz, rx.

        Returns:
            Three new angles for rz_rx_rz given the unitary matrix.
        """
        assert len(theta) == 4, str(len(theta))

        if paddle.max(paddle.abs(theta - paddle.zeros([4]))).item() < 1e-5:
            return np.array([0, 0, 0])

        def rx(phi: paddle.Tensor) -> paddle.Tensor:
            r"""A temporary function for rx generation.

            Args:
                phi: Parameter input for rx.

            Returns:
                Matrix representation for rx gate.
            """
            gate = [paddle.cos(phi / 2).cast('complex64'), -1j * paddle.sin(phi / 2).cast('complex64'),
                    -1j * paddle.sin(phi / 2).cast('complex64'), paddle.cos(phi / 2).cast('complex64')]
            return paddle.concat(gate).reshape([2, 2])

        def rz(phi: paddle.Tensor) -> paddle.Tensor:
            r"""A temporary function for rz generation.

            Args:
                phi: Parameter input for rz.

            Returns:
                Matrix representation for rz gate.
            """
            gate = [paddle.cos(phi / 2).cast('complex128') - 1j * paddle.sin(phi / 2).cast('complex128'), paddle.to_tensor(0, dtype='complex128'),
                    paddle.to_tensor(0, dtype='complex128'), paddle.cos(phi / 2).cast('complex128') + 1j * paddle.sin(phi / 2).cast('complex128')]
            return paddle.concat(gate).reshape([2, 2])

        if xzxz:
            unitary = rx(theta[0]) @ rz(theta[1]) @ rx(theta[2]) @ rz(theta[3])
        else:
            unitary = rz(theta[0]) @ rx(theta[1]) @ rz(theta[2]) @ rx(theta[3])

        x, y, z = symbols("x y z")
        expr_list = [
            exp(-I * (x + z) / 2) * cos(y / 2),
            -I * exp(-I * (x - z) / 2) * sin(y / 2),
            exp(I * (z + x) / 2) * cos(y / 2),
        ]

        element = np.reshape(unitary.numpy(), (4,))
        val_list = []
        for i, elem in enumerate(element):
            if i != 2:
                val_list.append(elem)

        eqs = []
        for expr, val in zip(expr_list, val_list):
            eqs.append(expr - val)

        err = True
        times = 0
        max_steps = 1000
        while err:
            try:
                solved_value_zxz = nsolve(
                    eqs,
                    [x, y, z],
                    2 * np.pi * np.array([np.random.random(),
                                          np.random.random(), np.random.random()]),
                    maxsteps=max_steps,
                    verify=True,
                )
                res = np.array([complex(item) for item in solved_value_zxz])
                err = False
            except:
                max_steps += 20
                times = times + 1
                if times >= 10:
                    err = False
                else:
                    res = 2 * np.pi * \
                        np.array(
                            [np.random.random(), np.random.random(), np.random.random()])
                    err = False
        return res

    @classmethod
    def __rule_5(cls, cir: Circuit) -> Circuit:
        r"""Simplify circuit when circuit absorbs more than three consecutive rotations on one qubit into rz_rx_rz
        we do rx_rz_rx_rz & rz_rx_rz_rx -> rz_rx_rz

        Args:
            cir: Quantum circuit.

        Returns:
            Simplified circuit.
        """
        while not cls.__check_4_consec_rotations(cir):
            for i in range(cir.num_qubits):
                history_i = cir.qubit_history[i]
                if not history_i:
                    continue
                for j in range(len(history_i) - 3):
                    if j >= len(history_i) - 3:
                        break

                    gates_name = [history_i[j + k][0]["gate"]
                                  for k in range(4)]
                    # find 4 consecutive rotations rx_rz_rx_rz or rz_rx_rz_rx
                    if (gates_name[0] == "rx"
                        and gates_name[1] == "rz"
                        and gates_name[2] == "rx"
                        and gates_name[3] == "rz") or (
                        gates_name[0] == "rz"
                        and gates_name[1] == "rx"
                        and gates_name[2] == "rz"
                        and gates_name[3] == "rx"
                    ):
                        # get the new angles for rz_rx_rz, assign the new angles to the last three rotations
                        theta = paddle.concat(
                            [history_i[j + k][0]["theta"] for k in range(4)]).cast('float64')
                        new_angles = np.real(cls.__calculate_new_angle(
                            theta, gates_name[0] == "rx"))

                        # get indexes of gates
                        idx = [history_i[j + k][1] - k for k in range(4)]

                        # remove the consecutive rotations
                        for k in range(4):
                            cir.pop(idx[k])
                        # add rz_rx_rz rotations
                        cir.insert(idx[0], RZ(i, new_angles[2]))
                        cir.insert(idx[1] + 1, RX(i, new_angles[1]))
                        cir.insert(idx[2] + 2, RZ(i, new_angles[0]))

                        # update qubit_history of ith qubit
                        history_i = cir.qubit_history[i]
        return cir

    @classmethod
    def __rule_6(cls, cir: Circuit) -> Circuit:
        r"""Simplify circuit when circuit has Rz(control) and CNOT(control, target) Rz(control) or 
        Rx(target) and CNOT(control, target) Rx(target). We map these gates to Rz/x(control) CNOT.

        Args:
            cir: Quantum circuit.

        Returns:
            Simplified circuit.
        """
        while not cls.__check_rz_cnot_rz_rx_cnot_rx(cir):
            for i in range(cir.num_qubits):
                history_i = cir.qubit_history[i]
                if not history_i:
                    continue
                for j in range(len(history_i) - 2):
                    if j == len(history_i) - 2:
                        break
                    if (history_i[j][0]["gate"] == "rz"
                            and history_i[j + 1][0]["gate"] == "cnot"
                            and history_i[j + 2][0]["gate"] == "rz"
                            and history_i[j + 1][0]["which_qubits"][0] == i) or (
                                history_i[j][0]["gate"] == "rx"
                            and history_i[j + 1][0]["gate"] == "cnot"
                            and history_i[j + 2][0]["gate"] == "rx"
                            and history_i[j + 1][0]["which_qubits"][1] == i):

                        idx = [history_i[j][1], history_i[j + 2][1]]
                        theta = history_i[j][0]["theta"] + \
                            history_i[j + 2][0]["theta"]

                        # delete two rx/z
                        cir.pop(idx[0])
                        cir.pop(idx[1] - 1)

                        # add values together for a new rx/z
                        cir.insert(idx[0], RX(i, param=theta) if history_i[j]
                                   [0]["gate"] == "rx" else RZ(i, param=theta))

                        # update qubit_history of ith qubit
                        history_i = cir.qubit_history[i]
        return cir

    @classmethod
    def simplify_circuit(cls, cir: Circuit, zero_init_state:  Optional[bool] = True) -> Circuit:
        r"""Combine all simplifications together.

        Args:
            cir: Quantum circuit.
            zero_init_state: Whether the initial state is :math:`|0\rangle`. Defaults to ``True``.

        Returns:
            Simplified circuit.
        """
        # Check if the initial state is the zero state
        if zero_init_state:
            while not (
                cls.__check_cnot_init(cir)
                and cls.__check_consec_cnot(cir)
                and cls.__check_rz_init(cir)
                and cls.__check_repeated_rotations(cir)
                and cls.__check_4_consec_rotations(cir)
                and cls.__check_rz_cnot_rz_rx_cnot_rx(cir)
            ):
                cir = cls.__rule_1(cir)
                cir = cls.__rule_2(cir)
                cir = cls.__rule_3(cir)
                cir = cls.__rule_4(cir)
                cir = cls.__rule_5(cir)
                cir = cls.__rule_6(cir)
        else:
            while not (
                cls.__check_consec_cnot(cir)
                and cls.__check_repeated_rotations(cir)
                and cls.__check_4_consec_rotations(cir)
                and cls.__check_rz_cnot_rz_rx_cnot_rx(cir)
            ):
                cir = cls.__rule_2(cir)
                cir = cls.__rule_4(cir)
                cir = cls.__rule_5(cir)
                cir = cls.__rule_6(cir)

        return cir


def cir_decompose(cir: Circuit, trainable: Optional[bool] = False) -> Circuit:
    r"""Decompose all layers of circuit into gates, and make all parameterized gates trainable if needed

    Args:
        cir: Target quantum circuit.
        trainable: whether the decomposed parameterized gates are trainable

    Returns:
        A quantum circuit with same structure and parameters but all layers are decomposed into Gates.

    Note:
        This function does not support customized gates, such as oracle and control-oracle.
    """
    gates_history = cir.gate_history
    new_cir = Circuit()
    for gate_info in gates_history:
        gate_name = gate_info['gate']
        qubits_idx = gate_info['which_qubits']
        param = gate_info['theta']
        # get gate function
        if param is None:
            getattr(new_cir, gate_name)(qubits_idx)
            continue
        
        if trainable:
            param = param.reshape([1] + param.shape)
            param = paddle.create_parameter(
                shape=param.shape, dtype=param.dtype,
                default_initializer=paddle.nn.initializer.Assign(param))
        getattr(new_cir, gate_name)(qubits_idx, param=param)
    return new_cir


class VAns:
    r"""Class of Variable Ansatz.

    User can initialize this class to find variational circuit to construct QML model.

    Note:
        The loss function passed in must have the quantum circuit as its first parameter.

    Args:
        n: Number of qubits.
        loss_func: Loss function that evaluate the loss of circuit.
        loss_func_args: Parameters of loss function other than circuit.
        epsilon: Range of random initialization of parameters. Defaults to ``0.1``.
        insert_rate: Rate of number of inserted blocks. Defaults to ``2``.
        iter: Number of iterations of optimizer. Defaults to ``100``.
        iter_out: Number of insert-simplify cycles. Defaults to ``10``.
        LR: Learning rate. Defaults to ``0.1``.
        threshold: Tolerance of incread loss after deleting one qubit gate. Defaults to ``0.002``.
        accpet_wall: Percentage of probability of accepting the circuit in current insert-simplify cycle. Defaults to ``100``.
        zero_init_state: Whether the initial state is :math:`|0\rangle`. Defaults to ``True``.
    """
    def __init__(
        self,
        n: int,
        loss_func: Callable[[Circuit, Any], paddle.Tensor],
        *loss_func_args: Any,
        epsilon: float = 0.1,
        insert_rate: float = 2,
        iter: int = 100,
        iter_out: int = 10,
        LR: float = 0.1,
        threshold: float = 0.002,
        accept_wall: float = 100,
        zero_init_state: bool = True
    ):
        self.n = n
        self.loss_func = loss_func
        self.loss_args = loss_func_args
        self.epsilon = epsilon
        self.insert_rate = insert_rate
        self.cir = self.__init_cir(self.n)
        self.iter = iter
        self.iter_out = iter_out
        self.loss = float("inf")
        self.threshold = threshold
        self.accept_wall = accept_wall
        self.LR = LR
        self.zero_init_state = zero_init_state

    def __init_cir(self, N: int) -> Circuit:
        r"""Create a QNN circuit for VAns.

        Args: 
            N: Number of qubits.

        Return:
            A quantum circuit.
        """
        cir = Circuit(N)

        # simple version
        #     for i in range(N):
        #         cir.rx(i)

        # complicated version
        for i in range(N):
            cir.rx(i)
            cir.rz(i)
            if i % 2 == 0:
                cir.cnot([i, (i + 1) % N])
        for i in range(N):
            cir.rx(i)
            cir.rz(i)
            if i % 2 == 1:
                cir.cnot([i, (i + 1) % N])

        return cir

    def train(self) -> Circuit:
        r"""Train the quantum circuit.

        Returns:
            The quantum circuit with the lowest loss.
        """
        for out_itr in range(1, self.iter_out + 1):
            print("Out iteration", out_itr, "for structure optimization:")
            if out_itr == 1:
                itr_loss = self.optimization(self.cir)
                self.loss = itr_loss
            else:  # insert + simplification
                # Insert
                new_cir = cir_decompose(self.cir, trainable=True)
                new_cir = Inserter.insert_identities(
                    new_cir, self.insert_rate, self.epsilon)

                new_cir = Simplifier.simplify_circuit(
                    new_cir, self.zero_init_state)

                itr_loss = self.optimization(new_cir)

                relative_diff = (itr_loss - self.loss) / abs(itr_loss)

                if relative_diff <= 0 or np.random.random() <= np.exp(-relative_diff * self.accept_wall):
                    print("     accpet the new circuit!")

                    # Remove gates that do not lower the loss
                    new_cir = self.delete_gates(new_cir, itr_loss)
                    new_cir = Simplifier.simplify_circuit(
                        new_cir, self.zero_init_state)
                    itr_loss = self.loss_func(new_cir, *self.loss_args).item()
                    self.cir = new_cir
                    self.loss = itr_loss
                else:
                    print("     decline the new circuit!")

            print(" Current loss:", self.loss)
            print(" Current cir:")
            print(self.cir, '\n')

        # Output the results
        print("\n")
        print("The final loss:", self.loss)
        print("The final circuit:")
        print(self.cir)

    def optimization(self, cir: Circuit) -> float:
        r"""Optimize circuit parameters with loss function.

        Args:
            cir: Quantum circuit.

        Returns:
            Optimized loss.
        """
        cir = cir_decompose(cir, trainable=True)

        opt = paddle.optimizer.Adam(
            learning_rate=self.LR, parameters=cir.parameters())

        for itr in range(1, self.iter+1):
            loss = self.loss_func(cir, *self.loss_args)
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
            if itr % 20 == 0:
                print("iter:", itr, "loss:", loss.numpy())

        return loss.numpy()

    def delete_gates(self, cir: Circuit, loss: float) -> Circuit:
        r"""Remove single qubit gates if the loss decreases or increases within a threshold.

        Args:
            cir: Quantum circuit.
            loss: Current loss.

        Returns:
            Circuit after deleting unnecessary gates.
        """
        count_delete = 0
        print("     start deleting gates")
        gate_history = cir.gate_history
        for idx, gate in enumerate(gate_history):
            candidate_cir = cir_decompose(cir)
            idx_true = idx - count_delete
            if gate["gate"] != "cnot":
                candidate_cir.pop(idx_true)
                candidate_loss = self.loss_func(
                    candidate_cir, *self.loss_args).item()

                if candidate_loss <= loss or (candidate_loss - loss) / abs(self.loss) <= self.threshold:
                    print("         Deletion: accept deletion with acceptable loss")
                    loss = candidate_loss
                    cir = candidate_cir
                    count_delete += 1
                    continue
                print("         Deletion: reject deletion")
        print("     ", count_delete, " gates are deleted!")
        return cir


if __name__ == "__main__":
    r"""
    A simple application on VQE
    """
    import paddle_quantum.qchem as qchem
    from paddle_quantum import set_backend, Hamiltonian

    # Construct a Hamiltonian
    geo = qchem.geometry(
        structure=[["H", [-0.0, 0.0, 0.0]], ["H", [-0.0, 0.0, 0.74]]])
    molecule = qchem.get_molecular_data(
        geometry=geo,
        basis="sto-3g",
        charge=0,
        multiplicity=1,
        method="fci",
        if_save=True,
        if_print=False,
    )

    molecular_hamiltonian = qchem.spin_hamiltonian(
        molecule=molecule,
        filename=None,
        multiplicity=1,
        mapping_method="jordan_wigner",
    )

    n = molecular_hamiltonian.n_qubits

    # Define a loss function, note that the first parameter should be an Circuit
    def loss_f(cir: Circuit, H: Hamiltonian) -> paddle.Tensor:
        return cir().expec_val(H)

    # Construct an instance of VAns, using n, loss function and other parameters.
    set_backend(backend='state_vector')
    vans = VAns(
        n,
        loss_f,
        Hamiltonian(molecular_hamiltonian.pauli_str),
        insert_rate=1,
        iter_out=10,
        zero_init_state=true,
    )

    # Train
    vans.train()
