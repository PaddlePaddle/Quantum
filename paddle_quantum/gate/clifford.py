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
Class for randomly generating a Clifford operator.
"""

import numpy as np
import paddle_quantum

__all__ = [
    "Clifford",
    "compose_clifford_circuit"
]


def _random_clifford(num_qubits):
    assert (num_qubits <= 100), "too many qubits"

    # Some constant matrices
    bigzero = np.zeros((2 * num_qubits, 2 * num_qubits), dtype=int)
    nzero = np.zeros((num_qubits, num_qubits), dtype=int)
    id_mat = np.identity(num_qubits, dtype=int)

    h, s = _sample_qmallows(num_qubits)
    gamma1 = np.copy(nzero)
    delta1 = np.copy(id_mat)
    gamma2 = np.copy(nzero)
    delta2 = np.copy(id_mat)

    for i in range(num_qubits):
        gamma2[i, i] = np.random.randint(2)
        if h[i] == 1:
            gamma1[i, i] = np.random.randint(2)

    # Constraints for canonical form
    for j in range(num_qubits):
        for i in range(j + 1, num_qubits):
            b = np.random.randint(2)
            gamma2[i, j] = b
            gamma2[j, i] = b
            delta2[i, j] = np.random.randint(2)
            if h[i] == 1 and h[j] == 1:
                b = np.random.randint(2)
                gamma1[i, j] = b
                gamma1[j, i] = b
            if h[i] == 1 and h[j] == 0 and s[i] < s[j]:
                b = np.random.randint(2)
                gamma1[i, j] = b
                gamma1[j, i] = b
            if h[i] == 0 and h[j] == 1 and s[i] > s[j]:
                b = np.random.randint(2)
                gamma1[i, j] = b
                gamma1[j, i] = b
            if h[i] == 0 and h[j] == 1:
                delta1[i, j] = np.random.randint(2)

            if h[i] == 1 and h[j] == 1 and s[i] > s[j]:
                delta1[i, j] = np.random.randint(2)

            if h[i] == 0 and h[j] == 0 and s[i] < s[j]:
                delta1[i, j] = np.random.randint(2)

    # Compute stabilizer table
    st1 = np.matmul(gamma1, delta1)
    st2 = np.matmul(gamma2, delta2)
    inv1 = np.linalg.inv(np.transpose(delta1))
    inv2 = np.linalg.inv(np.transpose(delta2))
    f_1 = np.block([[delta1, nzero], [st1, inv1]])
    f_2 = np.block([[delta2, nzero], [st2, inv2]])
    f_1 = f_1.astype(int) % 2
    f_2 = f_2.astype(int) % 2
    unitary = np.copy(bigzero)

    for i in range(num_qubits):
        unitary[i, :] = f_2[s[i], :]
        unitary[i + num_qubits, :] = f_2[s[i] + num_qubits, :]

    # Apply Hadamard layer to the stabilizer table
    for i in range(num_qubits):
        if h[i] == 1:
            unitary[(i, i + num_qubits), :] = unitary[(i + num_qubits, i), :]

    gamma = [gamma1, gamma2]
    delta = [delta1, delta2]
    return np.matmul(f_1, unitary) % 2, gamma, delta, h, s


def _sample_qmallows(num_qubits):
    # Hadamard layer
    h = np.zeros(num_qubits, dtype=int)
    # S layer
    s = np.zeros(num_qubits, dtype=int)
    a = list(range(0, num_qubits))

    for i in range(0, num_qubits):
        m = num_qubits - i
        r = np.random.uniform(0, 1)
        index = int(2 * m - np.ceil(np.log(r * (4 ** m - 1) + 1) / np.log(2.0)))
        h[i] = 1 * (index < m)
        if index < m:
            k = index
        else:
            k = 2 * m - index - 1
        s[i] = a[k]
        del a[k]
    return h, s


def compose_clifford_circuit(clifd1: 'Clifford', clifd2: 'Clifford') -> 'paddle_quantum.ansatz.Circuit':
    r"""Compute the composition of two Clifford operators and obtain the corresponding circuit.

    Args:
        clifd1: The first Clifford operator to be composed.
        clifd2: The second Clifford operator to be composed.

    Returns:
        Circuit corresponding to the composed Clifford operator.
    """
    assert clifd1.num_qubits == clifd2.num_qubits, "the number of qubits of two clifford circuits should be the same"
    circuit = paddle_quantum.ansatz.Circuit(clifd1.num_qubits)
    for sub_layer in clifd1.circuit().sublayers():
        circuit.append(sub_layer)
    for sub_layer in clifd2.circuit().sublayers():
        circuit.append(sub_layer)
    return circuit


class Clifford:
    """Users can instantiate this class to randomly generate a Clifford operator.

    Args:
        num_qubits: Number of qubits on which this Clifford operator acts.

    References:
        1. Bravyi, Sergey, and Dmitri Maslov. "Hadamard-free circuits expose the structure of the Clifford group."
        IEEE Transactions on Information Theory 67.7 (2021): 4546-4563.
    """
    def __init__(self, num_qubits: int):
        # number of qubit
        self.num_qubits = num_qubits
        self.__table, self.__gamma, self.__delta, self.__h, self.__s = _random_clifford(num_qubits)
        self.phase = []
        for qubit in range(2 * self.num_qubits):
            self.phase.append(np.random.randint(0, 2))

        # Initialize stabilizer table
        self.x = np.transpose(self.__table)[0: num_qubits, :]
        # Initialize destabilizer table
        self.z = np.transpose(self.__table)[num_qubits: 2 * num_qubits, :]

    def print_clifford(self):
        r"""Print how the Clifford operator acts on the Pauli basis.
        """
        base = []
        base_out = []
        num_qubits = 2 * self.num_qubits

        # Initialize Pauli basis
        for position in range(0, self.num_qubits):
            base.append('X' + str(position + 1))
        for position in range(0, self.num_qubits):
            base.append('Z' + str(position + 1))

        # Compute stabilizer table
        for i in range(0, self.num_qubits):
            temp = ''
            for jx in range(0, num_qubits):
                if self.x[i][jx] == 1:
                    temp += base[jx]
            base_out.append(temp)

            # Compute destabilizer table
            temp = ''
            for jz in range(num_qubits):
                if self.z[i][jz] == 1:
                    temp += base[jz]
            base_out.append(temp)

        for i in range(num_qubits):
            if i % 2 == 0:
                # Fix the phase
                if self.phase[i // 2] == 1:
                    print(base[i // 2] + ' |-> ' + '+' + base_out[i])
                else:
                    print(base[i // 2] + ' |-> ' + '-' + base_out[i])
            else:
                if self.phase[self.num_qubits + (i - 1) // 2] == 1:
                    print(base[self.num_qubits + (i - 1) // 2] + ' |-> ' + '+' + base_out[i])
                else:
                    print(base[self.num_qubits + (i - 1) // 2] + ' |-> ' + '-' + base_out[i])

    def sym(self) -> np.ndarray:
        r"""Obtain the Clifford operator's symplectic matrix.

        Returns:
            Symplectic matrix corresponding to this Clifford operator.
        """
        sym = []
        for i in range(0, self.num_qubits):
            tempx = []
            temp = self.x[i][self.num_qubits: 2 * self.num_qubits]
            for jx in range(0, self.num_qubits):
                tempx.append(self.x[i][jx])
                tempx.append(temp[jx])
            sym.append(tempx)

            tempz = []
            temp = self.z[i][self.num_qubits: 2 * self.num_qubits]
            for jz in range(0, self.num_qubits):
                tempz.append(self.z[i][jz])
                tempz.append(temp[jz])
            sym.append(tempz)

        return np.array(sym).T

    def tableau(self) -> np.ndarray:
        r"""Obtain the Clifford operator's table.

        For the number of qubits being ``num_qubits``, the first ``num_qubits`` lines correspoding to results of :math:`X_i`,
        and the last ``num_qubits`` lines correspoding to results of :math:`Z_i`.

        Returns:
            Table corresponding to this Clifford operator.
        """
        return np.transpose(self.__table)

    def circuit(self) -> 'paddle_quantum.ansatz.Circuit':
        r"""Obtain the circuit corresponding to the Clifford operator.

        Returns:
            Circuit corresponding to this Clifford operator.
        """
        cir = paddle_quantum.ansatz.Circuit(self.num_qubits)

        gamma1 = self.__gamma[0]
        gamma2 = self.__gamma[1]
        delta1 = self.__delta[0]
        delta2 = self.__delta[1]

        # The second cnot layer
        for bitindex in range(self.num_qubits):
            for j in range(bitindex + 1, self.num_qubits):
                if delta2[j][bitindex] == 1:
                    cir.cnot([bitindex, j])

        # The second cz layer
        for bitindex in range(self.num_qubits):
            for j in range(bitindex + 1, self.num_qubits):
                if gamma2[bitindex][j] == 1:
                    cir.cz([bitindex, j])

        # The second P layer
        for bitindex in range(self.num_qubits):
            if gamma2[bitindex][bitindex] == 1:
                cir.s(bitindex)

        # Pauli layer
        for bitindex in range(self.num_qubits):
            if self.phase[bitindex] == 1 and self.phase[bitindex + self.num_qubits] == 0:
                cir.x(bitindex)
            elif self.phase[bitindex] == 0 and self.phase[bitindex + self.num_qubits] == 1:
                cir.z(bitindex)
            elif self.phase[bitindex] == 0 and self.phase[bitindex + self.num_qubits] == 0:
                cir.y(bitindex)

        # S layer
        swapped = []
        for bitindex in range(self.num_qubits):
            if self.__s[bitindex] == bitindex:
                continue
            swapped.append(self.__s[bitindex])
            if bitindex in swapped:
                continue
            cir.swap([bitindex, self.__s[bitindex]])

        # Hadamard layer
        for bitindex in range(self.num_qubits):
            if self.__h[bitindex] == 1:
                cir.h(bitindex)

        # cnot layer
        for bitindex in range(self.num_qubits):
            for j in range(bitindex + 1, self.num_qubits):
                if delta1[j][bitindex] == 1:
                    cir.cnot([bitindex, j])

        # cz layer
        for bitindex in range(self.num_qubits):
            for j in range(bitindex + 1, self.num_qubits):
                if gamma1[bitindex][j] == 1:
                    cir.cz([bitindex, j])

        # P layer
        for bitindex in range(self.num_qubits):
            if gamma1[bitindex][bitindex] == 1:
                cir.s(bitindex)

        return cir
