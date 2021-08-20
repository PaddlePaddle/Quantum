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

"""
Class for randomly generating a Clifford operator
"""

import numpy as np
from paddle_quantum.circuit import UAnsatz

__all__ = [
    "Clifford",
    "compose_clifford_circuit"
]


class Clifford:
    r"""用户可以通过实例化该 ``class`` 来随机生成一个 Clifford operator。

    Attributes:
        n (int): 该 Clifford operator 作用的量子比特数目

    References:
        1. Sergey Bravyi and Dmitri Maslov, Hadamard-free circuits expose the structure of the clifford group. arXiv preprint arXiv:2003.09412, 2020.
    """

    def __init__(self, n):
        r"""Clifford 的构造函数，用于实例化一个 Clifford 对象

        Args:
            n (int): 该 Clifford operator 作用的量子比特数目
        """
        # number of qubit
        self.n = n
        self.__table, self.__Gamma, self.__Delta, self.__h, self.__s = _random_clifford(n)
        self.phase = []
        for qbit in range(2 * self.n):
            self.phase.append(np.random.randint(0, 2))

        # Initialize stabilizer table
        self.x = np.transpose(self.__table)[0:n, :]
        # Initialize destabilizer table
        self.z = np.transpose(self.__table)[n:2 * n, :]

    def print_clifford(self):
        r"""输出该 Clifford 在 Pauli 基上的作用关系，来描述这个 Clifford
        """
        base = []
        base_out = []
        n = 2 * self.n

        # Initialize Pauli basis
        for position in range(self.n):
            base.append('X' + str(position + 1))
        for position in range(self.n):
            base.append('Z' + str(position + 1))

        # Compute stabilizer table
        for i in range(self.n):
            temp = ''
            for jx in range(n):
                if self.x[i][jx] == 1:
                    temp += base[jx]
            base_out.append(temp)

            # Compute destabilizer table
            temp = ''
            for jz in range(n):
                if self.z[i][jz] == 1:
                    temp += base[jz]
            base_out.append(temp)

        for i in range(n):
            if i % 2 == 0:
                # Fix the phase
                if self.phase[i // 2] == 1:
                    print(base[i // 2] + ' |-> ' + '+' + base_out[i])
                else:
                    print(base[i // 2] + ' |-> ' + '-' + base_out[i])
            else:
                if self.phase[self.n + (i - 1) // 2] == 1:
                    print(base[self.n + (i - 1) // 2] + ' |-> ' + '+' + base_out[i])
                else:
                    print(base[self.n + (i - 1) // 2] + ' |-> ' + '-' + base_out[i])

    def sym(self):
        r"""获取该 Clifford operator 所对应的辛矩阵

        Returns:
            numpy.ndarray: Clifford 对应的辛矩阵
        """
        sym = []
        for i in range(self.n):
            tempx = []
            temp = self.x[i][self.n:2 * self.n]
            for jx in range(0, self.n):
                tempx.append(self.x[i][jx])
                tempx.append(temp[jx])
            sym.append(tempx)

            tempz = []
            temp = self.z[i][self.n:2 * self.n]
            for jz in range(0, self.n):
                tempz.append(self.z[i][jz])
                tempz.append(temp[jz])
            sym.append(tempz)

        return np.array(sym).T

    def tableau(self):
        r"""获取该 Clifford operator 所对应的 table，对 n 个 qubits 情况，前 n 行对应 X_i 的结果，后 n 行对应 Z_i 的结果。

        Returns:
            numpy.ndarray: 该 Clifford 的 table
        """
        return np.transpose(self.__table)

    def circuit(self):
        r"""获取该 Clifford operator 所对应的电路

        Returns:
            UAnsatz: 该 Clifford 对应的电路
        """
        cir = UAnsatz(self.n)
        gamma1 = self.__Gamma[0]
        gamma2 = self.__Gamma[1]
        delta1 = self.__Delta[0]
        delta2 = self.__Delta[1]

        # The second cnot layer
        for bitindex in range(self.n):
            for j in range(bitindex + 1, self.n):
                if delta2[j][bitindex] == 1:
                    cir.cnot([bitindex, j])

        # The second cz layer
        for bitindex in range(self.n):
            for j in range(bitindex + 1, self.n):
                if gamma2[bitindex][j] == 1:
                    cir.cz([bitindex, j])

        # The second P layer
        for bitindex in range(self.n):
            if gamma2[bitindex][bitindex] == 1:
                cir.s(bitindex)

        # Pauli layer
        for bitindex in range(self.n):
            if self.phase[bitindex] == 1 and self.phase[bitindex + self.n] == 0:
                cir.x(bitindex)
            elif self.phase[bitindex] == 0 and self.phase[bitindex + self.n] == 1:
                cir.z(bitindex)
            elif self.phase[bitindex] == 0 and self.phase[bitindex + self.n] == 0:
                cir.y(bitindex)

        # S layer
        swapped = []
        for bitindex in range(self.n):
            if self.__s[bitindex] == bitindex:
                continue
            swapped.append(self.__s[bitindex])
            if bitindex in swapped:
                continue
            cir.swap([bitindex, self.__s[bitindex]])

        # Hadamard layer
        for bitindex in range(self.n):
            if self.__h[bitindex] == 1:
                cir.h(bitindex)

        # cnot layer
        for bitindex in range(self.n):
            for j in range(bitindex + 1, self.n):
                if delta1[j][bitindex] == 1:
                    cir.cnot([bitindex, j])

        # cz layer
        for bitindex in range(self.n):
            for j in range(bitindex + 1, self.n):
                if gamma1[bitindex][j] == 1:
                    cir.cz([bitindex, j])

        # P layer
        for bitindex in range(self.n):
            if gamma1[bitindex][bitindex] == 1:
                cir.s(bitindex)

        return cir


def compose_clifford_circuit(clif1, clif2):
    r"""计算两个指定的 Clifford 的复合，得到复合后的电路

    Args:
        clif1 (Clifford): 需要复合的第 1 个 Clifford
        clif2 (Clifford): 需要复合的第 2 个 Clifford

    Returns:
        UAnsatz: 复合后的 Clifford 所对应的电路，作用的顺序为 clif1、clif2
    """
    assert clif1.n == clif2.n, "the number of qubits of two cliffords should be the same"

    return clif1.circuit() + clif2.circuit()


def _sample_qmallows(n):
    r"""n 量子比特的 quantum mallows 采样，来获得随机采样 Clifford 时所需要的 S 和 h

    Args:
        n (int): 量子比特数目

    Returns:
        tuple: 包含

            numpy.ndarray: Clifford 采样时需要的参数 h
            numpy.ndarray: Clifford 采样时需要的参数 S

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    # Hadamard layer
    h = np.zeros(n, dtype=int)
    # S layer
    S = np.zeros(n, dtype=int)
    A = list(range(n))

    for i in range(n):
        m = n - i
        r = np.random.uniform(0, 1)
        index = int(2 * m - np.ceil(np.log(r * (4 ** m - 1) + 1) / np.log(2.0)))
        h[i] = 1 * (index < m)
        if index < m:
            k = index
        else:
            k = 2 * m - index - 1
        S[i] = A[k]
        del A[k]
    return h, S


def _random_clifford(n):
    r"""随机生成一个指定量子比特数目 n 的 Clifford 所对应的 table 及 canonical form 中的参数

    Args:
        n (int): 量子比特数目

    Returns:
        tuple: 包含

            numpy.ndarray: 随机生成的 Clifford 所对应的 table
            list: 随机生成的 Clifford 所对应的参数 Gamma
            list: 随机生成的 Clifford 所对应的参数 Delta
            numpy.ndarray: 随机生成的 Clifford 所对应的参数 h
            numpy.ndarray: 随机生成的 Clifford 所对应的参数 S

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    assert (n <= 100), "too many qubits"

    # Some constant matrices
    bigzero = np.zeros((2 * n, 2 * n), dtype=int)
    nzero = np.zeros((n, n), dtype=int)
    I = np.identity(n, dtype=int)

    h, S = _sample_qmallows(n)
    Gamma1 = np.copy(nzero)
    Delta1 = np.copy(I)
    Gamma2 = np.copy(nzero)
    Delta2 = np.copy(I)

    for i in range(n):
        Gamma2[i, i] = np.random.randint(2)
        if h[i] == 1:
            Gamma1[i, i] = np.random.randint(2)

    # Constraints for canonical form
    for j in range(n):
        for i in range(j + 1, n):
            b = np.random.randint(2)
            Gamma2[i, j] = b
            Gamma2[j, i] = b
            Delta2[i, j] = np.random.randint(2)
            if h[i] == 1 and h[j] == 1:
                b = np.random.randint(2)
                Gamma1[i, j] = b
                Gamma1[j, i] = b
            if h[i] == 1 and h[j] == 0 and S[i] < S[j]:
                b = np.random.randint(2)
                Gamma1[i, j] = b
                Gamma1[j, i] = b
            if h[i] == 0 and h[j] == 1 and S[i] > S[j]:
                b = np.random.randint(2)
                Gamma1[i, j] = b
                Gamma1[j, i] = b
            if h[i] == 0 and h[j] == 1:
                Delta1[i, j] = np.random.randint(2)

            if h[i] == 1 and h[j] == 1 and S[i] > S[j]:
                Delta1[i, j] = np.random.randint(2)

            if h[i] == 0 and h[j] == 0 and S[i] < S[j]:
                Delta1[i, j] = np.random.randint(2)

    # Compute stabilizer table
    st1 = np.matmul(Gamma1, Delta1)
    st2 = np.matmul(Gamma2, Delta2)
    inv1 = np.linalg.inv(np.transpose(Delta1))
    inv2 = np.linalg.inv(np.transpose(Delta2))
    f_1 = np.block([[Delta1, nzero], [st1, inv1]])
    f_2 = np.block([[Delta2, nzero], [st2, inv2]])
    f_1 = f_1.astype(int) % 2
    f_2 = f_2.astype(int) % 2
    U = np.copy(bigzero)

    for i in range(n):
        U[i, :] = f_2[S[i], :]
        U[i + n, :] = f_2[S[i] + n, :]

    # Apply Hadamard layer to the stabilizer table
    for i in range(n):
        if h[i] == 1:
            U[(i, i + n), :] = U[(i + n, i), :]

    Gamma = [Gamma1, Gamma2]
    Delta = [Delta1, Delta2]
    return np.matmul(f_1, U) % 2, Gamma, Delta, h, S
