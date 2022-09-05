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
from paddle_quantum.ansatz.circuit import Circuit
from paddle_quantum import set_dtype
from paddle_quantum.linalg import is_hermitian, is_projector, is_unitary, dagger
from paddle_quantum.qinfo import partial_trace
import paddle
from math import log2, ceil
from scipy.linalg import expm
from numpy.polynomial.polynomial import Polynomial
from .qsp import reflection_based_quantum_signal_processing


"""
    Libraries for Quantum Singular Value Transformations
        referring to paper https://arxiv.org/abs/1806.01838

"""


def block_encoding_projector(
    num_qubits: int, num_projected_qubits: int = None
) -> paddle.Tensor:
    r"""Generate a projector that is used for simple block encoding

    Args:
        num_qubits: number of total qubits
        num_projected_qubits: number of projected qubits, default to be `num_qubits - 1`

    Returns:
        :math:`\ket{0}\bra{0} \otimes I`

    """
    if num_projected_qubits is None:
        num_projected_qubits = num_qubits - 1

    m, n = num_projected_qubits, num_qubits
    small_I = paddle.eye(2**m)
    ket_0 = paddle.zeros([2 ** (n - m), 2 ** (n - m)])
    ket_0[0, 0] += 1
    return paddle.kron(ket_0, small_I)


def qubitization(proj: paddle.Tensor, phi: paddle.Tensor) -> Circuit:
    r"""generate quantum circuit that is equivalent to :math:`e^{i \phi (2P - I)}`

    Args:
        proj: orthogonal projector
        phi: angle parameter

    Returns:
        a quantum circuit that is equivalent to e^{i \phi (2P - I)}.

    """
    assert is_hermitian(proj) and is_projector(proj)

    # preparation
    n = ceil(log2(proj.shape[0]))
    dtype = "complex64"
    X = paddle.to_tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)

    # define registers
    system_register = [i for i in range(n)]
    aux_register = [n]

    CPX_matrix = paddle.kron(proj, X) + paddle.kron(
        paddle.eye(2**n) - proj, paddle.eye(2)
    )

    cir = Circuit()
    cir.oracle(CPX_matrix, system_register + aux_register)

    cir.rz(aux_register, param=2 * phi)

    cir.oracle(CPX_matrix, system_register + aux_register)

    return cir


class QSVT(object):
    def __init__(
        self, poly_p: Polynomial, oracle: paddle.Tensor, m: int = None
    ) -> None:
        r"""Initialize a class that is used for QSP in multiple qubits

        Args:
            poly_p: polynomial :math:`P(x)`
            oracle: unitary :math:`U` which is a block encoding of a Hermitian :math:`X`
            m: log2(dimension of :math:`X`), default to be n - 1

        """

        shape = oracle.shape
        N = shape[0]
        n = int(log2(N))

        if m is None:
            m = n - 1

        assert is_unitary(oracle)  # assert U is a 2^n x 2^n unitary
        assert is_hermitian(
            oracle[0 : 2**m, 0 : 2**m]
        )  # assert the matrix block encoded by U is Hermitian

        self.I = paddle.eye(N)
        self.n = n
        self.U = oracle

        # find A
        assert m <= N  # make sure A is a block encoding of U

        # determine V

        self.V = block_encoding_projector(n, m)

        # determine phi
        self.Phi = paddle.to_tensor(
            reflection_based_quantum_signal_processing(poly_p), dtype="float32"
        )

    def block_encoding_matrix(self) -> paddle.Tensor:
        r"""provide the matrix of a block encoding of :math:`P(X)`

        Returns:
            block encoding of :math:`P(X)` in matrix form

        """
        k = len(self.Phi)

        matrix = self.I
        Vz = 2 * self.V - self.I

        for i in range(k):
            VRz = paddle.to_tensor(expm((Vz * 1j * self.Phi[i]).numpy()))

            if i % 2 != k % 2:
                matrix = matrix @ VRz @ self.U
            else:
                matrix = matrix @ VRz @ dagger(self.U)

        return matrix

    def block_encoding_circuit(self) -> Circuit:
        r"""generate a block encoding of :math:`P(X)` by quantum circuit

        Returns:
            a quantum circuit of unitary that is the block encoding of :math:`P(X)`

        """
        set_dtype("complex64")
        Phi = paddle.cast(self.Phi, dtype="float32")
        U = self.U
        system_register = [i for i in range(self.n)]

        cir = Circuit(self.n + 1)
        k = len(self.Phi)
        for i in range(k):
            if i % 2 == 0:
                cir.oracle(U, system_register)
            else:
                cir.oracle(dagger(U), system_register)
            cir.extend(qubitization(self.V, Phi[-i - 1]))

        return cir

    def block_encoding_unitary(self) -> paddle.Tensor:
        r"""generate the unitary of above circuit for verification

        Returns:
            a block encoding unitary of :math:`P(X)`

        """
        U = self.block_encoding_circuit().unitary_matrix()
        zero_state = paddle.zeros([2**1, 2**1])
        zero_state[0, 0] += 1
        U = U @ paddle.kron(paddle.eye(2**self.n), zero_state)
        return partial_trace(U, 2**self.n, 2**1, A_or_B=2)
