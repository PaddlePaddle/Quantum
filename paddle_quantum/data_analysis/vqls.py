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
The VQLS model.
"""

import logging
import os
from functools import partial
from tqdm import tqdm
from typing import Optional, List, Tuple, Callable, Union

# import cv2
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

import paddle_quantum as pq
from paddle_quantum.ansatz import Circuit
from paddle_quantum.gate import *
from paddle_quantum.state import *
from paddle_quantum.linalg import *
import warnings

warnings.filterwarnings("ignore", category=Warning)
pq.set_backend("state_vector")

def _nextPowerOf2(n: int):
    count = 0
    if n and not (n & (n - 1)):
        return n
    while n != 0:
        n >>= 1
        count += 1
    return 1 << count


def _preprocess(A: np.ndarray, b: np.ndarray):
    # extend input so dimensions are power of 2
    fill_dim = _nextPowerOf2(len(A))
    original_dim = len(A)
    A = np.pad(A, ((0, fill_dim-len(A)), (0, fill_dim-len(A))), 'constant', constant_values=0)

    # rescale b and convert to state
    b = np.pad(b, (0,fill_dim-len(b)), 'constant', constant_values=0)
    b_scale = np.linalg.norm(b)
    b = b / b_scale
    b = to_state(b)
    num_qubits = b.num_qubits

    # decompose A into 2 unitaries
    _, s, _ = np.linalg.svd(A)
    A_scale = max(np.abs(s))
    A_copy = A / (A_scale + 1)

    u, s, v = np.linalg.svd(A_copy)
    z = []
    z_conj = []
    for idx in range(len(s)):
        val = s[idx] + np.sqrt(1 - s[idx] ** 2) * 1j
        z.append(val)
        z_conj.append(np.conj(val))

    z = np.diag(z)
    z_conj = np.diag(z_conj)
    list_A = []
    coefficients_real = []
    coefficients_img = []
    list_A.append(paddle.to_tensor(np.matmul(np.matmul(u, z), v)))
    coefficients_real.append((A_scale+1)/2)
    coefficients_img.append(0)

    list_A.append(paddle.to_tensor(np.matmul(np.matmul(u, z_conj), v)))
    coefficients_real.append((A_scale+1)/2)
    coefficients_img.append(0)
    return b, num_qubits, list_A, coefficients_real, coefficients_img, original_dim, b_scale


def hadamard_test(phi: pq.state.State, U: paddle.Tensor, num_qubits: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""
    Given unitary U and state :math:`|\phi\rangle`, it computes the expectation of U with respect to :math:`|\phi\rangle`, i.e. :math:`|\phi\rangle`.

    Args:
        phi: State which the expectation is with respect to.
        U: Unitary which we are taking expectation of.
        num_qubits: Number of qubits of phi/U.

    Returns:
        Return the real and imaginary part of the expectation value.

    """
    # Return real part
    cir = Circuit(num_qubits + 1)
    cir.h([0])
    cir.control_oracle(U, range(num_qubits + 1))
    cir.h([0])

    input_state = to_state(paddle.kron(zero_state(1).data, phi.data))
    result_state = cir(input_state)
    measure = pq.loss.measure.Measure()
    prob_0 = measure(result_state, qubits_idx=0, desired_result='0')
    prob_1 = measure(result_state, qubits_idx=0, desired_result='1')
    real_exp = prob_0 - prob_1

    # Return imaginary part
    cir = Circuit(num_qubits + 1)
    cir.h([0])
    cir.sdg([0])
    cir.control_oracle(U, range(num_qubits + 1))
    cir.h([0])

    input_state = to_state(np.kron(zero_state(1).numpy(), phi.numpy()))
    result_state = cir(input_state)
    prob_0 = measure(result_state, qubits_idx=0, desired_result='0')
    prob_1 = measure(result_state, qubits_idx=0, desired_result='1')
    img_exp = prob_0 - prob_1

    return real_exp, img_exp


def hadamard_overlap_test(phi: State, b: State, An: paddle.Tensor, Am: paddle.Tensor, num_qubits: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""
    Given unitary Am, An and state :math:`|\phi\rangle`, b, it computes the value of :math:`\langle{b}| An |\phi\rangle\langle\phi| Am^\dagger |b\rangle`.

    Args:
        phi: State in the calculation.
        b: State in the calculation.
        Am: Unitary matrix in the calculation.
        An: Unitary matrix in the calculation.
        num_qubits: Number of qubits of the system.
    Returns:
        Return the real and imaginary part of the calculation.

    """
    # Return the real part
    cir = Circuit(2*num_qubits + 1)
    cir.h([0])
    cir.control_oracle(An, range(num_qubits + 1))
    cir.control_oracle(Am.conj().T, [0]+list(range(num_qubits+1, 2*num_qubits+1)))
    for idx in range(num_qubits):
        cir.cnot([idx+1, idx+1+num_qubits])
    cir.h(range(num_qubits+1))

    input_state = to_state(paddle.kron(paddle.kron(zero_state(1).data, phi.data), b.data))
    result_state = cir(input_state)
    measure = pq.loss.measure.Measure()

    # Calculate the result of the Overlap Circuit, obtain P_0 and P_1 as described in paper
    # See section IV.B. of https://arxiv.org/pdf/1303.6814.pdf
    bin_string = []
    for i in range(2 ** num_qubits, 2 ** (num_qubits + 1)):
        bin_string.append(bin(i)[3:])

    P_0 = paddle.zeros([1])
    P_1 = paddle.zeros([1])
    for i in bin_string:
        for j in bin_string:
            a = bin(int(i, base=2) & int(j, base=2))[2:].zfill(num_qubits)
            parity = a.count('1') % 2
            if parity == 0:
                P_0 = P_0 + measure(result_state, desired_result='0'+i+j)
                P_1 = P_1 + measure(result_state, desired_result='1'+i+j)
            else:
                P_0 = P_0 - measure(result_state, desired_result='0'+i+j)
                P_1 = P_1 - measure(result_state, desired_result='1'+i+j)
    real_exp = P_0 - P_1

    # Return the imaginary part
    cir = Circuit(2*num_qubits + 1)
    cir.h([0])
    cir.control_oracle(An, range(num_qubits + 1))
    cir.control_oracle(Am.conj().T, [0]+list(range(num_qubits+1, 2*num_qubits+1)))
    for idx in range(num_qubits):
        cir.cnot([idx+1,idx+1+num_qubits])
    cir.rz(qubits_idx=0, param=-np.pi/2)
    cir.h(range(num_qubits+1))

    result_state = cir(input_state)

    P_0 = paddle.zeros([1])
    P_1 = paddle.zeros([1])
    for i in bin_string:
        for j in bin_string:
            a = bin(int(i, base=2) & int(j, base=2))[2:].zfill(num_qubits)
            parity = a.count('1') % 2
            if parity == 0:
                P_0 = P_0 + measure(result_state, desired_result='0'+i+j)
                P_1 = P_1 + measure(result_state, desired_result='1'+i+j)
            else:
                P_0 = P_0 - measure(result_state, desired_result='0'+i+j)
                P_1 = P_1 - measure(result_state, desired_result='1'+i+j)
    img_exp = P_0 - P_1

    return real_exp, img_exp


def _complex_multiplication (x_real: float, x_img: float, y_real: float, y_img: float):
    real = x_real*y_real - x_img*y_img
    img = x_real*y_img + x_img*y_real
    return real, img


class VQLS(paddle.nn.Layer):
    r"""
    The class of the variational quantum linear solver (VQLS).

    Args:
        num_qubits: The number of qubits which the quantum circuit contains.
        A: List of unitaries in the decomposition of the input matrix.
        coefficients_real: Real part of coefficients of corresponding unitaries in the decomposition of the input matrix.
        coefficients_img: Imaginary part of coefficients of corresponding unitaries in the decomposition of the input matrix.
        b: The state which the input answer is encoded into.
        depth: Depth of the ansatz circuit.

    """
    def __init__(self, num_qubits: int, A: List[paddle.Tensor], coefficients_real: List[float],
                 coefficients_img: List[float],  b: State, depth: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.A = A
        self.b = b
        self.coefficients_real = coefficients_real
        self.coefficients_img = coefficients_img
        cir = Circuit(num_qubits)
        cir.complex_entangled_layer(depth=depth)
        self.cir = cir

    def forward(self) -> paddle.Tensor:
        r"""
        The forward function.

        Returns:
            Return the output of the model.
        """
        phi = self.cir(zero_state(self.num_qubits))
        numerator_real = paddle.zeros([1])
        denominator_real = paddle.zeros([1])
        for m in range(len(self.A)):
            for n in range(len(self.A)):
                prod_coeff_real, prod_coeff_img = _complex_multiplication(self.coefficients_real[m],-self.coefficients_img[m],self.coefficients_real[n],self.coefficients_img[n])
                temp_real, temp_img = hadamard_test(phi=phi, U=self.A[m].conj().T @ self.A[n], num_qubits=self.num_qubits)
                temp_real, temp_img = _complex_multiplication(prod_coeff_real, prod_coeff_img, temp_real, temp_img)
                denominator_real = denominator_real + temp_real

                temp_real, temp_img = hadamard_overlap_test(phi=phi, b=self.b, An=self.A[n], Am=self.A[m], num_qubits=self.num_qubits)
                temp_real, temp_img = _complex_multiplication(prod_coeff_real, prod_coeff_img, temp_real, temp_img)
                numerator_real = numerator_real + temp_real
        loss = 1 - numerator_real/denominator_real
        return loss


def _postprocess(scale: float, original_dim: int, A: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    # scale x to have the correct norm
    x = x[:original_dim]
    estimate = np.matmul(A,x)
    estimate_norm = np.linalg.norm(estimate)
    x = x * scale / estimate_norm

    # rotate x to have the correct phase
    phase = 0
    for i in range(len(b)):
        phase = phase + b[i] / (len(b) * np.matmul(A, x)[i])
    x = x * phase
    return x


def compute(A: np.ndarray, b: np.ndarray, depth: int, iterations: int, LR: float, gamma: Optional[float]=0) -> np.ndarray:
    r"""
    Solve the linear equation Ax=b.

    Args:
        A: Input matrix.
        b: Input vector.
        depth: Depth of ansatz circuit.
        iterations: Number of iterations for optimization.
        LR: Learning rate of optimizer.
        gamma: Extra option to end optimization early if loss is below this value. Default to '0'.

    Returns:
        Return the vector x that solves Ax=b.

    Raises:
        ValueError: A is not a square matrix.
        ValueError: dimension of A and b don't match.
        ValueError:  A is a singular matrix hence there's no unique solution.
            """
    # check dimension of input and invertibility of A
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not a square matrix")
    if len(A) != len(b):
        raise ValueError("dimension of A and b don't match")
    if np.linalg.det(A) == 0:
        raise ValueError("A cannot be inverted")

    logging.basicConfig(
        filename='./linear_solver.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO
    )

    msg = f"Input parameters:"
    logging.info(msg)
    msg = f"Depth of ansatz circuit: {depth}"
    logging.info(msg)
    msg = f"Learning rate: {LR}"
    logging.info(msg)
    msg = f"Number of iterations: {iterations}"
    logging.info(msg)
    if gamma == 0:
        msg = f"No threshold value given."
    else:
        msg = f"Threshold value: {gamma}"
    logging.info(msg)
    msg = f"Matrix A:\n{A};"
    logging.info(msg)
    msg = f"Vector b:\n{b}"
    logging.info(msg)

    b_rescale, num_qubits, list_A, coefficients_real, coefficients_img, original_dim, scale = _preprocess(A=A,b=b)
    vqls = VQLS(num_qubits=num_qubits, A=list_A, coefficients_real=coefficients_real, coefficients_img=coefficients_img, b=b_rescale, depth=depth)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=vqls.parameters())

    for itr in tqdm(range(1, iterations + 1)):
        loss = vqls()
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        if itr % 10 == 0:
            msg = (
                f"Iter:{itr:5d}, Loss:{loss.item(): 3.5f}"
            )
            logging.info(msg)
        if loss.item()<gamma:
            msg='Threshold value gamma reached, ending optimization'
            logging.info(msg)
            print(msg)
            break

    msg = f"Final parameters of quantum circuit:{vqls.parameters()}"
    logging.info(msg)
    estimate_state = vqls.cir(zero_state(num_qubits))
    x = _postprocess(scale=scale, original_dim=original_dim, A=A, x=estimate_state.numpy(), b=b)
    msg =f"x that solves Ax=b:\n{x}"
    logging.info(msg)
    return x

if __name__ == '__main__':
    exit(0)
