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
The common function of the quantum state.
"""

import numpy as np
import paddle
import paddle_quantum
import QCompute
from ..backend import Backend
from ..backend import quleaf
from .state import State
from typing import Union, Optional, List


def to_state(
        data: Union[paddle.Tensor, np.ndarray, QCompute.QEnv], num_qubits: Optional[int] = None,
        backend: Optional[paddle_quantum.Backend] = None, dtype: Optional[str] = None
) -> State:
    r"""The function to generate a specified state instance.

    Args:
        data: The analytical form of quantum state.
        num_qubits: The number of qubits contained in the quantum state. Defaults to ``None``, which means it will be inferred by the data.
        backend: Used to specify the backend used. Defaults to ``None``, which means to use the default backend.
        dtype: Used to specify the data dtype of the data. Defaults to ``None``, which means to use the default data type.

    Returns:
        The generated quantum state.
    """
    if isinstance(data, np.ndarray):
        data = paddle.to_tensor(data)
    return State(data, num_qubits, backend, dtype)


def zero_state(
        num_qubits: int, backend: Optional[paddle_quantum.Backend] = None, dtype: Optional[str] = None
) -> State:
    r"""The function to generate a zero state.

    Args:
        num_qubits: The number of qubits contained in the quantum state.
        backend: Used to specify the backend used. Defaults to ``None``, which means to use the default backend.
        dtype: Used to specify the data dtype of the data. Defaults to ``None``, which means to use the default data type.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
       The generated quantum state.
    """
    dtype = paddle_quantum.get_dtype() if dtype is None else dtype
    np_dtype = np.complex64 if dtype == 'complex64' else np.complex128
    data = np.zeros((2 ** num_qubits,), dtype=np_dtype)
    data[0] = 1
    data = paddle.to_tensor(data)
    backend = paddle_quantum.get_backend() if backend is None else backend
    if backend == Backend.StateVector:
        state = State(data, num_qubits, backend=backend, dtype=dtype)
    elif backend == Backend.DensityMatrix:
        data = paddle.unsqueeze(data, axis=1)
        data = paddle.matmul(data, paddle.conj(paddle.t(data)))
        state = State(data, num_qubits, backend=backend, dtype=dtype)
    elif backend == Backend.QuLeaf:
        data = QCompute.QEnv()
        data.backend(quleaf.get_quleaf_backend())
        data.Q.createList(num_qubits)
        state = State(data, num_qubits, backend=backend, dtype=dtype)
    else:
        raise NotImplementedError
    return state


def computational_basis(
        num_qubits: int, index: int, backend: Optional[paddle_quantum.Backend] = None, dtype: Optional[str] = None
) -> State:
    r"""Generate a computational basis state :math:`|e_{i}\rangle` , whose i-th element is 1 and all the other elements are 0.

    Args:
        num_qubits: The number of qubits contained in the quantum state.
        index:  Index :math:`i` of the computational basis state :math`|e_{i}rangle` .
        backend: Used to specify the backend used. Defaults to ``None``, which means to use the default backend.
        dtype: Used to specify the data dtype of the data. Defaults to ``None``, which means to use the default data type.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    dtype = paddle_quantum.get_dtype() if dtype is None else dtype
    np_dtype = np.complex64 if dtype == 'complex64' else np.complex128
    data = np.zeros((2 ** num_qubits,), dtype=np_dtype)
    data[index] = 1
    data = paddle.to_tensor(data)
    backend = paddle_quantum.get_backend() if backend is None else backend
    if backend == Backend.StateVector:
        state = State(data, num_qubits, backend=backend)
    elif backend == Backend.DensityMatrix:
        data = paddle.unsqueeze(data, axis=1)
        data = paddle.matmul(data, paddle.conj(paddle.t(data)))
        state = State(data, num_qubits, backend=backend)
    else:
        raise NotImplementedError
    return state


def bell_state(num_qubits: int, backend: Optional[paddle_quantum.Backend] = None) -> State:
    r"""Generate a bell state.

    Its matrix form is:

    .. math::

        |\Phi_{D}\rangle=\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}

    Args:
        num_qubits: The number of qubits contained in the quantum state.
        backend: Used to specify the backend used. Defaults to ``None``, which means to use the default backend.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    dim = 2 ** num_qubits
    local_dim = 2 ** int(num_qubits / 2)
    coeff = 1 / local_dim
    data = np.zeros((dim, dim), dtype=np_dtype)
    for i in range(0, dim, local_dim + 1):
        for j in range(0, dim, local_dim + 1):
            data[i, j] = coeff
    backend = paddle_quantum.get_backend() if backend is None else backend
    if backend == Backend.StateVector:
        eigenvalue, eigenvector = paddle.linalg.eig(paddle.to_tensor(data))
        data = eigenvector[:, paddle.argmax(paddle.real(eigenvalue))]
        state = State(data, num_qubits, backend=backend)
    elif backend == Backend.DensityMatrix:
        state = State(paddle.to_tensor(data), num_qubits, backend=backend)
    else:
        raise NotImplementedError
    return state


def bell_diagonal_state(prob: List[float]) -> State:
    r"""Generate a bell diagonal state.

    Its matrix form is:

    .. math::

        p_{1}|\Phi^{+}\rangle\langle\Phi^{+}|+p_{2}| \Psi^{+}\rangle\langle\Psi^{+}|+p_{3}| \Phi^{-}\rangle\langle\Phi^{-}| +
        p_{4}|\Psi^{-}\rangle\langle\Psi^{-}|

    Args:
        prob: The prob of each bell state.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    p1, p2, p3, p4 = prob
    assert 0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= p3 <= 1 and 0 <= p4 <= 1, "Each probability must be in [0, 1]."
    assert abs(p1 + p2 + p3 + p4 - 1) < 1e-6, "The sum of probabilities should be 1."

    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    coeff = np.sqrt(0.5)
    phi_p_vec = np.array([[coeff, 0, 0, coeff]], dtype=np_dtype)
    phi_p_mat = np.matmul(phi_p_vec.T, phi_p_vec)
    phi_m_vec = np.array([[coeff, 0, 0, -coeff]], dtype=np_dtype)
    phi_m_mat = np.matmul(phi_m_vec.T, phi_m_vec)
    psi_p_vec = np.array([[0, coeff, coeff, 0]], dtype=np_dtype)
    psi_p_mat = np.matmul(psi_p_vec.T, psi_p_vec)
    psi_m_vec = np.array([[0, coeff, -coeff, 0]], dtype=np_dtype)
    psi_m_mat = np.matmul(psi_m_vec.T, psi_m_vec)

    state = p1 * phi_p_mat + p2 * psi_p_mat + p3 * phi_m_mat + p4 * psi_m_mat

    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        trace_rho = paddle.trace(paddle.to_tensor(state) @ paddle.to_tensor(state))
        if trace_rho.numpy()[0] == 1:
            eigenvalue, eigenvector = paddle.linalg.eig(paddle.to_tensor(state))
            data = eigenvector[:, paddle.argmax(paddle.real(eigenvalue))]
            state = State(data, backend=backend)
        else:
            raise Exception("The state is not a pure state")
    elif backend == Backend.DensityMatrix:
        state = State(paddle.to_tensor(state), backend=backend)
    else:
        raise NotImplementedError
    return state


def random_state(num_qubits: int, is_real: Optional[bool] = False, rank: Optional[int] = None) -> State:
    r"""Generate a random quantum state.

    Args:
        num_qubits: The number of qubits contained in the quantum state.
        is_real: If the quantum state only contains the real number. Defaults to ``False``.
        rank: The rank of the density matrix. Defaults to ``None`` which means full rank.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    float_dtype = 'float32' if paddle_quantum.get_dtype() == 'complex64' else 'float64'
    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        if is_real:
            data = paddle.rand((2 ** num_qubits,), dtype=float_dtype)
        else:
            data_real = paddle.rand((2 ** num_qubits,), dtype=float_dtype)
            data_imag = paddle.rand((2 ** num_qubits,), dtype=float_dtype)
            data = data_real + 1j * data_imag
        norm = np.linalg.norm(data.numpy())
        data = data / paddle.to_tensor(norm)
        state = State(data, num_qubits, backend=backend)
    elif backend == Backend.DensityMatrix:
        rank = 2 ** num_qubits if rank is None else rank
        if is_real:
            data = paddle.rand((2 ** num_qubits, rank), dtype=float_dtype)
        else:
            data_real = paddle.rand((2 ** num_qubits, rank), dtype=float_dtype)
            data_imag = paddle.rand((2 ** num_qubits, rank), dtype=float_dtype)
            data = data_real + 1j * data_imag
        data = paddle.matmul(data, paddle.conj(paddle.t(data)))
        data = data / paddle.trace(data)
        state = State(data, num_qubits, backend=backend)
    else:
        raise NotImplementedError
    return state


def w_state(num_qubits: int) -> 'State':
    r"""Generate a W-state.

    Args:
        num_qubits: The number of qubits contained in the quantum state.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    coeff = np.ones((1, 2 ** num_qubits)) / np.sqrt(num_qubits)
    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        state = np.zeros((1, 2 ** num_qubits), dtype=np_dtype)
        for i in range(num_qubits):
            state[0][2 ** i] = coeff[0][num_qubits - i - 1]
        state = State(paddle.to_tensor(state), num_qubits, backend=backend)
    elif backend == Backend.DensityMatrix:
        state = np.zeros(2 ** num_qubits, dtype=np_dtype)
        for i in range(num_qubits):
            state[2 ** i] = coeff[0][num_qubits - i - 1]
        state = paddle.to_tensor(state)
        state = paddle.unsqueeze(state, axis=1)
        state = paddle.matmul(state, paddle.conj(paddle.t(state)))
        state = State(state, num_qubits, backend=backend)
    else:
        raise NotImplementedError
    return state


def ghz_state(num_qubits: int) -> 'State':
    r"""Generate a GHZ-state.

    Args:
        num_qubits: The number of qubits contained in the quantum state.

    Raises:
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        state = np.zeros((1, 2 ** num_qubits))
        state[0][0] = 1 / np.sqrt(2)
        state[0][-1] = 1 / np.sqrt(2)
        state = State(paddle.to_tensor(state), num_qubits, backend=backend)
    elif backend == Backend.DensityMatrix:
        state = np.zeros(2 ** num_qubits, dtype=np_dtype)
        state[0] = 1 / np.sqrt(2)
        state[-1] = 1 / np.sqrt(2)
        state = paddle.to_tensor(state)
        state = paddle.unsqueeze(state, axis=1)
        state = paddle.matmul(state, paddle.conj(paddle.t(state)))
        state = State(state, num_qubits, backend=backend)
    else:
        raise NotImplementedError
    return state


def completely_mixed_computational(num_qubits: int) -> State:
    r"""Generate the density matrix of the completely mixed state.

    Args:
        num_qubits: The number of qubits contained in the quantum state.

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    assert num_qubits > 0, 'qubit number must be positive'

    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    state = np.eye(2 ** num_qubits) / (2 ** num_qubits)
    state = state.astype(np_dtype)
    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        trace_rho = paddle.trace(paddle.to_tensor(state) @ paddle.to_tensor(state))
        if trace_rho.numpy()[0] == 1:
            eigenvalue, eigenvector = paddle.linalg.eig(paddle.to_tensor(state))
            data = eigenvector[:, paddle.argmax(paddle.real(eigenvalue))]
            state = State(data, backend=backend)
        else:
            raise Exception("The state is not a pure state")
    elif backend == Backend.DensityMatrix:
        state = State(paddle.to_tensor(state), num_qubits, backend=backend)
    else:
        raise NotImplementedError
    return state


def r_state(prob: float) -> State:
    r"""Generate an R-state.

    Its matrix form is:

    .. math::

        p|\Psi^{+}\rangle\langle\Psi^{+}| + (1 - p)|11\rangle\langle11|

    Args:
        prob: The parameter of the R-state to be generated. It should be in :math:`[0,1]` .

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"

    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    coeff = np.sqrt(0.5)
    psi_p_vec = np.array([[0, coeff, coeff, 0]])
    psi_p_mat = np.matmul(psi_p_vec.T, psi_p_vec)
    state_11 = np.zeros((4, 4))
    state_11[3, 3] = 1
    state = prob * psi_p_mat + (1 - prob) * state_11
    state = state.astype(np_dtype)
    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        trace_rho = paddle.trace(paddle.to_tensor(state) @ paddle.to_tensor(state))
        if trace_rho.numpy()[0] == 1:
            eigenvalue, eigenvector = paddle.linalg.eig(paddle.to_tensor(state))
            data = eigenvector[:, paddle.argmax(paddle.real(eigenvalue))]
            state = State(data, backend=backend)
        else:
            raise Exception("The state is not a pure state")
    elif backend == Backend.DensityMatrix:
        state = State(paddle.to_tensor(state), backend=backend)
    else:
        raise NotImplementedError
    return state


def s_state(prob: float) -> State:
    r"""Generate the S-state.

    Its matrix form is:

    .. math::

        p|\Phi^{+}\rangle\langle\Phi^{+}| + (1 - p)|00\rangle\langle00|

    Args:
        prob: The parameter of the S-state to be generated. It should be in :math:`[0,1]` .

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"

    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    phi_p = bell_state(2).data.numpy()
    psi0 = np.zeros_like(phi_p)
    psi0[0, 0] = 1
    state = prob * phi_p + (1 - prob) * psi0
    state = state.astype(np_dtype)

    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        trace_rho = paddle.trace(paddle.to_tensor(state) @ paddle.to_tensor(state))
        if trace_rho.numpy()[0] == 1:
            eigenvalue, eigenvector = paddle.linalg.eig(paddle.to_tensor(state))
            data = eigenvector[:, paddle.argmax(paddle.real(eigenvalue))]
            state = State(data, backend=backend)
        else:
            raise Exception("The state is not a pure state")
    elif backend == Backend.DensityMatrix:
        state = State(paddle.to_tensor(state), backend=backend)
    else:
        raise NotImplementedError
    return state


def isotropic_state(num_qubits: int, prob: float) -> State:
    r"""Generate the isotropic state.

    Its matrix form is:

    .. math::

        p(\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}) + (1 - p)\frac{I}{2^n}

    Args:
        num_qubits: The number of qubits contained in the quantum state.
        prob: The parameter of the isotropic state to be generated. It should be in :math:`[0,1]` .

    Raises:
        Exception: The state should be a pure state if the backend is state_vector.
        NotImplementedError: If the backend is wrong or not implemented.

    Returns:
        The generated quantum state.
    """
    assert 0 <= prob <= 1, "Probability must be in [0, 1]"

    np_dtype = np.complex64 if paddle_quantum.get_dtype() == 'complex64' else np.complex128
    dim = 2 ** num_qubits
    phi_b = bell_state(num_qubits).data.numpy()
    state = prob * phi_b + (1 - prob) * np.eye(dim) / dim
    state = state.astype(np_dtype)

    backend = paddle_quantum.get_backend()
    if backend == Backend.StateVector:
        trace_rho = paddle.trace(paddle.to_tensor(state) @ paddle.to_tensor(state))
        if trace_rho.numpy()[0] == 1:
            eigenvalue, eigenvector = paddle.linalg.eig(paddle.to_tensor(state))
            data = eigenvector[:, paddle.argmax(paddle.real(eigenvalue))]
            state = State(data, backend=backend)
        else:
            raise Exception("The state is not a pure state")
    elif backend == Backend.DensityMatrix:
        state = State(paddle.to_tensor(state), backend=backend)
    else:
        raise NotImplementedError
    return state
