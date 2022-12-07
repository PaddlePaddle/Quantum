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
The library of functions in quantum information.
"""

import math
import warnings
import re
import numpy as np
from scipy.linalg import logm, sqrtm
import cvxpy
import matplotlib.image

import paddle
import paddle_quantum as pq
from .state import State, _type_fetch, _type_transform
from .base import get_dtype
from .intrinsic import _get_float_dtype
from .linalg import dagger, NKron, unitary_random
from .channel.custom import ChoiRepr, KrausRepr, StinespringRepr
from typing import Optional, Tuple, List, Union


def partial_trace(state: Union[np.ndarray, paddle.Tensor, State], 
                  dim1: int, dim2: int, A_or_B: int) -> Union[np.ndarray, paddle.Tensor, State]:
    r"""Calculate the partial trace of the quantum state.

    Args:
        state: Input quantum state.
        dim1: The dimension of system A.
        dim2: The dimension of system B.
        A_or_B: 1 or 2. 1 means to calculate partial trace on system A; 2 means to calculate partial trace on system B.

    Returns:
        Partial trace of the input quantum state.
    """
    type_str = _type_fetch(state)
    rho_AB = _type_transform(state, "density_matrix").data if type_str == "state_vector" \
                                                        else _type_transform(state, "tensor")

    higher_dims = rho_AB.shape[:-2]
    new_state = paddle.trace(
        paddle.reshape(
            rho_AB, 
            higher_dims.copy() + [dim1, dim2, dim1, dim2]
        ), 
        axis1 = -1 + A_or_B + len(higher_dims), 
        axis2 = 1 + A_or_B + len(higher_dims)
    )
    
    return _type_transform(new_state, type_str)


def partial_trace_discontiguous(state: Union[np.ndarray, paddle.Tensor, State], 
                                preserve_qubits: list=None) -> Union[np.ndarray, paddle.Tensor, State]:
    r"""Calculate the partial trace of the quantum state with arbitrarily selected subsystem

    Args:
        state: Input quantum state.
        preserve_qubits: Remaining qubits, default is None, indicate all qubits remain.

    Returns:
        Partial trace of the quantum state with arbitrarily selected subsystem.
    """
    type_str = _type_fetch(state)
    rho = _type_transform(state, "density_matrix").data if type_str == "state_vector" \
                                                        else _type_transform(state, "tensor")

    if preserve_qubits is None:
        return state

    n = int(math.log2(rho.shape[-1]))

    def new_partial_trace_singleOne(rho: paddle.Tensor, at: int) -> paddle.Tensor:
        n_qubits = int(math.log2(rho.shape[-1]))
        higher_dims = rho.shape[:-2]
        rho = paddle.trace(
            paddle.reshape(
                rho,
                higher_dims.copy() + [2 ** at, 2, 2 ** (n_qubits - at - 1), 2 ** at, 2, 2 ** (n_qubits - at - 1)]
            ),
            axis1=1+len(higher_dims),
            axis2=4+len(higher_dims)
        )
        return paddle.reshape(rho, higher_dims.copy() + [2 ** (n_qubits - 1), 2 ** (n_qubits - 1)])
    
    for i, at in enumerate(x for x in range(n) if x not in preserve_qubits):
        rho = new_partial_trace_singleOne(rho, at - i)
    
    return _type_transform(rho, type_str)


def trace_distance(rho: Union[np.ndarray, paddle.Tensor, State], 
                   sigma: Union[np.ndarray, paddle.Tensor, State]) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the trace distance of two quantum states.

    .. math::
        D(\rho, \sigma) = 1 / 2 * \text{tr}|\rho-\sigma|

    Args:
        rho: a quantum state.
        sigma: a quantum state.

    Returns:
        The trace distance between the input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "density_matrix").data
    sigma = _type_transform(sigma, "density_matrix").data

    assert rho.shape == sigma.shape, 'The shape of two quantum states are different'
    eigval, _ = paddle.linalg.eig(rho - sigma)
    dist = 0.5 * paddle.sum(paddle.abs(eigval))

    return dist.item() if type_rho == "numpy" and type_sigma == "numpy" else dist


def state_fidelity(rho: Union[np.ndarray, paddle.Tensor, State], 
                   sigma: Union[np.ndarray, paddle.Tensor, State]) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the fidelity of two quantum states.

    .. math::
        F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

    Args:
        rho: a quantum state.
        sigma: a quantum state.

    Returns:
        The fidelity between the input quantum states.
    """
    type_rho, type_sigma = _type_fetch(rho), _type_fetch(sigma)
    rho = _type_transform(rho, "density_matrix").numpy()
    sigma = _type_transform(sigma, "density_matrix").numpy()
    
    assert rho.shape == sigma.shape, 'The shape of two quantum states are different'
    fidelity = np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real

    if type_rho == "numpy" and type_sigma == "numpy":
        return fidelity
    return paddle.to_tensor(fidelity)


def gate_fidelity(U: Union[np.ndarray, paddle.Tensor], 
                  V: Union[np.ndarray, paddle.Tensor]) -> Union[np.ndarray, paddle.Tensor]:
    r"""calculate the fidelity between gates

    .. math::

        F(U, V) = |\text{tr}(UV^\dagger)|/2^n

    :math:`U` is a :math:`2^n\times 2^n` unitary gate

    Args:
        U: quantum gate :math:`U`  in matrix form
        V: quantum gate :math:`V`  in matrix form

    Returns:
        fidelity between gates
    
    """
    type_u, type_v = _type_fetch(U), _type_fetch(V)
    U, V = _type_transform(U, "tensor"), _type_transform(V, "tensor")

    V = paddle.cast(V, dtype=U.dtype)
    assert U.shape == V.shape, 'The shape of two matrices are different'
    fidelity = paddle.abs(paddle.trace(U @ dagger(V))) / U.shape[0]

    return fidelity.item() if type_u == "numpy" or type_v == "numpy" else fidelity


def purity(rho: Union[np.ndarray, paddle.Tensor, State]) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the purity of a quantum state.

    .. math::

        P = \text{tr}(\rho^2)

    Args:
        rho: Density matrix form of the quantum state.

    Returns:
        The purity of the input quantum state.
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "density_matrix").data
    gamma = paddle.trace(rho @ rho).real()

    return gamma.item() if type_rho == "numpy" else gamma


def von_neumann_entropy(rho: Union[np.ndarray, paddle.Tensor, State], base: Optional[int] = 2) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the von Neumann entropy of a quantum state.

    .. math::

        S = -\text{tr}(\rho \log(\rho))

    Args:
        rho: Density matrix form of the quantum state.
        base: The base of logarithm. Defaults to 2.

    Returns:
        The von Neumann entropy of the input quantum state.
    """
    type_rho = _type_fetch(rho)
    rho = _type_transform(rho, "density_matrix").data
    rho_eigenvalues = paddle.real(paddle.linalg.eigvals(rho))
    entropy = -1 * (math.log(math.e, base)) * sum([eigenvalue * paddle.log(eigenvalue) for eigenvalue in rho_eigenvalues if eigenvalue >= 1e-8])

    return entropy.item() if type_rho == "numpy" else entropy


def relative_entropy(rho: Union[np.ndarray, paddle.Tensor, State], 
                     sig: Union[np.ndarray, paddle.Tensor, State], base: Optional[int] = 2) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the relative entropy of two quantum states.

    .. math::

        S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)

    Args:
        rho: Density matrix form of the quantum state.
        sig: Density matrix form of the quantum state.
        base: The base of logarithm. Defaults to 2.

    Returns:
        Relative entropy between input quantum states.
    """
    type_rho, type_sig = _type_fetch(rho), _type_fetch(sig)
    rho = _type_transform(rho, "density_matrix").numpy()
    sig = _type_transform(sig, "density_matrix").numpy()
    assert rho.shape == sig.shape, 'The shape of two quantum states are different'
    entropy = (math.log(math.e, base)) * np.trace(rho @ logm(rho) - rho @ logm(sig)).real
    
    if type_rho == "numpy" or type_sig == "numpy":
        return entropy
    return paddle.to_tensor(entropy)


def random_pauli_str_generator(n: int, terms: Optional[int] = 3) -> List:
    r"""Generate a random observable in list form.

    An observable :math:`O=0.3X\otimes I\otimes I+0.5Y\otimes I\otimes Z`'s list form is
    ``[[0.3, 'x0'], [0.5, 'y0,z2']]``.  Such an observable is generated by 
    ``random_pauli_str_generator(3, terms=2)`` 

    Args:
        n: Number of qubits.
        terms: Number of terms in the observable. Defaults to 3.

    Returns:
        The randomly generated observable's list form.
    """
    pauli_str = []
    for sublen in np.random.randint(1, high=n + 1, size=terms):
        # Tips: -1 <= coeff < 1
        coeff = np.random.rand() * 2 - 1
        ops = np.random.choice(['x', 'y', 'z'], size=sublen)
        pos = np.random.choice(range(n), size=sublen, replace=False)
        op_list = [ops[i] + str(pos[i]) for i in range(sublen)]
        pauli_str.append([coeff, ','.join(op_list)])
    return pauli_str


def pauli_str_to_matrix(pauli_str: list, n: int) -> paddle.Tensor:
    r"""Convert the input list form of an observable to its matrix form.

    For example, if the input ``pauli_str`` is ``[[0.7, 'z0,x1'], [0.2, 'z1']]`` and ``n=3``,
    then this function returns the observable :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I`
    in matrix form.

    Args:
        pauli_str: A list form of an observable.
        n: Number of qubits.

    Raises:
        ValueError: Only Pauli operator "I" can be accepted without specifying its position.

    Returns:
        The matrix form of the input observable.
    """
    pauli_dict = {'i': np.eye(2) + 0j, 'x': np.array([[0, 1], [1, 0]]) + 0j,
                  'y': np.array([[0, -1j], [1j, 0]]), 'z': np.array([[1, 0], [0, -1]]) + 0j}

    # Parse pauli_str; 'x0,z1,y4' to 'xziiy'
    new_pauli_str = []
    for coeff, op_str in pauli_str:
        init = list('i' * n)
        op_list = re.split(r',\s*', op_str.lower())
        for op in op_list:
            if len(op) > 1:
                pos = int(op[1:])
                assert pos < n, 'n is too small'
                init[pos] = op[0]
            elif op.lower() != 'i':
                raise ValueError('Only Pauli operator "I" can be accepted without specifying its position')
        new_pauli_str.append([coeff, ''.join(init)])

    # Convert new_pauli_str to matrix; 'xziiy' to NKron(x, z, i, i, y)
    matrices = []
    for coeff, op_str in new_pauli_str:
        sub_matrices = [pauli_dict[op.lower()] for op in op_str]
        if len(op_str) == 1:
            matrices.append(coeff * sub_matrices[0])
        else:
            mat = sub_matrices[0]
            for idx in range(1, len(sub_matrices)):
                mat = np.kron(mat, sub_matrices[idx])
            matrices.append(coeff * mat)

    return paddle.to_tensor(sum(matrices), dtype=get_dtype())


def partial_transpose_2(density_op: Union[np.ndarray, paddle.Tensor, State], 
                        sub_system: int = None) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.
        sub_system: 1 or 2. 1 means to perform partial transpose on system A; 
                    2 means to perform partial transpose on system B. Default is 2.

    Returns:
        The partial transpose of the input quantum state.
    
    Example:

    .. code-block:: python

        import paddle
        from paddle_quantum.qinfo import partial_transpose_2

        rho_test = paddle.arange(1,17).reshape([4,4])
        partial_transpose_2(rho_test, sub_system=1)

    ::

       [[ 1,  2,  9, 10],
        [ 5,  6, 13, 14],
        [ 3,  4, 11, 12],
        [ 7,  8, 15, 16]]
    """
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "density_matrix").numpy()
    
    sys_idx = 2 if sub_system is None else 1

    # Copy the density matrix and not corrupt the original one
    transposed_density_op = np.copy(density_op)
    if sys_idx == 2:
        for j in [0, 2]:
            for i in [0, 2]:
                transposed_density_op[i:i + 2, j:j + 2] = density_op[i:i + 2, j:j + 2].transpose()
    else:
        transposed_density_op[2:4, 0:2] = density_op[0:2, 2:4]
        transposed_density_op[0:2, 2:4] = density_op[2:4, 0:2]

    if type_str == "numpy":
        return transposed_density_op
    return paddle.to_tensor(transposed_density_op)


def partial_transpose(density_op: Union[np.ndarray, paddle.Tensor, State], 
                      n: int) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.
        n: Number of qubits of subsystem A, with qubit indices as [0, 1, ..., n-1]

    Returns:
        The partial transpose of the input quantum state.
    """
    # Copy the density matrix and not corrupt the original one
    type_str = _type_fetch(density_op)
    density_op = _type_transform(density_op, "density_matrix")
    n_qubits = density_op.num_qubits
    density_op = density_op.data

    density_op = paddle.reshape(density_op, [2 ** n, 2 ** (n_qubits - n), 2 ** n, 2 ** (n_qubits - n)])
    density_op = paddle.transpose(density_op, [2, 1, 0, 3])
    density_op = paddle.reshape(density_op, [2 ** n_qubits, 2 ** n_qubits])

    return density_op.numpy() if type_str == "numpy" else density_op


def negativity(density_op: Union[np.ndarray, paddle.Tensor, State]) -> Union[np.ndarray, paddle.Tensor]:
    r"""Compute the Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Negativity of the input quantum state.
    """
    # Implement the partial transpose
    density_op_T = partial_transpose_2(density_op)
    type_str = _type_fetch(density_op_T)
    density_op = _type_transform(density_op_T, "density_matrix").numpy()

    # Calculate through the equivalent expression N = sum(abs(\lambda_i)) when \lambda_i<0
    n = 0.0
    eigen_val, _ = np.linalg.eig(density_op_T)
    for val in eigen_val:
        if val < 0:
            n = n + np.abs(val)

    return n if type_str == "numpy" else paddle.to_tensor(n)


def logarithmic_negativity(density_op: Union[np.ndarray, paddle.Tensor, State]) -> Union[np.ndarray, paddle.Tensor]:
    r"""Calculate the Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Logarithmic Negativity of the input quantum state.
    """
    # Calculate the negativity
    n = negativity(density_op)

    return paddle.log2(2 * n + 1)


def is_ppt(density_op: Union[np.ndarray, paddle.Tensor, State]) -> bool:
    r"""Check if the input quantum state is PPT.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        Whether the input quantum state is PPT.
    """
    return bool(negativity(density_op) <= 0)


def schmidt_decompose(psi: Union[np.ndarray, paddle.Tensor, State], 
                      sys_A: List[int]=None) -> Union[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
                                                      Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""Calculate the Schmidt decomposition of a quantum state :math:`\lvert\psi\rangle=\sum_ic_i\lvert i_A\rangle\otimes\lvert i_B \rangle`.

    Args:
        psi: State vector form of the quantum state, with shape (2**n)
        sys_A: Qubit indices to be included in subsystem A (other qubits are included in subsystem B), default are the first half qubits of :math:`\lvert \psi\rangle`

    Returns:
        contains elements

        * A one dimensional array composed of Schmidt coefficients, with shape ``(k)``
        * A high dimensional array composed of bases for subsystem A :math:`\lvert i_A\rangle`, with shape ``(k, 2**m, 1)``
        * A high dimensional array composed of bases for subsystem B :math:`\lvert i_B\rangle` , with shape ``(k, 2**m, 1)``
    """
    type_psi = _type_fetch(psi)
    psi = _type_transform(psi, "state_vector").numpy()
    assert psi.ndim == 1, 'Psi must be a one dimensional vector.'
    assert np.log2(psi.size).is_integer(), 'The number of amplitutes must be an integral power of 2.'


    tot_qu = int(np.log2(psi.size))
    sys_A = sys_A if sys_A is not None else list(range(tot_qu//2))
    sys_B = [i for i in range(tot_qu) if i not in sys_A]

    # Permute qubit indices
    psi = psi.reshape([2] * tot_qu).transpose(sys_A + sys_B)

    # construct amplitude matrix
    amp_mtr = psi.reshape([2**len(sys_A), 2**len(sys_B)])

    # Standard process to obtain schmidt decomposition
    u, c, v = np.linalg.svd(amp_mtr)

    k = np.count_nonzero(c > 1e-13)
    c, u, v = c[:k], u.T[:k].reshape([k, -1, 1]), v[:k].reshape([k, -1, 1])

    if type_psi == "numpy":
        return c, u, v
    return paddle.to_tensor(c), paddle.to_tensor(u), paddle.to_tensor(v)


def image_to_density_matrix(image_filepath: str) -> State:
    r"""Encode image to density matrix

    Args:
        image_filepath: Path to the image file.

    Returns:
        The density matrix obtained by encoding
    """
    image_matrix = matplotlib.image.imread(image_filepath)

    # Converting images to grayscale
    image_matrix = image_matrix.mean(axis=2)

    # Fill the matrix so that it becomes a matrix whose shape is [2**n,2**n]
    length = int(2**np.ceil(np.log2(np.max(image_matrix.shape))))
    image_matrix = np.pad(image_matrix, ((0, length-image_matrix.shape[0]), (0, length-image_matrix.shape[1])), 'constant')
    # Density matrix whose trace  is 1
    rho = image_matrix@image_matrix.T
    rho = rho/np.trace(rho)
    return State(rho, backend=pq.Backend.DensityMatrix)


def shadow_trace(state: 'State', hamiltonian: pq.Hamiltonian, 
                 sample_shots: int, method: Optional[str] = 'CS') -> float:
    r"""Estimate the expectation value :math:`\text{trace}(H\rho)`  of an observable :math:`H`.

    Args:
        state: Quantum state.
        hamiltonian: Observable.
        sample_shots: Number of samples.
        method: Method used to, which should be one of “CS”, “LBCS”, and “APS”. Default is “CS”.

    Raises:
        ValueError: Hamiltonian has a bad form
    
    Returns:
        The estimated expectation value for the hamiltonian.
    """
    if not isinstance(hamiltonian, list):
        hamiltonian = hamiltonian.pauli_str
    num_qubits = state.num_qubits
    mode = state.backend
    if method == "LBCS":
        result, beta = pq.shadow.shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian, method)
    else:
        result = pq.shadow.shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian, method)

    def prepare_hamiltonian(hamiltonian, num_qubits):
        new_hamiltonian = []
        for idx, (coeff, pauli_str) in enumerate(hamiltonian):
            pauli_str = re.split(r',\s*', pauli_str.lower())
            pauli_term = ['i'] * num_qubits
            for item in pauli_str:
                if len(item) > 1:
                    pauli_term[int(item[1:])] = item[0]
                elif item[0].lower() != 'i':
                    raise ValueError('Expecting I for ', item[0])
            new_term = [coeff, ''.join(pauli_term)]
            new_hamiltonian.append(new_term)
        return new_hamiltonian

    hamiltonian = prepare_hamiltonian(hamiltonian, num_qubits)

    sample_pauli_str = [item for item, _ in result]
    sample_measurement_result = [item for _, item in result]
    coeff_terms = []
    pauli_terms = []
    for coeff, pauli_term in hamiltonian:
        coeff_terms.append(coeff)
        pauli_terms.append(pauli_term)

    pauli2idx = {'x': 0, 'y': 1, 'z': 2}

    def estimated_weight_cs(sample_pauli_str, pauli_term):
        result = 1
        for i in range(num_qubits):
            if sample_pauli_str[i] == 'i' or pauli_term[i] == 'i':
                continue
            elif sample_pauli_str[i] == pauli_term[i]:
                result *= 3
            else:
                result = 0
        return result

    def estimated_weight_lbcs(sample_pauli_str, pauli_term, beta):
        # beta is 2-d, and the shape looks like (len, 3)
        assert len(sample_pauli_str) == len(pauli_term)
        result = 1
        for i in range(num_qubits):
            # The probability distribution is different at each qubit
            score = 0
            idx = pauli2idx[sample_pauli_str[i]]
            if sample_pauli_str[i] == 'i' or pauli_term[i] == 'i':
                score = 1
            elif sample_pauli_str[i] == pauli_term[i] and beta[i][idx] != 0:
                score = 1 / beta[i][idx]
            result *= score
        return result

    def estimated_value(pauli_term, measurement_result):
        value = 1
        for idx in range(num_qubits):
            if pauli_term[idx] != 'i' and measurement_result[idx] == '1':
                value *= -1
        return value

    # Define the functions required by APS
    def is_covered(pauli, pauli_str):
        for qubit_idx in range(num_qubits):
            if pauli[qubit_idx] not in ('i', pauli_str[qubit_idx]):
                return False
        return True

    def update_pauli_estimator(hamiltonian, pauli_estimator, pauli_str, measurement_result):
        for coeff, pauli_term in hamiltonian:
            last_estimator = pauli_estimator[pauli_term]['value'][-1]
            if is_covered(pauli_term, pauli_str):
                value = estimated_value(pauli_term, measurement_result)
                chose_number = pauli_estimator[pauli_term]['times']
                new_estimator = 1 / (chose_number + 1) * (chose_number * last_estimator + value)
                pauli_estimator[pauli_term]['times'] += 1
                pauli_estimator[pauli_term]['value'].append(new_estimator)
            else:
                pauli_estimator[pauli_term]['value'].append(last_estimator)

    trace_estimation = 0
    if method == "CS":
        for sample_idx in range(sample_shots):
            estimation = 0
            for i in range(len(pauli_terms)):
                value = estimated_value(pauli_terms[i], sample_measurement_result[sample_idx])
                weight = estimated_weight_cs(sample_pauli_str[sample_idx], pauli_terms[i])
                estimation += coeff_terms[i] * weight * value
            trace_estimation += estimation
        trace_estimation /= sample_shots
    elif method == "LBCS":
        for sample_idx in range(sample_shots):
            estimation = 0
            for i in range(len(pauli_terms)):
                value = estimated_value(pauli_terms[i], sample_measurement_result[sample_idx])
                weight = estimated_weight_lbcs(sample_pauli_str[sample_idx], pauli_terms[i], beta)
                estimation += coeff_terms[i] * weight * value
            trace_estimation += estimation
        trace_estimation /= sample_shots
    elif method == "APS":
        # Create a search dictionary for easy storage
        pauli_estimator = {}
        for coeff, pauli_term in hamiltonian:
            pauli_estimator[pauli_term] = {'times': 0, 'value': [0]}
        for sample_idx in range(sample_shots):
            update_pauli_estimator(
                hamiltonian,
                pauli_estimator,
                sample_pauli_str[sample_idx],
                sample_measurement_result[sample_idx]
            )
        for sample_idx in range(sample_shots):
            estimation = 0
            for coeff, pauli_term in hamiltonian:
                estimation += coeff * pauli_estimator[pauli_term]['value'][sample_idx + 1]
            trace_estimation = estimation

    return trace_estimation


def tensor_state(state_a: State, state_b: State, *args: State) -> State:
    r"""calculate tensor product (kronecker product) between at least two state. This function automatically returns State instance

    Args:
        state_a: State
        state_b: State
        *args: other states

    Returns:
        tensor product state of input states
        
    Notes:
        Need to be careful with the backend of states; 
        Use ``paddle_quantum.linalg.NKron`` if the input datatype is ``paddle.Tensor`` or ``numpy.ndarray``.
        
    """
    state_a, state_b = _type_transform(state_a, "tensor"), _type_transform(state_b, "tensor")
    return State(NKron(state_a, state_b, [_type_transform(st, "tensor") for st in args]))


def diamond_norm(channel_repr: Union[ChoiRepr, KrausRepr, StinespringRepr, paddle.Tensor],
                 dim_io: Union[int, Tuple[int, int]] = None, **kwargs) -> float:
    r'''Calculate the diamond norm of input.

    Args:
        channel_repr: A ``ChoiRepr`` or ``KrausRepr`` or ``StinespringRepr`` instance or a ``paddle.Tensor`` instance.
        dim_io: The input and output dimensions.
        **kwargs: Parameters to set cvx.

    Raises:
        RuntimeError: `channel_repr` must be `ChoiRepr`or `KrausRepr` or `StinespringRepr` or `paddle.Tensor`.
        TypeError: "dim_io" should be "int" or "tuple".

    Warning:
        `channel_repr` is not in Choi representaiton, and is converted into `ChoiRepr`.

    Returns:
        Its diamond norm.

    Reference:
        Khatri, Sumeet, and Mark M. Wilde. "Principles of quantum communication theory: A modern approach."
        arXiv preprint arXiv:2011.04672 (2020).
        Watrous, J. . "Semidefinite Programs for Completely Bounded Norms." 
        Theory of Computing 5.1(2009):217-238.
    '''
    if isinstance(channel_repr, ChoiRepr):
        choi_matrix = channel_repr.choi_oper

    elif isinstance(channel_repr, paddle.Tensor):
        choi_matrix = channel_repr

    elif isinstance(channel_repr, (KrausRepr, StinespringRepr)):
        warnings.warn('`channel_repr` is not in Choi representaiton, and is converted into `ChoiRepr`')
        choi_matrix = channel_convert(channel_repr, 'Choi').choi_oper

    else:
        raise RuntimeError('`channel_repr` must be `ChoiRepr`or `KrausRepr` or `StinespringRepr` or `paddle.Tensor`.')

    if dim_io is None:    # Default to dim_in == dim_out
        dim_in = dim_out = int(math.sqrt(choi_matrix.shape[0]))
    elif isinstance(dim_io, tuple):
        dim_in = int(dim_io[0])
        dim_out = int(dim_io[1])
    elif isinstance(dim_io, int):
        dim_in = dim_io
        dim_out = dim_io
    else:
        raise TypeError('"dim_io" should be "int" or "tuple".')
    kron_size = dim_in * dim_out

    # Cost function : Trace( \Omega @ Choi_matrix )
    rho = cvxpy.Variable(shape=(dim_in, dim_in), complex=True)
    omega = cvxpy.Variable(shape=(kron_size, kron_size), complex=True)
    identity = np.eye(dim_out)

    # \rho \otimes 1 \geq \Omega
    cons_matrix = cvxpy.kron(rho, identity) - omega
    cons = [
        rho >> 0,
        rho.H == rho,
        cvxpy.trace(rho) == 1,

        omega >> 0,
        omega.H == omega,

        cons_matrix >> 0
    ]

    obj = cvxpy.Maximize(2 * cvxpy.real((cvxpy.trace(omega @ choi_matrix))))
    prob = cvxpy.Problem(obj, cons)

    return prob.solve(**kwargs)


def channel_convert(original_channel: Union[ChoiRepr, KrausRepr, StinespringRepr], target: str, tol: float = 1e-6) -> Union[ChoiRepr, KrausRepr, StinespringRepr]:
    r"""convert the given channel to the target implementation

    Args:
        original_channel: input quantum channel
        target: target implementation, should to be ``Choi``, ``Kraus`` or ``Stinespring``
        tol: error tolerance of the convert, 1e-6 by default

    Raises:
        ValueError: Unsupported channel representation: require Choi, Kraus or Stinespring.

    Returns:
        Union[ChoiRepr, KrausRepr]: quantum channel by the target implementation
        
    Note:
        choi -> kraus currently has the error of order 1e-6 caused by eigh
        
    Raises:
        NotImplementedError: does not support the conversion of input data type
        
    """
    target = target.capitalize()
    if target not in ['Choi', 'Kraus', 'Stinespring']:
        raise ValueError(f"Unsupported channel representation: require Choi, Kraus or Stinespring, not {target}")
    if type(original_channel).__name__ == f'{target}Repr':
        return original_channel

    if isinstance(original_channel, KrausRepr) and target == 'Choi':
        kraus_oper = original_channel.kraus_oper
        ndim = original_channel.kraus_oper[0].shape[0]
        kraus_oper_tensor = paddle.concat([paddle.kron(x, x.conj().T) for x in kraus_oper]).reshape([len(kraus_oper), ndim, -1])
        chan = paddle.sum(kraus_oper_tensor, axis=0).reshape([ndim for _ in range(4)]).transpose([2, 1, 0, 3])
        choi_repr = chan.transpose([0, 2, 1, 3]).reshape([ndim * ndim, ndim * ndim])
        result = ChoiRepr(choi_repr, original_channel.qubits_idx)
        result.qubits_idx = original_channel.qubits_idx

    elif isinstance(original_channel, KrausRepr) and target == 'Stinespring':
        kraus_oper = original_channel.kraus_oper
        j_dim = len(kraus_oper)
        i_dim = kraus_oper[0].shape[0]
        kraus_oper_tensor = paddle.concat(kraus_oper).reshape([j_dim, i_dim, -1])
        stinespring_mat = kraus_oper_tensor.transpose([1, 0, 2])
        stinespring_mat = stinespring_mat.reshape([i_dim * j_dim, i_dim])
        result = StinespringRepr(stinespring_mat, original_channel.qubits_idx)
        result.qubits_idx = original_channel.qubits_idx

    elif isinstance(original_channel, ChoiRepr) and target == 'Kraus':
        choi_repr = original_channel.choi_oper
        ndim = int(math.sqrt(original_channel.choi_oper.shape[0]))
        w, v = paddle.linalg.eigh(choi_repr)

        # add abs to make eigvals safe
        w = paddle.abs(w)
        l_cut = 0
        for l in range(len(w) - 1, -1, -1):
            if paddle.sum(paddle.abs(w[l:])) / paddle.sum(paddle.abs(w)) > 1 - tol:
                l_cut = l
                break
        kraus = [(v * paddle.sqrt(w))[:, l].reshape([ndim, ndim]).T for l in range(l_cut, ndim**2)]

        result = KrausRepr(kraus, original_channel.qubits_idx)
        result.qubits_idx = original_channel.qubits_idx

    elif isinstance(original_channel, StinespringRepr) and target == 'Kraus':
        stinespring_mat = original_channel.stinespring_matrix
        i_dim = stinespring_mat.shape[1]
        j_dim = stinespring_mat.shape[0] // i_dim
        kraus_oper = stinespring_mat.reshape([i_dim, j_dim, i_dim]).transpose([1, 0, 2])
        kraus_oper = [kraus_oper[j] for j in range(j_dim)]
        result = KrausRepr(kraus_oper, original_channel.qubits_idx)
        result.qubits_idx = original_channel.qubits_idx

    else:
        raise NotImplementedError(
            f"does not support the conversion of {type(original_channel)}")
    
    return result


def kraus_oper_random(num_qubits: int, num_oper: int) -> list:
    r""" randomly generate a set of kraus operators for quantum channel

    Args:
        num_qubits: The amount of qubits of quantum channel.
        num_oper: The amount of kraus operators to be generated.

    Returns:
        a list of kraus operators

    """
    float_dtype = _get_float_dtype(get_dtype())
    prob = [paddle.sqrt(paddle.to_tensor(1/num_oper, dtype = float_dtype))] * num_oper
    return [prob[idx] * unitary_random(num_qubits) for idx in range(num_oper)]
