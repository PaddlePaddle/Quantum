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
The function for quantum information.
"""

import paddle_quantum
import math
import re
import numpy as np
from scipy.linalg import logm, sqrtm
import paddle
from paddle import kron
from paddle import matmul
from paddle import transpose
from paddle_quantum.intrinsic import _get_float_dtype
import matplotlib.image
from paddle_quantum.linalg import dagger, is_unitary, NKron
from typing import Optional, Tuple, List


def partial_trace(rho_AB: paddle_quantum.State, dim1: int, dim2: int, A_or_B: int) -> paddle.Tensor:
    r"""Calculate the partial trace of the quantum state.

    Args:
        rho_AB: Input quantum state.
        dim1: The dimension of system A.
        dim2: The dimension of system B.
        A_or_B: 1 or 2. 1 means to calculate partial trace on system A; 2 means to calculate partial trace on system B.

    Returns:
        Partial trace of the input quantum state.
    """
    if A_or_B == 2:
        dim1, dim2 = dim2, dim1
    
    rho_AB = rho_AB.data
    complex_dtype = paddle_quantum.get_dtype()
    float_dtype = _get_float_dtype(complex_dtype)

    idty_np = np.identity(dim2).astype(complex_dtype)
    idty_B = paddle.to_tensor(idty_np)

    zero_np = np.zeros([dim2, dim2], complex_dtype)
    res = paddle.to_tensor(zero_np)

    for dim_j in range(dim1):
        row_top = paddle.zeros([1, dim_j])
        row_mid = paddle.ones([1, 1])
        row_bot = paddle.zeros([1, dim1 - dim_j - 1])
        bra_j = paddle.concat([row_top, row_mid, row_bot], axis=1)
        bra_j = paddle.cast(bra_j, complex_dtype)

        if A_or_B == 1:
            row_tmp = kron(bra_j, idty_B)
            row_tmp_conj = paddle.conj(row_tmp)
            res = paddle.add(res, paddle.matmul(paddle.matmul(row_tmp, rho_AB), paddle.transpose(row_tmp_conj, perm=[1, 0]), ), )

        if A_or_B == 2:
            row_tmp = kron(idty_B, bra_j)
            row_tmp_conj = paddle.conj(row_tmp)
            res = paddle.add(res, paddle.matmul(paddle.matmul(row_tmp, rho_AB), paddle.transpose(row_tmp_conj, perm=[1, 0]), ), )

    return res


def partial_trace_discontiguous(rho: paddle_quantum.State, preserve_qubits: Optional[list] = None) -> paddle.Tensor:
    r"""Calculate the partial trace of the quantum state with arbitrarily selected subsystem

    Args:
        rho: Input quantum state.
        preserve_qubits: Remaining qubits, default is None, indicate all qubits remain.

    Returns:
        Partial trace of the quantum state with arbitrarily selected subsystem.
    """
    if type(rho) == paddle_quantum.State:
        rho = rho.data
    complex_dtype = paddle_quantum.get_dtype()
    float_dtype = _get_float_dtype(complex_dtype)

    if preserve_qubits is None:
        return rho
    else:
        n = int(math.log2(rho.size) // 2)
        num_preserve = len(preserve_qubits)

        shape = paddle.ones((n + 1,))
        shape = 2 * shape
        shape[n] = 2 ** n
        shape = paddle.cast(shape, "int32")
        identity = paddle.eye(2 ** n)
        identity = paddle.reshape(identity, shape=shape)
        discard = list()
        for idx in range(0, n):
            if idx not in preserve_qubits:
                discard.append(idx)
        addition = [n]
        preserve_qubits.sort()

        preserve_qubits = paddle.to_tensor(preserve_qubits)
        discard = paddle.to_tensor(discard)
        addition = paddle.to_tensor(addition)
        permute = paddle.concat([discard, preserve_qubits, addition])

        identity = paddle.transpose(identity, perm=permute)
        identity = paddle.reshape(identity, (2 ** n, 2 ** n))

        result = np.zeros((2 ** num_preserve, 2 ** num_preserve), dtype=complex_dtype)
        result = paddle.to_tensor(result)

        for i in range(0, 2 ** (n - num_preserve)):
            bra = identity[i * 2 ** num_preserve:(i + 1) * 2 ** num_preserve, :]
            result = result + matmul(matmul(bra, rho), transpose(bra, perm=[1, 0]))

        return result


def trace_distance(rho: paddle_quantum.State, sigma: paddle_quantum.State) -> paddle.Tensor:
    r"""Calculate the trace distance between two quantum states.

    .. math::

        D(\rho, \sigma) = 1 / 2 * \text{tr}|\rho-\sigma|

    Args:
        rho: Density matrix form of the quantum state.
        sigma: Density matrix form of the quantum state.

    Returns:
        Trace distance between the input quantum states.
    """
    assert rho.data.shape == sigma.data.shape, 'The shape of two quantum states are different'
    A = rho.data.numpy() - sigma.data.numpy()
    distance = 1 / 2 * np.sum(np.abs(np.linalg.eigvals(A)))

    return paddle.to_tensor(distance)


def state_fidelity(rho: paddle_quantum.State, sigma: paddle_quantum.State) -> paddle.Tensor:
    r"""Calculate the fidelity of two quantum states.

    .. math::
        F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

    Args:
        rho: Density matrix form of the quantum state.
        sigma: Density matrix form of the quantum state.

    Returns:
        The fidelity between the input quantum states.
    """
    rho = rho.data.numpy()
    sigma = sigma.data.numpy()
    assert rho.shape == sigma.shape, 'The shape of two quantum states are different'
    fidelity = np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real

    return paddle.to_tensor(fidelity)


def gate_fidelity(U: paddle.Tensor, V: paddle.Tensor) -> paddle.Tensor:
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
    complex_dtype = paddle_quantum.get_dtype()
    U = paddle.to_tensor(U, dtype=complex_dtype)
    V = paddle.to_tensor(V, dtype=complex_dtype)
    # assert is_unitary(U), "U is not a unitary"
    # assert is_unitary(V), "V is not a unitary"
    assert U.shape == V.shape, 'The shape of two unitary matrices are different'
    
    fidelity = paddle.abs(paddle.trace(U @ dagger(V))) / U.shape[0]
    return fidelity


def purity(rho: paddle_quantum.State) -> paddle.Tensor:
    r"""Calculate the purity of a quantum state.

    .. math::

        P = \text{tr}(\rho^2)

    Args:
        rho: Density matrix form of the quantum state.

    Returns:
        The purity of the input quantum state.
    """
    rho = rho.data
    gamma = paddle.trace(paddle.matmul(rho, rho))

    return gamma.real()


def von_neumann_entropy(rho: paddle_quantum.State) -> paddle.Tensor:
    r"""Calculate the von Neumann entropy of a quantum state.

    .. math::

        S = -\text{tr}(\rho \log(\rho))

    Args:
        rho: Density matrix form of the quantum state.

    Returns:
        The von Neumann entropy of the input quantum state.
    """
    rho = rho.data.numpy()
    rho_eigenvalues = np.real(np.linalg.eigvals(rho))
    entropy = 0
    for eigenvalue in rho_eigenvalues:
        if np.abs(eigenvalue) < 1e-8:
            continue
        entropy -= eigenvalue * np.log(eigenvalue)

    return paddle.to_tensor(entropy)


def relative_entropy(rho: paddle_quantum.State, sig: paddle_quantum.State) -> paddle.Tensor:
    r"""Calculate the relative entropy of two quantum states.

    .. math::

        S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)


    Args:
        rho: Density matrix form of the quantum state.
        sig: Density matrix form of the quantum state.

    Returns:
        Relative entropy between input quantum states.
    """
    rho = rho.data.numpy() 
    sig = sig.data.numpy()
    assert rho.shape == sig.shape, 'The shape of two quantum states are different'
    res = np.trace(rho @ logm(rho) - rho @ logm(sig))
    return paddle.to_tensor(res.real)


def random_pauli_str_generator(n: int, terms: Optional[int] = 3) -> List:
    r"""Generate a random observable in list form.

    An observable :math:`O=0.3X\otimes I\otimes I+0.5Y\otimes I\otimes Z`'s list form is
    ``[[0.3, 'x0'], [0.5, 'y0,z2']]``.  Such an observable is generated by 
    ``random_pauli_str_generator(3, terms=2)`` 

    Args:
        n: Number of qubits.
        terms: Number of terms in the observable. Defaults to 3.

    Returns:
        The randomly generated observable’s list form.
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
        sub_matrices = []
        for op in op_str:
            sub_matrices.append(pauli_dict[op.lower()])
        if len(op_str) == 1:
            matrices.append(coeff * sub_matrices[0])
        else:
            matrices.append(coeff * NKron(sub_matrices[0], sub_matrices[1], *sub_matrices[2:]))

    return paddle.to_tensor(sum(matrices), dtype=paddle_quantum.get_dtype())


def partial_transpose_2(density_op: paddle_quantum.State, sub_system: Optional[int] = None) -> paddle.Tensor:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.
        sub_system: 1 or 2. 1 means to perform partial transpose on system A; 2 means to perform partial trace on system B. Default is 2.

    Returns:
        The partial transpose of the input quantum state.
    """
    sys_idx = 2 if sub_system is None else 1

    # Copy the density matrix and not corrupt the original one
    density_op = density_op.data.numpy()
    transposed_density_op = np.copy(density_op)
    if sys_idx == 2:
        for j in [0, 2]:
            for i in [0, 2]:
                transposed_density_op[i:i + 2, j:j + 2] = density_op[i:i + 2, j:j + 2].transpose()
    else:
        transposed_density_op[2:4, 0:2] = density_op[0:2, 2:4]
        transposed_density_op[0:2, 2:4] = density_op[2:4, 0:2]

    return paddle.to_tensor(transposed_density_op)


def partial_transpose(density_op: paddle_quantum.State, n: int) -> paddle.Tensor:
    r"""Calculate the partial transpose :math:`\rho^{T_A}` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.
        n: Number of qubits of the system to be transposed.

    Returns:
        The partial transpose of the input quantum state.
    """
    # Copy the density matrix and not corrupt the original one
    density_op = density_op.data.numpy()
    transposed_density_op = np.copy(density_op)
    for j in range(0, 2 ** n, 2):
        for i in range(0, 2 ** n, 2):
            transposed_density_op[i:i + 2, j:j + 2] = density_op[i:i + 2, j:j + 2].transpose()

    return paddle.to_tensor(transposed_density_op)


def negativity(density_op: paddle_quantum.State) -> paddle.Tensor:
    r"""Compute the Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Negativity of the input quantum state.
    """
    # Implement the partial transpose
    density_op_T = partial_transpose_2(density_op)

    # Calculate through the equivalent expression N = sum(abs(\lambda_i)) when \lambda_i<0
    n = 0.0
    eigen_val, _ = np.linalg.eig(density_op_T)
    for val in eigen_val:
        if val < 0:
            n = n + np.abs(val)
    return paddle.to_tensor(n)


def logarithmic_negativity(density_op: paddle_quantum.State) -> paddle.Tensor:
    r"""Calculate the Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||` of the input quantum state.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        The Logarithmic Negativity of the input quantum state.
    """
    # Calculate the negativity
    n = negativity(density_op)

    # Calculate through the equivalent expression
    log2_n = paddle.log2(2 * n + 1)
    return log2_n


def is_ppt(density_op: paddle_quantum.State) -> bool:
    r"""Check if the input quantum state is PPT.

    Args:
        density_op: Density matrix form of the quantum state.

    Returns:
        Whether the input quantum state is PPT.
    """
    # By default the PPT condition is satisfied
    ppt = True

    # Detect negative eigenvalues from the partial transposed density_op
    if negativity(density_op) > 0:
        ppt = False
    return ppt


def schmidt_decompose(psi: paddle_quantum.State, sys_A: Optional[List[int]] = None) -> Tuple[paddle.Tensor]:
    r"""Calculate the Schmidt decomposition of a quantum state :math:`\lvert\psi\rangle=\sum_ic_i\lvert i_A\rangle\otimes\lvert i_B \rangle`.

    Args:
        psi: State vector form of the quantum state, with shape（2**n）
        sys_A: Qubit indices to be included in subsystem A (other qubits are included in subsystem B), default are the first half qubits of :math:`\lvert \psi\rangle`

    Returns:
        contains elements

        * A one dimensional array composed of Schmidt coefficients, with shape ``(k)``
        * A high dimensional array composed of bases for subsystem A :math:`\lvert i_A\rangle`, with shape ``(k, 2**m, 1)``
        * A high dimensional array composed of bases for subsystem B :math:`\lvert i_B\rangle` , with shape ``(k, 2**m, 1)``
    """
    psi = psi.data.numpy()
    assert psi.ndim == 1, 'Psi must be a one dimensional vector.'
    assert np.log2(psi.size).is_integer(), 'The number of amplitutes must be an integral power of 2.'


    tot_qu = int(np.log2(psi.size))
    sys_A = sys_A if sys_A is not None else [i for i in range(tot_qu//2)]
    sys_B = [i for i in range(tot_qu) if i not in sys_A]

    # Permute qubit indices
    psi = psi.reshape([2] * tot_qu).transpose(sys_A + sys_B)

    # construct amplitute matrix
    amp_mtr = psi.reshape([2**len(sys_A), 2**len(sys_B)])

    # Standard process to obtain schmidt decomposition
    u, c, v = np.linalg.svd(amp_mtr)

    k = np.count_nonzero(c > 1e-13)
    c = c[:k]
    u = u.T[:k].reshape([k, -1, 1])
    v = v[:k].reshape([k, -1, 1])
    return paddle.to_tensor(c), paddle.to_tensor(u), paddle.to_tensor(v)


def image_to_density_matrix(image_filepath: str) -> paddle_quantum.State:
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
    return paddle_quantum.State(paddle.to_tensor(rho), backend=paddle_quantum.Backend.DensityMatrix)


def shadow_trace(state: 'paddle_quantum.State', hamiltonian: paddle_quantum.Hamiltonian, 
                 sample_shots: int, method: Optional[str] = 'CS') -> float:
    r"""Estimate the expectation value :math:`\text{trace}(H\rho)`  of an observable :math:`H`.

    Args:
        state: Quantum state.
        hamiltonian: Observable.
        sample_shots: Number of samples.
        method: Method used to , which should be one of “CS”, “LBCS”, and “APS”. Default is “CS”.
    
    Returns:
        The estimated expectation value for the hamiltonian.
    """
    if not isinstance(hamiltonian, list):
        hamiltonian = hamiltonian.pauli_str
    num_qubits = state.num_qubits
    mode = state.backend
    if method == "LBCS":
        result, beta = paddle_quantum.shadow.shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian, method)
    else:
        result = paddle_quantum.shadow.shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian, method)

    def prepare_hamiltonian(hamiltonian, num_qubits):
        new_hamiltonian = list()
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
            if not pauli[qubit_idx] in ('i', pauli_str[qubit_idx]):
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
