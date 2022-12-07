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
The library of functions in linear algebra.
"""

import paddle
import math
import numpy as np
import scipy
from scipy.stats import unitary_group
from functools import reduce
from typing import Optional, Union, Callable

import paddle_quantum as pq
from .intrinsic import _get_float_dtype
from .state import State, _type_fetch, _type_transform


def abs_norm(mat: Union[np.ndarray, paddle.Tensor, State]) -> float:
    r""" tool for calculation of matrix norm

    Args:
        mat: matrix

    Returns:
        norm of mat

    """
    mat = _type_transform(mat, "tensor")
    mat = mat.cast(pq.get_dtype())
    return paddle.norm(paddle.abs(mat)).item()


def dagger(mat: Union[np.ndarray, paddle.Tensor]) -> Union[np.ndarray, paddle.Tensor]:
    r""" tool for calculation of matrix dagger

    Args:
        mat: matrix

    Returns:
        The dagger of matrix

    """
    type_str = _type_fetch(mat)
    return np.conj(mat.T) if type_str == "numpy" else paddle.conj(mat.T)


def is_hermitian(mat: Union[np.ndarray, paddle.Tensor], eps: Optional[float] = 1e-6) -> bool:
    r""" verify whether ``mat`` is Hermitian

    Args:
        mat: hermitian candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`P - P^\dagger = 0`

    """
    mat = _type_transform(mat, "tensor")
    shape = mat.shape
    if len(shape) != 2 or shape[0] != shape[1] or math.log2(shape[0]) != math.ceil(math.log2(shape[0])):
        # not a mat / not a square mat / shape is not in form 2^num_qubits x 2^num_qubits
        return False
    return abs_norm(mat - dagger(mat)) < eps


def is_projector(mat: Union[np.ndarray, paddle.Tensor], eps: Optional[float] = 1e-6) -> bool:
    r""" verify whether ``mat`` is a projector

    Args:
        mat: projector candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`PP - P = 0`

    """
    mat = _type_transform(mat, "tensor")
    shape = mat.shape
    if len(shape) != 2 or shape[0] != shape[1] or math.log2(shape[0]) != math.ceil(math.log2(shape[0])):
        # not a mat / not a square mat / shape is not in form 2^num_qubits x 2^num_qubits
        return False
    return abs_norm(mat @ mat - mat) < eps


def is_unitary(mat: Union[np.ndarray, paddle.Tensor], eps: Optional[float] = 1e-4) -> bool:
    r""" verify whether ``mat`` is a unitary

    Args:
        mat: unitary candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`PP^\dagger - I = 0`

    """
    mat = _type_transform(mat, "tensor").cast('complex128')
    shape = mat.shape
    eps = min(eps * shape[0], 1e-2)
    if len(shape) != 2 or shape[0] != shape[1] or math.log2(shape[0]) != math.ceil(math.log2(shape[0])):
        # not a mat / not a square mat / shape is not in form 2^num_qubits x 2^num_qubits
        return False
    return abs_norm(mat @ dagger(mat) - paddle.cast(paddle.eye(shape[0]), mat.dtype)) < eps


def hermitian_random(num_qubits: int) -> paddle.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` hermitian matrix

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` hermitian matrix

    """
    assert num_qubits > 0
    n = 2 ** num_qubits
    
    mat = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    for i in range(n):
        mat[i, i] = np.abs(mat[i, i])
        for j in range(i):
            mat[i, j] = np.conj(mat[j, i])
    
    eigval= np.linalg.eigvalsh(mat)
    max_eigval = np.max(np.abs(eigval))
    return paddle.to_tensor(mat / max_eigval, dtype=pq.get_dtype())


def orthogonal_projection_random(num_qubits: int) -> paddle.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` rank-1 orthogonal projector

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` orthogonal projector
    """
    assert num_qubits > 0
    n = 2 ** num_qubits
    float_dtype = _get_float_dtype(pq.get_dtype())
    vec = paddle.randn([n, 1], dtype=float_dtype) + 1j * paddle.randn([n, 1], dtype=float_dtype)
    mat = vec @ dagger(vec)
    return mat / paddle.trace(mat)


def density_matrix_random(num_qubits: int) -> paddle.Tensor:
    r""" randomly generate an num_qubits-qubit state in density matrix form
    
    Args:
        num_qubits: number of qubits :math:`n`
    
    Returns:
        a :math:`2^n \times 2^n` density matrix
        
    """
    float_dtype = _get_float_dtype(pq.get_dtype())
    real = paddle.rand([2 ** num_qubits, 2 ** num_qubits], dtype=float_dtype)
    imag = paddle.rand([2 ** num_qubits, 2 ** num_qubits], dtype=float_dtype)
    M = real + 1j * imag
    M = M @ dagger(M)
    return M / paddle.trace(M)


def unitary_random(num_qubits: int) -> paddle.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` unitary

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` unitary matrix
         
    """
    return paddle.to_tensor(unitary_group.rvs(2 ** num_qubits), dtype=pq.get_dtype())


def unitary_hermitian_random(num_qubits: int) -> paddle.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` hermitian unitary

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` hermitian unitary matrix
         
    """
    proj_mat = orthogonal_projection_random(num_qubits)
    id_mat = paddle.eye(2 ** num_qubits)
    return (2 + 0j) * proj_mat - id_mat


def unitary_random_with_hermitian_block(num_qubits: int, is_unitary: bool = False) -> paddle.Tensor:
    r"""randomly generate a unitary :math:`2^n \times 2^n` matrix that is a block encoding of a :math:`2^{n/2} \times 2^{n/2}` Hermitian matrix

    Args:
        num_qubits: number of qubits :math:`n`
        is_unitary: whether the hermitian block is a unitary divided by 2 (for tutorial only)

    Returns:
         a :math:`2^n \times 2^n` unitary matrix that its upper-left block is a Hermitian matrix

    """
    assert num_qubits > 0
    
    if is_unitary:
        mat0 = unitary_hermitian_random(num_qubits - 1).numpy() / 2
    else:
        mat0 = hermitian_random(num_qubits - 1).numpy()
    id_mat = np.eye(2 ** (num_qubits - 1))
    mat1 = 1j * scipy.linalg.sqrtm(id_mat - np.matmul(mat0, mat0))

    mat = np.block([[mat0, mat1], [mat1, mat0]])

    return paddle.to_tensor(mat, dtype=pq.get_dtype())


def block_enc_herm(mat: Union[np.ndarray, paddle.Tensor], 
                   num_block_qubits: int = 1) -> Union[np.ndarray, paddle.Tensor]:
    r""" generate a (qubitized) block encoding of hermitian ``mat``
    
    Args:
        mat: matrix to be block encoded
        num_block_qubits: ancilla qubits used in block encoding
    
    Returns:
        a unitary that is a block encoding of ``mat``
    
    """
    assert is_hermitian(mat), "the input matrix is not a hermitian"
    assert mat.shape[0] == mat.shape[1], "the input matrix is not a square matrix"
    
    type_mat = _type_fetch(mat)
    H = _type_transform(mat, "numpy")
    complex_dtype = mat.dtype
    
    num_qubits = int(math.log2(mat.shape[0]))
    H_complement = scipy.linalg.sqrtm(np.eye(2 ** num_qubits) - H @ H)
    block_enc = np.block([[H, 1j * H_complement], [1j * H_complement, H]])
    block_enc = paddle.to_tensor(block_enc, dtype=complex_dtype)
    
    if num_block_qubits > 1:
        block_enc = direct_sum(block_enc, paddle.eye(2 ** (num_block_qubits + num_qubits) - 2 ** (num_qubits + 1)).cast(complex_dtype))
    
    return _type_transform(block_enc, type_mat)


def haar_orthogonal(num_qubits: int) -> paddle.Tensor:
    r""" randomly generate an orthogonal matrix following Haar random, referenced by arXiv:math-ph/0609050v2

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
        a :math:`2^n \times 2^n` orthogonal matrix
        
    """
    # Matrix dimension
    dim = 2 ** num_qubits
    # Step 1: sample from Ginibre ensemble
    ginibre = (np.random.randn(dim, dim))
    # Step 2: perform QR decomposition of G
    mat_q, mat_r = np.linalg.qr(ginibre)
    # Step 3: make the decomposition unique
    mat_lambda = np.diag(mat_r) / abs(np.diag(mat_r))
    mat_u = mat_q @ np.diag(mat_lambda)
    return paddle.to_tensor(mat_u, dtype=pq.get_dtype())


def haar_unitary(num_qubits: int) -> paddle.Tensor:
    r""" randomly generate a unitary following Haar random, referenced by arXiv:math-ph/0609050v2

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
        a :math:`2^n \times 2^n` unitary
        
    """
    # Matrix dimension
    dim = 2 ** num_qubits
    # Step 1: sample from Ginibre ensemble
    ginibre = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)) / np.sqrt(2)
    # Step 2: perform QR decomposition of G
    mat_q, mat_r = np.linalg.qr(ginibre)
    # Step 3: make the decomposition unique
    mat_lambda = np.diag(mat_r) / np.abs(np.diag(mat_r))
    mat_u = mat_q @ np.diag(mat_lambda)
    return paddle.to_tensor(mat_u, dtype=pq.get_dtype())


def haar_state_vector(num_qubits: int, is_real: Optional[bool] = False) -> paddle.Tensor:
    r""" randomly generate a state vector following Haar random

        Args:
            num_qubits: number of qubits :math:`n`
            is_real: whether the vector is real, default to be False

        Returns:
            a :math:`2^n \times 1` state vector
            
    """
    # Vector dimension
    dim = 2 ** num_qubits
    if is_real:
        # Generate a Haar random orthogonal matrix
        mat_orthog = haar_orthogonal(num_qubits)
        # Perform u onto |0>, i.e., the first column of o
        phi = mat_orthog[:, 0]
    else:
        # Generate a Haar random unitary
        unitary = haar_unitary(num_qubits)
        # Perform u onto |0>, i.e., the first column of u
        phi = unitary[:, 0]

    return paddle.to_tensor(phi, dtype=pq.get_dtype())


def haar_density_operator(num_qubits: int, rank: Optional[int] = None, is_real: Optional[bool] = False) -> paddle.Tensor:
    r""" randomly generate a density matrix following Haar random

        Args:
            num_qubits: number of qubits :math:`n`
            rank: rank of density matrix, default to be False refering to full ranks
            is_real: whether the density matrix is real, default to be False

        Returns:
            a :math:`2^n \times 2^n` density matrix
    """
    dim = 2 ** num_qubits
    rank = rank if rank is not None else dim
    assert 0 < rank <= dim, 'rank is an invalid number'
    if is_real:
        ginibre_matrix = np.random.randn(dim, rank)
        rho = ginibre_matrix @ ginibre_matrix.T
    else:
        ginibre_matrix = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
        rho = ginibre_matrix @ ginibre_matrix.conj().T
    rho = rho / np.trace(rho)
    return paddle.to_tensor(rho / np.trace(rho), dtype=pq.get_dtype())


def direct_sum(A: Union[np.ndarray, paddle.Tensor], 
               B: Union[np.ndarray, paddle.Tensor]) -> Union[np.ndarray, paddle.Tensor]:
    r""" calculate the direct sum of A and B
    
    Args:
        A: :math:`m \times n` matrix
        B: :math:`p \times q` matrix
        
    Returns:
        a direct sum of A and B, with shape :math:`(m + p) \times (n + q)`
    
    
    """
    type_A, type_B = _type_fetch(A), _type_fetch(B)
    A, B = _type_transform(A, "numpy"), _type_transform(B, "numpy")

    assert A.dtype == B.dtype, f"A's dtype {A.dtype} does not agree with B's dtype {B.dtype}"

    zero_AB, zero_BA = np.zeros([A.shape[0], B.shape[1]]), np.zeros([B.shape[0], A.shape[1]])
    mat = np.block([[A, zero_AB], [zero_BA, B]])

    return mat if type_A == "numpy" or type_B == "numpy" else paddle.to_tensor(mat)


def NKron(
        matrix_A: Union[paddle.Tensor, np.ndarray],
        matrix_B: Union[paddle.Tensor, np.ndarray],
        *args: Union[paddle.Tensor, np.ndarray]
    ) -> Union[paddle.Tensor, np.ndarray]:
    r""" calculate Kronecker product of at least two matrices

    Args:
        matrix_A: matrix, as paddle.Tensor or numpy.ndarray
        matrix_B: matrix, as paddle.Tensor or numpy.ndarray
        *args: other matrices, as paddle.Tensor or numpy.ndarray

    Returns:
        Kronecker product of matrices, determined by input type of matrix_A

    .. code-block:: python

        from pq.state import density_op_random
        from pq.linalg import NKron
        A = density_op_random(2)
        B = density_op_random(2)
        C = density_op_random(2)
        result = NKron(A, B, C)

    Note:
        ``result`` from above code block should be A \otimes B \otimes C
    """
    type_A, type_B = _type_fetch(matrix_A), _type_fetch(matrix_A)
    assert type_A == type_B, f"the input data types do not agree: received {type_A} and {type_B}"

    if type_A == "tensor":
        return reduce(lambda result, index: paddle.kron(result, index), args, paddle.kron(matrix_A, matrix_B), )
    else:
        return reduce(lambda result, index: np.kron(result, index), args, np.kron(matrix_A, matrix_B), )

    
def herm_transform(fcn: Callable[[float], float], mat: Union[paddle.Tensor, np.ndarray, State], 
                   ignore_zero: Optional[bool] = False) -> paddle.Tensor:
    r""" function transformation for Hermitian matrix
    
    Args:
        fcn: function :math:`f` that can be expanded by Taylor series
        mat: hermitian matrix :math:`H`
        ignore_zero: whether ignore eigenspaces with zero eigenvalue, defaults to be ``False``
    
    Returns
        :math:`f(H)`
    
    """
    assert is_hermitian(mat), \
        "the input matrix is not Hermitian: check your input"
    type_str = _type_fetch(mat)
    mat = _type_transform(mat, "tensor") if type_str != "state_vector" else mat.ket @ mat.bra
    
    eigval, eigvec = paddle.linalg.eigh(mat)
    eigval = eigval.tolist()
    eigvec = eigvec.T
    
    mat = paddle.zeros(mat.shape).cast(mat.dtype)
    for i in range(len(eigval)):
        vec = eigvec[i].reshape([mat.shape[0], 1])
        
        if np.abs(eigval[i]) < 1e-5 and ignore_zero:
            continue
        mat += (fcn(eigval[i]) + 0j) * vec @ dagger(vec)
    return mat.numpy() if type_str == "numpy" else mat
