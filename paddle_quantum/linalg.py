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
import itertools
from scipy.stats import unitary_group
from functools import reduce
from typing import Optional, Union, Callable, List, Tuple

import paddle_quantum as pq
from .intrinsic import get_dtype, _get_float_dtype, _type_fetch, _type_transform


def abs_norm(mat: Union[np.ndarray, paddle.Tensor, pq.State]) -> float:
    r""" tool for calculation of matrix norm

    Args:
        mat: matrix

    Returns:
        norm of input matrix

    """
    mat = _type_transform(mat, "tensor")
    mat = mat.cast(get_dtype())
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
    mat = _type_transform(mat, "tensor").cast('complex128')
    shape = mat.shape
    if len(shape) != 2 or shape[0] != shape[1] or math.log2(shape[0]) != math.ceil(math.log2(shape[0])):
        # not a mat / not a square mat / shape is not in form 2^num_qubits x 2^num_qubits
        return False
    return abs_norm(mat - dagger(mat)) < eps

def is_positive(mat: Union[np.ndarray, paddle.Tensor], eps: Optional[float] = 1e-6) -> bool:
    r""" verify whether ``mat`` is a positive semi-definite matrix.

    Args:
        mat: positive operator candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`P` is Hermitian and eigenvalues are non-negative
    
    """
    if is_hermitian(mat, eps):
        mat = _type_transform(mat, "tensor").cast('complex128')
        return (min(paddle.linalg.eigvalsh(mat)) >= -eps).item()
    return False


def is_state_vector(vec: Union[np.ndarray, paddle.Tensor], eps: Optional[float] = None) -> Tuple[bool, int]:
    r""" verify whether ``vec`` is a legal quantum state vector
    
    Args:
        vec: state vector candidate :math:`x`
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness
    
    Returns:
        determine whether :math:`x^\dagger x = 1`, and return the number of qubits or an error message
        
    Note:
        error message is:
        * ``-1`` if the above equation does not hold
        * ``-2`` if the dimension of ``vec`` is not a power of 2
        * ``-3`` if ``vec`` is not a vector
    
    """
    vec = _type_transform(vec, "tensor")
    vec = paddle.squeeze(vec)
    
    dimension = vec.shape[0]
    if len(vec.shape) != 1:
        return False, -3
    
    num_qubits = int(math.log2(dimension))
    if 2 ** num_qubits != dimension:
        return False, -2
    
    if eps is None:
        return True, num_qubits
    
    vec = vec.reshape([dimension, 1])
    vec_bra = paddle.conj(vec.T)   
    eps = min(eps * dimension, 1e-2)
    return {False, -1} if paddle.abs(vec_bra @ vec - (1 + 0j)) > eps else {True, num_qubits}


def is_density_matrix(rho: Union[np.ndarray, paddle.Tensor], eps: Optional[float] = None) -> Tuple[bool, int]:
    r""" verify whether ``rho`` is a legal quantum density matrix
    
    Args:
        rho: density matrix candidate
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness
    
    Returns:
        determine whether ``rho`` is a PSD matrix with trace 1 and return the number of qubits or an error message.
    
    Note:
        error message is:
        * ``-1`` if ``rho`` is not PSD
        * ``-2`` if the trace of ``rho`` is not 1
        * ``-3`` if the dimension of ``rho`` is not a power of 2 
        * ``-4`` if ``rho`` is not a square matrix
    
    """
    rho = _type_transform(rho, "tensor")
    
    dimension = rho.shape[0]
    if len(rho.shape) != 2 or dimension != rho.shape[1]:
        return False, -4
    
    num_qubits = int(math.log2(dimension))
    if 2 ** num_qubits != dimension:
        return False, -3
    
    if eps is None:
        return True, num_qubits
    
    eps = min(eps * dimension, 1e-2)
    if paddle.abs(paddle.trace(rho) - (1 + 0j)).item() > eps:
        return False, -2
    
    return {False, -1} if not is_positive(rho, eps) else {True, num_qubits}


def is_projector(mat: Union[np.ndarray, paddle.Tensor], eps: Optional[float] = 1e-6) -> bool:
    r""" verify whether ``mat`` is a projector

    Args:
        mat: projector candidate :math:`P`
        eps: tolerance of error

    Returns:
        determine whether :math:`PP - P = 0`

    """
    mat = _type_transform(mat, "tensor").cast('complex128')
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
    return paddle.to_tensor(mat / max_eigval, dtype=get_dtype())


def orthogonal_projection_random(num_qubits: int) -> paddle.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` rank-1 orthogonal projector

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` orthogonal projector
    """
    assert num_qubits > 0
    n = 2 ** num_qubits
    float_dtype = _get_float_dtype(get_dtype())
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
    return haar_density_operator(num_qubits, rank=np.random.randint(1,2**num_qubits))


def unitary_random(num_qubits: int) -> paddle.Tensor:
    r"""randomly generate a :math:`2^n \times 2^n` unitary

    Args:
        num_qubits: number of qubits :math:`n`

    Returns:
         a :math:`2^n \times 2^n` unitary matrix
         
    """
    return paddle.to_tensor(unitary_group.rvs(2 ** num_qubits), dtype=get_dtype())


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

    return paddle.to_tensor(mat, dtype=get_dtype())


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
    return paddle.to_tensor(mat_u, dtype=get_dtype())


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
    return paddle.to_tensor(mat_u, dtype=get_dtype())


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

    return paddle.to_tensor(phi, dtype=get_dtype())


def haar_density_operator(num_qubits: int, rank: Optional[int] = None, is_real: Optional[bool] = False) -> paddle.Tensor:
    r""" randomly generate a density matrix following Haar random

        Args:
            num_qubits: number of qubits :math:`n`
            rank: rank of density matrix, default to be ``None`` refering to full ranks
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
    return paddle.to_tensor(rho, dtype=get_dtype())


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

        from paddle_quantum.state import density_op_random
        from paddle_quantum.linalg import NKron
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

    
def herm_transform(fcn: Callable[[float], float], mat: Union[paddle.Tensor, np.ndarray, pq.State], 
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


def pauli_basis_generation(num_qubits: int) -> List[paddle.Tensor]:
    r"""Generate a Pauli basis.
    
    Args:
        num_qubits: the number of qubits :math:`n`.
        
    Returns:
        The Pauli basis of :math:`\mathbb{C}^{2^n \times 2^n}`.

    """
    
    def __single_pauli_basis() -> List[paddle.Tensor]:
        r"""The Pauli basis in single-qubit case.
        """
        complex_dtype = get_dtype()
        I = paddle.to_tensor([[0.5, 0],
                            [0, 0.5]], dtype=complex_dtype)
        X = paddle.to_tensor([[0, 0.5],
                            [0.5, 0]], dtype=complex_dtype)
        Y = paddle.to_tensor([[0, -0.5j],
                            [0.5j, 0]], dtype=complex_dtype)
        Z = paddle.to_tensor([[0.5, 0],
                            [0, -0.5]], dtype=complex_dtype)
        return [I, X, Y, Z]
    
    def __basis_kron(basis_A: List[paddle.Tensor], basis_B: List[paddle.Tensor]) -> List[paddle.Tensor]:
        r"""Kronecker product between bases
        """
        return [
            paddle.kron(basis_A[i], basis_B[j])
            for i, j in itertools.product(range(len(basis_A)), range(len(basis_B)))
        ]
    
    
    list_bases = [__single_pauli_basis() for _ in range(num_qubits)]
    if num_qubits == 1:
        return list_bases[0]
    
    return reduce(lambda result, index: __basis_kron(result, index), list_bases[2:], __basis_kron(list_bases[0], list_bases[1]))


def pauli_decomposition(mat: Union[np.ndarray, paddle.Tensor]) -> Union[np.ndarray, paddle.Tensor]:
    r"""Decompose the matrix by the Pauli basis.
    
    Args:
        mat: the matrix to be decomposed
    
    Returns:
        The list of coefficients corresponding to Pauli basis.
    
    """
    type_str = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")
    
    dimension = mat.shape[0]
    num_qubits = int(math.log2(dimension))
    assert 2 ** num_qubits == dimension, \
        f"Input matrix is not a valid quantum data: received shape {mat.shape}"
        
    basis = pauli_basis_generation(num_qubits)
    decomp = paddle.concat([paddle.trace(mat @ basis[i]) for i in range(dimension ** 2)])
    return _type_transform(decomp, type_str)


def subsystem_decomposition(mat: Union[np.ndarray, paddle.Tensor], 
                            first_basis: Union[List[np.ndarray], List[paddle.Tensor]], 
                            second_basis: Union[List[np.ndarray], List[paddle.Tensor]],
                            inner_prod: Union[Callable[[np.ndarray, np.ndarray], np.ndarray], 
                                              Callable[[paddle.Tensor, paddle.Tensor], paddle.Tensor]]
                            ) -> Union[np.ndarray, paddle.Tensor]:
    r"""Decompose the input matrix by two given bases in two subsystems.
    
    Args:
        mat: the matrix :math:`w` to be decomposed
        first_basis: a basis :math:`\{e_i\}_i` from the first space
        second_basis: a basis :math:`\{f_j\}_j` from the second space
        inner_prod: the inner product of these subspaces
        
    Returns:
        a coefficient matrix :math:`[\beta_{ij}]` such that :math:`w = \sum_{i, j} \beta_{ij} e_i \otimes f_j`.
    
    """
    type_str = _type_fetch(mat)
    mat = _type_transform(mat, "tensor")
    
    if type_str == "numpy":
        first_basis = [paddle.to_tensor(ele) for ele in first_basis]
        second_basis = [paddle.to_tensor(ele) for ele in second_basis]
    
    assert mat.shape == paddle.kron(first_basis[0], second_basis[0]).shape, \
        f"The shape does not agree: received {mat.shape, first_basis[0].shape, second_basis[0].shape}"
    
    first_dim, second_dim = len(first_basis), len(second_basis)
    coef = [
        inner_prod(paddle.kron(first_basis[i], second_basis[j]), mat)
        for i, j in itertools.product(range(first_dim), range(second_dim))
    ]
    coef = paddle.concat(coef).reshape([first_dim, second_dim])
    
    return _type_transform(coef, type_str)
