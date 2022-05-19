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
Linear algebra functions which support complex data type (Complex64, Complex128).
"""

import paddle

__all__ = ["_general_vnorm", "_general_vv", "_general_mv", "_hermitian_expv"]


def _general_vnorm(vec: paddle.Tensor) -> paddle.Tensor:
    r"""
    Calculate the vector norm for complex vector ||v||^2.

    Args:
        vec: Complex valued vector.
    
    Returns:
        Real scalar.
    """

    if vec.dtype in [paddle.complex64, paddle.complex128]:
        return paddle.dot(vec.real(), vec.real()) + paddle.dot(vec.imag(), vec.imag())
    else:
        return paddle.dot(vec, vec)


def _general_vv(vec1: paddle.Tensor, vec2: paddle.Tensor) -> paddle.Tensor:
    r"""
    Calculate the vector dot product between two complex vectors.
    
    Args:
        vec1: Complex valued vector.
        vec2: Complex valued vector.

    Returns:
        Complex scalar.
    """

    try:
        vec1.dtype == vec2.dtype
    except AssertionError:
        raise ValueError(f"Tensors are expected to have the same dtype, but receive {vec1.dtype} and {vec2.dtype}")
    if vec1.dtype in [paddle.complex64, paddle.complex128]:
        return paddle.dot(vec1.real(), vec2.real()) + paddle.dot(vec2.imag(), vec2.imag()) + \
               1j * paddle.dot(vec1.real(), vec2.imag()) - 1j * paddle.dot(vec1.imag(), vec2.real())
    else:
        return paddle.dot(vec1, vec2)


def _general_mv(mat: paddle.Tensor, vec: paddle.Tensor) -> paddle.Tensor:
    r"""
    Calculate the complex matrix vector multiplication.

    Args:
        mat: Complex valued matrix.
        vec: Complex valued vector.

    Returns:
        Complex valued tensor.
    """

    try:
        mat.dtype == vec.dtype
    except AssertionError:
        raise ValueError(f"Tensors are expected to have the same dtype, but receive {mat.dtype} and {vec.dtype}")
    if mat.dtype in [paddle.complex64, paddle.complex128]:
        return paddle.mv(mat.real(), vec.real()) - paddle.mv(mat.imag(), vec.imag()) + \
               1j * paddle.mv(mat.real(), vec.imag()) + 1j * paddle.mv(mat.imag(), vec.real())
    else:
        return paddle.mv(mat, vec)


def _hermitian_expv(mat: paddle.Tensor, vec: paddle.Tensor) -> paddle.Tensor:
    r"""
    Calculate 
    .. math:
        \langle v|M|v\rangle

    where M is a Hermitian matrix:
        M.real(): real symmetric matrix.
        M.imag(): real antisymmetric matrix.

    Args:
        mat: Complex valued Hermitian matrix.
        vec: Complex valued vector.

    Returns:
        A scalar, paddle.Tensor.
    """

    try:
        mat.dtype == vec.dtype
    except AssertionError:
        raise ValueError(f"Tensors are expected to have the same dtype, but receive {mat.dtype} and {vec.dtype}")
    if mat.dtype in [paddle.complex64, paddle.complex128]:
        return paddle.dot(vec.real(), paddle.mv(mat.real(), vec.real())) + \
               paddle.dot(vec.imag(), paddle.mv(mat.real(), vec.imag())) + \
               2 * paddle.dot(vec.imag(), paddle.mv(mat.imag(), vec.real()))
    else:
        return paddle.dot(vec, paddle.mv(mat, vec))
