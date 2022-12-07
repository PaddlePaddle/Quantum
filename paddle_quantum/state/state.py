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
The basic class of the quantum state.
"""

import collections
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
import random

import paddle
import QCompute
import paddle_quantum
from ..backend import Backend
from ..backend import state_vector, density_matrix, quleaf
from ..hamiltonian import Hamiltonian
from typing import Optional, Union, Iterable, Tuple

class State(object):
    r"""The quantum state class.

        Args:
            data: The mathematical analysis of quantum state.
            num_qubits: The number of qubits contained in the quantum state. Defaults to ``None``, which means it will be inferred by the data.
            backend: Used to specify the backend used. Defaults to ``None``, which means to use the default backend.
            dtype: Used to specify the data dtype of the data. Defaults to ``None``, which means to use the default data type.
            override: whether override the input test. ONLY for internal use. Defaults to ``False``.
        
        Raises:
            ValueError: Cannot recognize the backend.
        """
    def __init__(
            self, data: Union[paddle.Tensor, np.ndarray, QCompute.QEnv], num_qubits: Optional[int] = None,
            backend: Optional[paddle_quantum.Backend] = None, dtype: Optional[str] = None, override: Optional[bool] = False
    ):
        # TODO: need to check whether it is a legal quantum state
        super().__init__()
        self.backend = paddle_quantum.get_backend() if backend is None else backend
        self.dtype = dtype if dtype is not None else paddle_quantum.get_dtype()
        if self.backend != Backend.QuLeaf and not isinstance(data, paddle.Tensor):
            data = paddle.to_tensor(data, dtype=self.dtype)
        self.data = data

        # data input test
        if self.backend == Backend.StateVector:
            if data.shape[-1] == 1:
                self.data = paddle.squeeze(data)

            if not override:
                is_vec, data_message = is_state_vector(data)
                assert is_vec, \
                        f"The input data is not a legal state vector: error message {data_message}"
                assert num_qubits is None or num_qubits == data_message, \
                        f"num_qubits does not agree: expected {num_qubits}, received {data_message}"
                num_qubits = data_message

        elif self.backend == Backend.DensityMatrix:

            if not override:
                is_den, data_message = is_density_matrix(data)
                assert is_den, \
                        f"The input data is not a legal density matrix: error message {data_message}"
                assert num_qubits is None or num_qubits == data_message, \
                        f"num_qubits does not agree: expected {num_qubits}, received {data_message}"
                num_qubits = data_message

        elif self.backend == Backend.UnitaryMatrix:
            # no need to check the state in unitary backend
            if num_qubits is None:
                num_qubits = int(math.log2(data.shape[0]))

        elif self.backend == Backend.QuLeaf:
            if quleaf.get_quleaf_backend() != QCompute.BackendName.LocalBaiduSim2:
                assert quleaf.get_quleaf_token() is not None, "You must input token tu use cloud server."
            self.gate_history = []
            self.param_list = []
            self.num_param = 0
        else:
            raise ValueError(
                f"Cannot recognize the backend {self.backend}")
        self.num_qubits = num_qubits

    @property
    def ket(self) -> paddle.Tensor:
        r"""Return the ket form in state_vector backend

        Raises:
            ValueError: the backend must be in StateVector

        Returns:
            ket form of the state

        """
        if self.backend != Backend.StateVector:
            raise ValueError(
                f"the backend must be in StateVector: received {self.backend}")

        return self.data.reshape([2 ** self.num_qubits, 1])
    
    @property
    def bra(self) -> paddle.Tensor:
        r"""Return the bra form in state_vector backend

        Raises:
            ValueError: the backend must be in StateVector

        Returns:
            bra form of the state

        """
        if self.backend != Backend.StateVector:
            raise ValueError(
                f"the backend must be in StateVector: received {self.backend}")

        return paddle.conj(self.data.reshape([1, 2 ** self.num_qubits]))
    
    def normalize(self) -> None:
        r"""Normalize this state
        
        Raises:
            NotImplementedError: does not support normalization for the backend

        """
        data = self.data
        if self.backend == Backend.StateVector:
            data /= paddle.norm(paddle.abs(data))
        elif self.backend == Backend.DensityMatrix:
            data /= paddle.trace(data)
        else:
            raise NotImplementedError(
                f"Does not support normalization for the backend {self.backend}")
        self.data = data
        
    def evolve(self, H: Union[np.ndarray, paddle.Tensor, Hamiltonian], t: float) -> None:
        r"""Evolve the state under the Hamiltonian `H` i.e. apply unitary operator :math:`e^{-itH}`
        
        Args:
            H: the Hamiltonian of the system
            t: the evolution time
        
        Raises:
            NotImplementedError: does not support evolution for the backend

        """
        if isinstance(H, Hamiltonian):
            num_qubits = max(self.num_qubits, H.n_qubits)
            H = H.construct_h_matrix(qubit_num=num_qubits)
        else:
            num_qubits = int(math.log2(H.shape[0]))
            H = _type_transform(H, "numpy")
        assert num_qubits == self.num_qubits, \
            f"the # of qubits of Hamiltonian and this state are not the same: received {num_qubits}, expect {self.num_qubits}"
        e_itH = paddle.to_tensor(expm(-1j * t * H), dtype=self.dtype)
        
        if self.backend == Backend.StateVector:
            self.data = paddle.squeeze(e_itH @ self.ket)
        elif self.backend == Backend.DensityMatrix:
            self.data = e_itH @ self.data @ paddle.conj(e_itH.T)
        else:
            raise NotImplementedError(
                f"Does not support evolution for the backend {self.backend}")
    
    def kron(self, other: 'State') -> 'State':
        r"""Kronecker product between states
        
        Args:
            other: a quantum state
            
        Returns:
            the tensor product of these two states
            
        """
        backend, dtype = self.backend, self.dtype
        assert backend == other.backend, \
            f"backends between two States are not the same: received {backend} and {other.backend}"
        assert dtype == other.dtype, \
            f"dtype between two States are not the same: received {dtype} and {other.dtype}"
        
        if backend == Backend.StateVector:
            return State(paddle.kron(self.ket, other.ket), backend=backend)
        else:
            return State(paddle.kron(self.data, other.data), backend=backend, dtype=dtype)
        
    def __matmul__(self, other: 'State') -> paddle.Tensor:
        r"""Matrix product between states or between tensor and state
        
        Args:
            other: a quantum state

        Raises:
            NotImplementedError: does not support the product between State and input format.
            ValueError: Cannot multiply two state vectors: check the backend

        Returns:
            the product of these two states
        """
        if not isinstance(other, State):
            raise NotImplementedError(
                f"does not support the product between State and {type(other)}")
        if self.backend == Backend.StateVector:
            raise ValueError(
                "Cannot multiply two state vectors: check the backend")
        return self.data @ other.data
    
    def __rmatmul__(self, other: 'State') -> paddle.Tensor:
        r"""Matrix product between states or between state and tensor
        
        Args:
            other: a quantum state

        Raises:
            NotImplementedError: does not support the product between State and input format.
            ValueError: Cannot multiply two state vectors: check the backend

        Returns:
            the product of these two states

        """
        if not isinstance(other, State):
            raise NotImplementedError(
                f"does not support the product between {type(other)} and State")
        if self.backend == Backend.StateVector:
            raise ValueError(
                "Cannot multiply two state vectors: check your backend")
        return other.data @ self.data
    
    def numpy(self) -> np.ndarray:
        r"""Get the data in numpy.

        Returns:
            The numpy array of the data for the quantum state.
        """
        return self.data.numpy()

    def to(self, backend: str, dtype: str = None, device: str = None, blocking: str = None) -> None:
        r"""Change the property of the state.

        Args:
            backend: Specify the new backend of the state.
            dtype: Specify the new data type of the state.
            device: Specify the new device of the state.
            blocking: Specify the new blocking of the state.
        
        Raises
            NotImplementedError: only support transformation between StateVector and DensityMatrix
            NotImplementedError: Transformation for device or blocking is not supported.

        """
        if (backend not in {"state_vector", "density_matrix"} or 
            self.backend not in {Backend.StateVector, Backend.DensityMatrix}):
            raise NotImplementedError(
                "Only support transformation between StateVector and DensityMatrix")
        
        if device is not None or blocking is not None:
            raise NotImplementedError(
                "Transformation for device or blocking is not supported")
        
        self.data = _type_transform(self, backend).data
        self.dtype = self.dtype if dtype is None else dtype

    def clone(self) -> 'State':
        r"""Return a copy of the quantum state.

        Returns:
            A new state which is identical to this state.
        """
        return State(self.data, self.num_qubits, self.backend, self.dtype, override=True)

    def __str__(self):
        return str(self.data.numpy())

    def expec_val(self, hamiltonian: Hamiltonian, shots: Optional[int] = 0) -> float:
        r"""The expectation value of the observable with respect to the quantum state.

        Args:
            hamiltonian: Input observable.
            shots: Number of measurement shots.

        Raises:
            NotImplementedError: If the backend is wrong or not implemented.

        Returns:
            The expectation value of the input observable for the quantum state.
        """
        if shots == 0:
            func = paddle_quantum.loss.ExpecVal(hamiltonian)
            result = func(self)
            return result
        result = 0
        gate_for_x = paddle.to_tensor([
            [1 / math.sqrt(2), 1 / math.sqrt(2)],
            [1 / math.sqrt(2), -1 / math.sqrt(2)],
        ], dtype=self.dtype)
        gate_for_y = paddle.to_tensor([
            [1 / math.sqrt(2), -1j / math.sqrt(2)],
            [1 / math.sqrt(2), 1j / math.sqrt(2)],
        ], dtype=self.dtype)
        if self.backend == Backend.StateVector:
            simulator = state_vector.unitary_transformation
        elif self.backend == Backend.DensityMatrix:
            simulator = density_matrix.unitary_transformation
        else:
            raise NotImplementedError
        for coeff, pauli_str in hamiltonian.pauli_str:
            pauli_list = pauli_str.split(',')
            state_data = paddle.to_tensor(self.data.numpy())
            if pauli_str.lower() == 'i':
                result += coeff
                continue
            qubits_list = []
            for pauli_term in pauli_list:
                pauli = pauli_term[0]
                idx = int(pauli_term[1:])
                qubits_list.append(idx)
                if pauli.lower() == 'x':
                    state_data = simulator(
                        state_data, gate_for_x, idx, self.num_qubits
                    )
                elif pauli.lower() == 'y':
                    state_data = simulator(
                        state_data, gate_for_y, idx, self.num_qubits
                    )
            if self.backend == paddle_quantum.Backend.StateVector:
                prob_amplitude = paddle.multiply(paddle.conj(state_data), state_data).real()
            elif self.backend == paddle_quantum.Backend.DensityMatrix:
                prob_amplitude = paddle.zeros([2 ** self.num_qubits])
                for idx in range(2 ** self.num_qubits):
                    prob_amplitude[idx] += state_data[idx, idx].real()
            else:
                raise NotImplementedError
            prob_amplitude = prob_amplitude.tolist()
            all_case = [bin(element)[2:].zfill(self.num_qubits) for element in range(2 ** self.num_qubits)]
            samples = random.choices(all_case, weights=prob_amplitude, k=shots)
            counter = collections.Counter(samples)
            measured_results = [([key[idx] for idx in qubits_list], val) for key, val in counter.items()]
            temp_res = sum(((-1) ** key.count('1') * val / shots for key, val in measured_results))
            result += coeff * temp_res
        return result

    def measure(self, shots: Optional[int] = 0, qubits_idx: Optional[Union[Iterable[int], int]] = None, 
                plot: Optional[bool] = False) -> dict:
        r"""Measure the quantum state

        Args:
            shots: the number of measurements on the quantum state output by the quantum circuit.
                Default is ``0``, which means the exact probability distribution of measurement results are returned.
            qubits_idx: The index of the qubit to be measured. Defaults to ``None``, which means all qubits are measured.
            plot: Whether to draw the measurement result plot. Defaults to ``False`` which means no plot.

        Raises:
            Exception: The number of shots should be greater than 0.
            NotImplementedError: If the backend is wrong or not implemented.
            NotImplementedError: The qubit index is wrong or not supported.

        Returns:
            Measurement results
        """
        if self.backend == paddle_quantum.Backend.QuLeaf:
            if shots == 0:
                    raise Exception("The quleaf server requires the number of shots to be greater than 0.")
            state_data = self.data
            QCompute.MeasureZ(*state_data.Q.toListPair())
            result = state_data.commit(shots, fetchMeasure=True)['counts']
            result = {''.join(reversed(key)): value for key, value in result.items()}
        elif self.backend == paddle_quantum.Backend.StateVector:
            prob_amplitude = paddle.multiply(paddle.conj(self.data), self.data).real()
        elif self.backend == paddle_quantum.Backend.DensityMatrix:
            prob_amplitude = paddle.zeros([2 ** self.num_qubits])
            for idx in range(2 ** self.num_qubits):
                prob_amplitude[idx] += self.data[idx, idx].real()
        else:
            raise NotImplementedError
        if qubits_idx is None:
            prob_array = prob_amplitude
            num_measured = self.num_qubits
        elif isinstance(qubits_idx, (Iterable, int)):
            qubits_idx = [qubits_idx] if isinstance(qubits_idx, int) else list(qubits_idx)
            num_measured = len(qubits_idx)
            prob_array = paddle.zeros([2 ** num_measured])
            for idx in range(2 ** self.num_qubits):
                binary = bin(idx)[2:].zfill(self.num_qubits)
                target_qubits = ''.join(binary[qubit_idx] for qubit_idx in qubits_idx)
                prob_array[int(target_qubits, base=2)] += prob_amplitude[idx]
        else:
            raise NotImplementedError
        if shots == 0:
            freq = prob_array.tolist()
            result = {bin(idx)[2:].zfill(num_measured): val for idx, val in enumerate(freq)}
        else:
            samples = random.choices(range(0, 2 ** num_measured), weights=prob_array, k=shots)
            freq = [0] * (2 ** num_measured)
            for item in samples:
                freq[item] += 1
            freq = [val / shots for val in freq]
            result = {bin(idx)[2:].zfill(num_measured): val for idx, val in enumerate(freq)}
        if plot:
            assert num_measured < 6, "Too many qubits to plot"
            ylabel = "Measured Probabilities" if shots == 0 else "Probabilities"
            state_list = [bin(idx)[2:].zfill(num_measured) for idx in range(0, 2 ** num_measured)]
            plt.bar(range(2 ** num_measured), freq, tick_label=state_list)
            plt.xticks(rotation=90)
            plt.xlabel("Qubit State")
            plt.ylabel(ylabel)
            plt.show()
        return result


def _type_fetch(data: Union[np.ndarray, paddle.Tensor, State]) -> str:
    r""" fetch the type of ``data``
    
    Args:
        data: the input data, and datatype of which should be either ``numpy.ndarray``,
    ''paddle.Tensor'' or ``paddle_quantum.State``
    
    Returns:
        string of datatype of ``data``, can be either ``"numpy"``, ``"tensor"``,
    ``"state_vector"`` or ``"density_matrix"``
    
    Raises:
        ValueError: does not support the current backend of input state.
        TypeError: cannot recognize the current type of input data.
    
    """
    if isinstance(data, np.ndarray):
        return "numpy"
    
    if isinstance(data, paddle.Tensor):
        return "tensor"
    
    if isinstance(data, State):
        if data.backend == Backend.StateVector:
            return "state_vector"
        if data.backend == Backend.DensityMatrix:
            return "density_matrix"
        raise ValueError(
            f"does not support the current backend {data.backend} of input state.")
        
    raise TypeError(
        f"cannot recognize the current type {type(data)} of input data.")


def _density_to_vector(rho: Union[np.ndarray, paddle.Tensor]) -> Union[np.ndarray, paddle.Tensor]:
    r""" transform a density matrix to a state vector
    
    Args:
        rho: a density matrix (pure state)
        
    Returns:
        a state vector
    
    Raises:
        ValueError: the output state may not be a pure state
    
    """
    type_str = _type_fetch(rho)
    rho = paddle.to_tensor(rho)
    eigval, eigvec = paddle.linalg.eigh(rho)
            
    max_eigval = paddle.max(eigval).item()
    err = np.abs(max_eigval - 1)
    if err > 1e-6:
        raise ValueError(
            f"the output state may not be a pure state, maximum distance: {err}")
    
    state = eigvec[:, paddle.argmax(eigval)]
    
    return state.numpy() if type_str == "numpy" else state
    

def _type_transform(data: Union[np.ndarray, paddle.Tensor, State],
                   output_type: str) -> Union[np.ndarray, paddle.Tensor, State]:
    r""" transform the datatype of ``input`` to ``output_type``
    
    Args:
        data: data to be transformed
        output_type: datatype of the output data, type is either ``"numpy"``, ``"tensor"``,
    ``"state_vector"`` or ``"density_matrix"``
    
    Returns:
        the output data with expected type
        
    Raises:
        ValueError: does not support transformation to type.
    
    """
    current_type = _type_fetch(data)
    
    support_type = {"numpy", "tensor", "state_vector", "density_matrix"}
    if output_type not in support_type:
        raise ValueError(
            f"does not support transformation to type {output_type}")
        
    if current_type == output_type:
        return data
    
    if current_type == "numpy":
        if output_type == "tensor":
            return paddle.to_tensor(data)
        
        data = np.squeeze(data)        
        # state_vector case
        if output_type == "state_vector":
            if len(data.shape) == 2:
                data = _density_to_vector(data)
            return State(data, backend=Backend.StateVector)
        # density_matrix case
        if len(data.shape) == 1:
            data = data.reshape([len(data), 1])
            data = data @ np.conj(data.T)
        return State(data, backend=Backend.DensityMatrix)
    
    if current_type == "tensor":
        if output_type == "numpy":
            return data.numpy()
        
        data = paddle.squeeze(data)
        # state_vector case
        if output_type == "state_vector":
            if len(data.shape) == 2:
                data = _density_to_vector(data)
            return State(data, backend=Backend.StateVector)
        
        # density_matrix case
        if len(data.shape) == 1:
            data = data.reshape([len(data), 1])
            data = data @ paddle.conj(data.T)
        return State(data, backend=Backend.DensityMatrix)

    if current_type == "state_vector":
        if output_type == "density_matrix":
            return State(data.ket @ data.bra, backend=Backend.DensityMatrix, num_qubits=data.num_qubits, override=True)
        return data.ket.numpy() if output_type == "numpy" else data.ket
    
    # density_matrix data
    if output_type == "state_vector":
        return State(_density_to_vector(data.data), backend=Backend.StateVector, num_qubits=data.num_qubits, override=True)
    return data.numpy() if output_type == "numpy" else data.data


def is_state_vector(vec: Union[np.ndarray, paddle.Tensor], 
                    eps: Optional[float] = None) -> Tuple[bool, int]:
    r""" verify whether ``vec`` is a legal quantum state vector
    
    Args:
        vec: state vector candidate :math:`x`
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness
    
    Returns:
        determine whether :math:`x^\dagger x = 1`, and return the number of qubits or an error message
        
    Notes:
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


def is_density_matrix(rho: Union[np.ndarray, paddle.Tensor], 
                      eps: Optional[float] = None) -> Tuple[bool, int]:
    r""" verify whether ``rho`` is a legal quantum density matrix
    
    Args:
        rho: density matrix candidate
        eps: tolerance of error, default to be `None` i.e. no testing for data correctness
    
    Returns:
        determine whether ``rho`` is a PSD matrix with trace 1 and return the number of qubits or an error message.
    
    Notes:
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
    
    min_eigval = paddle.min(paddle.linalg.eigvalsh(rho))
    return {False, -1} if paddle.abs(min_eigval) > eps else {True, num_qubits}
