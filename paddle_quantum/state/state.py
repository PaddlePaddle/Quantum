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
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
import random

import paddle
import QCompute
import paddle_quantum as pq
from ..backend import Backend, state_vector, density_matrix, quleaf
from ..base import get_backend, get_dtype
from typing import Optional, Union, Iterable, List, Dict


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
            backend: Optional[Backend] = None, dtype: Optional[str] = None, override: Optional[bool] = False
    ):
        # TODO: need to check whether it is a legal quantum state
        super().__init__()
        self.backend = get_backend() if backend is None else backend
        self.dtype = dtype if dtype is not None else get_dtype()
        # TODO: If the dtype is right, don't do the cast operation.
        if isinstance(data, paddle.Tensor):
            self.data = paddle.cast(data, self.dtype) if data.dtype != self.dtype else data
        elif self.backend != Backend.QuLeaf:
            data = paddle.to_tensor(data, dtype=self.dtype)
            self.data = paddle.cast(data, self.dtype)
        # data input test
        if self.backend == Backend.StateVector:
            if data.shape[-1] == 1:
                self.data = paddle.squeeze(data)
            if not override:
                is_vec, data_message = pq.linalg.is_state_vector(data)
                assert is_vec, f"The input data is not a legal state vector: error message {data_message}"
                assert num_qubits is None or num_qubits == data_message, (
                    f"num_qubits does not agree: expected {num_qubits}, received {data_message}")
                num_qubits = data_message
        elif self.backend == Backend.DensityMatrix:
            if not override:
                is_den, data_message = pq.linalg.is_density_matrix(data)
                assert is_den, f"The input data is not a legal density matrix: error message {data_message}"
                assert num_qubits is None or num_qubits == data_message, (
                    f"num_qubits does not agree: expected {num_qubits}, received {data_message}")
                num_qubits = data_message
        elif self.backend == Backend.UnitaryMatrix:
            # no need to check the state in unitary backend
            if num_qubits is None:
                num_qubits = int(math.log2(data.shape[0]))
        elif self.backend == Backend.QuLeaf:
            self.data = data
            if quleaf.get_quleaf_backend() != QCompute.BackendName.LocalBaiduSim2:
                assert quleaf.get_quleaf_token() is not None, "You must input token tu use cloud server."
        else:
            raise ValueError(
                f"Cannot recognize the backend {self.backend}")
        self.num_qubits = num_qubits
        
        # initialize default qubit sequence 
        self.qubit_sequence = list(range(num_qubits))

        # whether use the old doubly-swap logic to compute the matrix multiplication. Defaults to ``False``.
        self.is_swap_back = True

    def __getitem__(self, key: Union[int, slice]) -> 'State':
        r"""Indexing of the State class
        """
        qubits = list(range(self.num_qubits))[key]
        return pq.qinfo.partial_trace_discontiguous(self, qubits if isinstance(qubits, List) else [qubits])

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

    def evolve(self, H: Union[np.ndarray, paddle.Tensor, 'pq.Hamiltonian'], t: float) -> None:
        r"""Evolve the state under the Hamiltonian `H` i.e. apply unitary operator :math:`e^{-itH}`

        Args:
            H: the Hamiltonian of the system
            t: the evolution time

        Raises:
            NotImplementedError: does not support evolution for the backend

        """
        if isinstance(H, pq.Hamiltonian):
            num_qubits = max(self.num_qubits, H.n_qubits)
            H = H.construct_h_matrix(qubit_num=num_qubits)
        else:
            num_qubits = int(math.log2(H.shape[0]))
            H = pq.intrinsic._type_transform(H, "numpy")
        assert num_qubits == self.num_qubits, (
            "the # of qubits of Hamiltonian and this state are not the same: "
            f"received {num_qubits}, expect {self.num_qubits}")
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
        if (
            backend not in {"state_vector", "density_matrix"} or
            self.backend not in {Backend.StateVector, Backend.DensityMatrix}
        ):
            raise NotImplementedError(
                "Only support transformation between StateVector and DensityMatrix")

        if device is not None or blocking is not None:
            raise NotImplementedError(
                "Transformation for device or blocking is not supported")

        self.data = pq.intrinsic._type_transform(self, backend).data
        self.dtype = self.dtype if dtype is None else dtype

    def clone(self) -> 'State':
        r"""Return a copy of the quantum state.

        Returns:
            A new state which is identical to this state.
        """
        new_state = State(paddle.clone(self.data), self.num_qubits, self.backend, self.dtype, override=True)
        new_state.qubit_sequence = self.qubit_sequence
        new_state.is_swap_back = self.is_swap_back
        return new_state

    def __copy__(self) -> 'State':
        return self.clone()

    def __str__(self):
        return str(self.data.numpy())
    
    @property
    def oper_history(self) -> List[Dict[str, Union[str, List[int], paddle.Tensor]]]:
        r"""The operator history stored for the QPU backend
        
        Raises:
            NotImplementedError: This property should be called for the backend ``quleaf`` only.
            ValueError: This state does not have operator history: run the circuit first.
        
        """
        if self.backend != Backend.QuLeaf:
            raise NotImplementedError(
                "This property should be called for backend `quleaf only.")
        
        if not hasattr(self, '_State__oper_history'):
            raise ValueError(
                "This state does not have operator history: run the circuit first.")
        return self.__oper_history

    @oper_history.setter
    def oper_history(self, oper_history: List[Dict[str, Union[str, List[int], paddle.Tensor]]]) -> None:
        if self.backend != Backend.QuLeaf:
            raise NotImplementedError(
                "This property should be changed for backend `quleaf only.")
        self.__oper_history = oper_history

    def expec_val(self, hamiltonian: 'pq.Hamiltonian', shots: Optional[int] = 0) -> float:
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
            func = pq.loss.ExpecVal(hamiltonian)
            result = func(self)
            return result.item()
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
            if self.backend == Backend.StateVector:
                prob_amplitude = paddle.multiply(paddle.conj(state_data), state_data).real()
            elif self.backend == Backend.DensityMatrix:
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

    def measure(
            self, shots: Optional[int] = 0, qubits_idx: Optional[Union[Iterable[int], int]] = None,
            plot: Optional[bool] = False, record: Optional[bool] = False
    ) -> dict:
        r"""Measure the quantum state

        Args:
            shots: the number of measurements on the quantum state output by the quantum circuit.
                Default is ``0``, which means the exact probability distribution of measurement results are returned.
            qubits_idx: The index of the qubit to be measured. Defaults to ``None``, which means all qubits are measured.
            plot: Whether to draw the measurement result plot. Defaults to ``False`` which means no plot.
            record: Whether to return the original measurement record. Defaults to ``False`` which means no record.

        Raises:
            ValueError: The number of shots should be greater than 0.
            NotImplementedError: When the backend is Quleaf, record is not supprted .
            NotImplementedError: If the backend is wrong or not implemented.
            NotImplementedError: The qubit index is wrong or not supported.
            ValueError: Returning records requires the number of shots to be greater than 0.

        Returns:
            Measurement results
        """
        if self.backend == Backend.QuLeaf:
            if shots == 0:
                raise ValueError("The quleaf server requires the number of shots to be greater than 0.")
            if record == True:
                raise NotImplementedError
            state_data = copy.deepcopy(self.data)
            if qubits_idx is None:
                qubits_idx = list(range(self.num_qubits))
            elif isinstance(qubits_idx, int):
                qubits_idx = [qubits_idx]
            num_measured = len(qubits_idx)
            q_reg, _ = state_data.Q.toListPair()
            q_reg = [q_reg[idx] for idx in qubits_idx]
            c_reg = list(range(num_measured))
            state_data = quleaf._act_gates_to_state(self.oper_history, state_data)
            QCompute.MeasureZ(q_reg, c_reg)
            result = state_data.commit(shots, fetchMeasure=True)['counts']
            result = {key[::-1]: value for key, value in result.items()}
            basis_list = [bin(idx)[2:].zfill(num_measured) for idx in range(2 ** num_measured)]
            freq = [0 if basis not in result else result[basis] / shots for basis in basis_list]
        else:
            if self.backend == Backend.StateVector:
                prob_amplitude = paddle.multiply(paddle.conj(self.data), self.data).real()
            elif self.backend == Backend.DensityMatrix:
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
                if record == True:
                    raise ValueError("Returning records requires the number of shots to be greater than 0.")
                freq = prob_array.tolist()
            else:
                result_record = [] # record of original measurement results
                samples = random.choices(range(2 ** num_measured), weights=prob_array, k=shots)
                freq = [0] * (2 ** num_measured)
                for item in samples:
                    freq[item] += 1
                    result_record.append(bin(item)[2:].zfill(num_measured))
                freq = [val / shots for val in freq]
            if record == False:
                result = {bin(idx)[2:].zfill(num_measured): val for idx, val in enumerate(freq)}
            else:
                result = {"Measurement Record": result_record}
        if plot:
            assert num_measured < 6, "Too many qubits to plot"
            ylabel = "Measured Probabilities" if shots == 0 else "Probabilities"
            basis_list = [bin(idx)[2:].zfill(num_measured) for idx in range(2 ** num_measured)]
            plt.bar(range(2 ** num_measured), freq, tick_label=basis_list)
            plt.xticks(rotation=90)
            plt.xlabel("Qubit State")
            plt.ylabel(ylabel)
            plt.show()
        return result
    
    def __set_qubit_seq__(self, qubit_sequence: List[int]) -> None:

        data = self.data
        perm_map = pq.intrinsic._perm_of_list(self.qubit_sequence, qubit_sequence)
        if self.backend == Backend.StateVector:
            higher_dims = data.shape[:-1]
            data = pq.intrinsic._base_transpose(data, perm_map)
            data = paddle.reshape(data, higher_dims.copy() + [2 ** self.num_qubits])

        elif self.backend == Backend.DensityMatrix:
            higher_dims = data.shape[:-2]
            data = pq.intrinsic._base_transpose_for_dm(data, perm_map)
            data = paddle.reshape(data, higher_dims.copy() + [2 ** self.num_qubits] * 2)
        else:
            raise NotImplementedError

        self.data = data
        self.qubit_sequence = qubit_sequence

    def reset_sequence(self, target_sequence: Optional[List[int]] = None) -> None: 
        r"""reset the qubit order to a given sequence

        Args:
            target_sequence: target sequence. Defaults to None.

        Returns:
            State: state with given qubit order
        """

        if target_sequence is None:
            self.__set_qubit_seq__(list(range(self.num_qubits)))
        elif target_sequence == self.qubit_sequence:
            pass
        else:
            if len(target_sequence) != self.num_qubits:
                raise ValueError(f"The length of target sequence does not match! expected {self.num_qubits}, received {len(target_sequence)}")
            self.__set_qubit_seq__(target_sequence)
