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
import random
import paddle
import QCompute
import paddle_quantum
from ..backend import Backend
from ..backend import state_vector, density_matrix, quleaf
from ..hamiltonian import Hamiltonian
from typing import Optional, Union, Iterable


class State(object):
    r"""The quantum state class.

        Args:
            data: The mathematical analysis of quantum state.
            num_qubits: The number of qubits contained in the quantum state. Defaults to ``None``, which means it will be inferred by the data.
            backend: Used to specify the backend used. Defaults to ``None``, which means to use the default backend.
            dtype: Used to specify the data dtype of the data. Defaults to ``None``, which means to use the default data type.
        
        Raises:
            Exception: The shape of the data is not correct.
            NotImplementedError: If the backend is wrong or not implemented.
        """
    def __init__(
            self, data: Union[paddle.Tensor, np.ndarray, QCompute.QEnv], num_qubits: Optional[int] = None,
            backend: Optional[paddle_quantum.Backend] = None, dtype: Optional[str] = None
    ):
        # TODO: need to check whether it is a legal quantum state
        super().__init__()
        self.backend = paddle_quantum.get_backend() if backend is None else backend
        if self.backend != Backend.QuLeaf:
            if not isinstance(data, paddle.Tensor):
                data = paddle.to_tensor(data)
            if dtype is not None:
                data = paddle.cast(data, dtype)
        self.dtype = dtype if dtype is not None else paddle_quantum.get_dtype()
        if self.backend == Backend.StateVector:
            if data.shape[-1] == 1:
                data = paddle.squeeze(data)
            self.data = data
            if num_qubits is not None:
                if data.shape[-1] != 2 ** num_qubits:
                    raise Exception("The shape of the data should (2 ** num_qubits, ).")
            else:
                num_qubits = int(math.log2(data.shape[-1]))
                assert 2 ** num_qubits == data.shape[-1], "The length of the state should be the integer power of two."
        elif self.backend == Backend.DensityMatrix:
            self.data = data
            if num_qubits is not None:
                if data.shape[-1] != 2 ** num_qubits or data.shape[-2] != 2 ** num_qubits:
                    raise Exception("The shape of the data should (2 ** num_qubits, 2 ** num_qubits).")
            else:
                assert data.shape[-1] == data.shape[-2], "The data should be a square matrix."
                num_qubits = int(math.log2(data.shape[-1]))
                assert 2 ** num_qubits == data.shape[-1], "The data should be integer power of 2."
        elif self.backend == Backend.UnitaryMatrix:
            self.data = data
            if num_qubits is not None:
                if data.shape[-1] != 2 ** num_qubits or data.shape[-2] != 2 ** num_qubits:
                    raise Exception("The shape of the data should (2 ** num_qubits, 2 ** num_qubits).")
            else:
                assert data.shape[-1] == data.shape[-2], "The data should be a square matrix."
                num_qubits = int(math.log2(data.shape[-1]))
                assert 2 ** num_qubits == data.shape[-1], "The data should be integer power of 2."
        elif self.backend == Backend.QuLeaf:
            self.data = data
            if quleaf.get_quleaf_backend() != QCompute.BackendName.LocalBaiduSim2:
                assert quleaf.get_quleaf_token() is not None, "You must input token tu use cloud server."
            self.gate_history = []
            self.param_list = []
            self.num_param = 0
        else:
            raise NotImplementedError
        self.num_qubits = num_qubits

    @property
    def ket(self) -> paddle.Tensor:
        r""" return the ket form in state_vector backend

        Returns:
            ket form of the state

        """
        if self.backend != Backend.StateVector:
            raise Exception("the backend must be in state_vector to raise the ket form of state")

        return self.data.reshape([2 ** self.num_qubits, 1])
    
    @property
    def bra(self) -> paddle.Tensor:
        r""" return the bra form in state_vector backend

        Returns:
            bra form of the state

        """
        if self.backend != Backend.StateVector:
            raise Exception("the backend must be in state_vector to raise the bra form of state")

        return paddle.conj(self.data.reshape([1, 2 ** self.num_qubits]))


    def numpy(self) -> np.ndarray:
        r"""get the data in numpy.

        Returns:
            The numpy array of the data for the quantum state.
        """
        return self.data.numpy()

    def to(self, backend: str, dtype: str, device: str, blocking: str):
        r"""Change the property of the state.

        Args:
            backend: Specify the new backend of the state.
            dtype: Specify the new data type of the state.
            device: Specify the new device of the state.
            blocking: Specify the new blocking of the state.

        Returns:
            Return a error because this function is not implemented.
        """
        # TODO: to be implemented
        return NotImplementedError

    def clone(self) -> 'paddle_quantum.State':
        r"""Return a copy of the quantum state.

        Returns:
            A new state which is identical to this state.
        """
        return State(self.data, self.num_qubits, self.backend, self.dtype)

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
                for idx in range(0, 2 ** self.num_qubits):
                    prob_amplitude[idx] += state_data[idx, idx].real()
            else:
                raise NotImplementedError
            prob_amplitude = prob_amplitude.tolist()
            all_case = [bin(element)[2:].zfill(self.num_qubits) for element in range(0, 2 ** self.num_qubits)]
            samples = random.choices(all_case, weights=prob_amplitude, k=shots)
            counter = collections.Counter(samples)
            measured_results = []
            for key, val in counter.items():
                measured_results.append(([key[idx] for idx in qubits_list], val))
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
            if qubits_idx is not None:
                # new_result = [(self.__process_string(key, qubits_idx), value) for key, value in result.items()]
                # result = self.__process_similiar(new_result)
                pass
        elif self.backend == paddle_quantum.Backend.StateVector:
            prob_amplitude = paddle.multiply(paddle.conj(self.data), self.data).real()
        elif self.backend == paddle_quantum.Backend.DensityMatrix:
            prob_amplitude = paddle.zeros([2 ** self.num_qubits])
            for idx in range(0, 2 ** self.num_qubits):
                prob_amplitude[idx] += self.data[idx, idx].real()
        else:
            raise NotImplementedError
        if qubits_idx is None:
            prob_array = prob_amplitude
            num_measured = self.num_qubits
        elif isinstance(qubits_idx, (Iterable, int)):
            if isinstance(qubits_idx, int):
                qubits_idx = [qubits_idx]
            else:
                qubits_idx = list(qubits_idx)
            num_measured = len(qubits_idx)
            prob_array = paddle.zeros([2 ** num_measured])
            for idx in range(0, 2 ** self.num_qubits):
                binary = bin(idx)[2:].zfill(self.num_qubits)
                target_qubits = ''
                for qubit_idx in qubits_idx:
                    target_qubits += binary[qubit_idx]
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
            if shots == 0:
                ylabel = "Measured Probabilities"
            else:
                ylabel = "Probabilities"
            state_list = [bin(idx)[2:].zfill(num_measured) for idx in range(0, 2 ** num_measured)]
            plt.bar(range(2 ** num_measured), freq, tick_label=state_list)
            plt.xticks(rotation=90)
            plt.xlabel("Qubit State")
            plt.ylabel(ylabel)
            plt.show()
        return result
