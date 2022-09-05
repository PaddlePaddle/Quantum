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
The source file of the quleaf backend.
"""

import copy
import math
import numpy as np
import paddle
import QCompute
import re
from QCompute import MeasureZ, RX, RY
import paddle_quantum

BACKEND = 'local_baidu_sim2'
TOKEN = None
QCompute.Define.Settings.outputInfo = False
QCompute.Define.Settings.drawCircuitControl = []


def set_quleaf_backend(backend: str) -> None:
    r"""Set the backend of the QuLeaf.

    Args:
        backend: The backend you want to set.
    """
    global BACKEND
    BACKEND = backend


def get_quleaf_backend() -> str:
    r"""Get the current backend of the QuLeaf.

    Returns:
        Current backend of the QuLeaf.
    """
    if isinstance(BACKEND, str):
        return QCompute.BackendName(BACKEND)
    return BACKEND


def set_quleaf_token(token: str) -> None:
    r"""Set the token of the QuLeaf.

    You need to input your token if you want tu use the cloud server.

    Args:
        token: Your token.
    """
    global TOKEN
    TOKEN = token
    QCompute.Define.hubToken = token


def get_quleaf_token() -> str:
    r"""Get the token you set.

    Returns:
        The token you set.
    """
    return TOKEN


def _act_gates_to_state(gate_history: list, quleaf_state: QCompute.QEnv, param_all: list) -> QCompute.QEnv:
    r"""The function to act the quantum gate to the quantum state in the QuLeaf backend.

    Args:
        gate_history: The history of quantum gate, which records the type, parameters and qubits index of the gates.
        quleaf_state: The quantum state in QuLeaf.
        param_all: All the parameters in the gates.

    Raises:
        NotImplementedError: Some quantum gate is not supported in QuLeaf yet.

    Returns:
        The acted quantum state.
    """
    single_qubit_gates = {
        's': QCompute.S, 't': QCompute.T, 'sdg': QCompute.SDG, 'tdg': QCompute.TDG,
        'h': QCompute.H, 'x': QCompute.X, 'y': QCompute.Y, 'z': QCompute.Z,
        'u3': QCompute.U, 'rx': QCompute.RX, 'ry': QCompute.RY, 'rz': QCompute.RZ
    }
    multi_qubits_gates = {
        'cnot': QCompute.CX, 'cx': QCompute.CX, 'cy': QCompute.CY, 'cz': QCompute.CZ, 'swap': QCompute.SWAP,
        'cu': QCompute.CU, 'crx': QCompute.CRX, 'cry': QCompute.CRY, 'crz': QCompute.CRZ,
        'cswap': QCompute.CSWAP, 'ccx': QCompute.CCX
    }
    for gate in gate_history:
        gate_name = gate['gate_name']
        if gate_name in single_qubit_gates:
            gate_func = single_qubit_gates[gate_name]
            fixed_gate = ['s', 't', 'sdg', 'tdg', 'h', 'x', 'y', 'z']
            if gate_name in fixed_gate:
                for _ in range(0, gate['depth']):
                    for qubit_idx in gate['qubits_idx']:
                        gate_func(quleaf_state.Q[qubit_idx])
            elif gate_name == 'u':
                for depth_idx in range(0, gate['depth']):
                    for idx, qubit_idx in enumerate(gate['qubits_idx']):
                        if gate['param_sharing']:
                            param_idx = gate['param'][depth_idx]
                        else:
                            param_idx = gate['param'][depth_idx][idx]
                        gate_param = [param_all[idx] for idx in param_idx]
                        gate_func(*gate_param)(quleaf_state.Q[qubit_idx])
            else:
                for depth_idx in range(0, gate['depth']):
                    for idx, qubit_idx in enumerate(gate['qubits_idx']):
                        if gate['param_sharing']:
                            param_idx = gate['param'][depth_idx]
                        else:
                            param_idx = gate['param'][depth_idx][idx]
                        gate_func(param_all[param_idx])(quleaf_state.Q[qubit_idx])
        elif gate_name in multi_qubits_gates:
            gate_func = multi_qubits_gates[gate_name]
            fixed_gate = ['cnot', 'cx', 'cy', 'cz', 'swap', 'cswap', 'ccx']
            if gate_name in fixed_gate:
                for _ in range(0, gate['depth']):
                    for qubits_idx in gate['qubits_idx']:
                        qubit_list = [quleaf_state.Q[qubit_idx] for qubit_idx in qubits_idx]
                        gate_func(*qubit_list)
            elif gate_name == 'cu':
                for depth_idx in range(0, gate['depth']):
                    for idx, qubits_idx in enumerate(gate['qubits_idx']):
                        if gate['param_sharing']:
                            param_idx = gate['param'][depth_idx]
                        else:
                            param_idx = gate['param'][depth_idx][idx]
                        gate_param = [param_all[idx] for idx in param_idx]
                        qubit_list = [quleaf_state.Q[qubit_idx] for qubit_idx in qubits_idx]
                        gate_func(*gate_param)(*qubit_list)
            else:
                for depth_idx in range(0, gate['depth']):
                    for idx, qubits_idx in enumerate(gate['qubits_idx']):
                        if gate['param_sharing']:
                            param_idx = gate['param'][0]
                        else:
                            param_idx = gate['param'][depth_idx][idx]
                        qubit_list = [quleaf_state.Q[qubit_idx] for qubit_idx in qubits_idx]
                        gate_func(param_all[param_idx])(*qubit_list)
        else:
            raise NotImplementedError
    return quleaf_state


def _expec_val_on_quleaf(state: 'QCompute.QEnv', coeff: 'float', pauli_str: 'str', shots: 'int') -> float:
    r"""Compute the expectation value of the observable with respect to the input state in the QuLeaf backend.

    Args:
        state: The quantum state in the QuLeaf backend.
        coeff: The coefficient value of the pauli string.
        pauli_str: The pauli string, which is a term in hamiltonian.
        shots: The number of measurement shots.

    Raises:
        ValueError: The pauli string should be legal.

    Returns:
        The expectation value of the observable with respect to the input quantum state.
    """
    if pauli_str.lower() == 'i':
        return coeff
    pauli_terms = re.split(r',\s*', pauli_str.lower())
    observed_qubits = []
    # _state = state
    _state = copy.deepcopy(state)
    for pauli_term in pauli_terms:
        pauli_matrix = pauli_term[0]
        qubit_idx = int(pauli_term[1:])
        observed_qubits.append(qubit_idx)
        if pauli_matrix == 'x':
            RY(-np.pi / 2)(_state.Q[qubit_idx])
        elif pauli_matrix == 'y':
            RX(np.pi / 2)(_state.Q[qubit_idx])
        elif pauli_matrix == 'z':
            pass
        else:
            raise ValueError("Cannot recognize the pauli words of the hamiltonian.")
    MeasureZ(*_state.Q.toListPair())
    counts = _state.commit(shots, fetchMeasure=True)['counts']
    filtered_counts = [(counts[key], [key[-idx - 1] for idx in observed_qubits]) for key in counts]
    val = coeff * sum([((-1) ** key.count('1')) * val / shots for val, key in filtered_counts])
    return val


def _get_param_for_gate_list(gate_history: list) -> list:
    r"""Get the all parameters from the gate list.

    Args:
        gate_history: The gate history, it is automatically generated.

    Returns:
        The list of the parameters.
    """
    param_for_gate_list = []
    num_param_in_gate = {
        'rx': 1, 'ry': 1, 'rz': 1, 'u3': 3,
        'crx': 1, 'cry': 1, 'crz': 1, 'cu': 1,
    }
    for gate in gate_history:
        gate_name = gate['gate_name']
        if gate_name in num_param_in_gate:
            num_param = np.array(gate['param']).size
            for _ in range(0, num_param):
                param_for_gate_list.append(gate_name)
    return param_for_gate_list


class ExpecValOp(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
            ctx: paddle.autograd.PyLayerContext,
            param: paddle.Tensor,
            state: 'paddle_quantum.State',
            hamiltonian: 'paddle_quantum.Hamiltonian',
            shots: int,
    ) -> paddle.Tensor:
        r"""The forward function to compute the expectation value of the observable in the QuLeaf Backend.

        Args:
            ctx: To save some variables so that they can be used in the backward function.
            param: The parameters in the previous quantum gates.
            state: The quantum state to be measured.
            hamiltonian: The observable.
            shots: The number of measurement shots.

        Returns:
            The expectation value of the observable for the input state.
        """
        ctx.save_for_backward(param)
        quleaf_state = copy.deepcopy(state.data)
        ctx.quleaf_state = quleaf_state
        gate_history = state.gate_history
        ctx.gate_history = gate_history
        ctx.hamiltonian = hamiltonian
        ctx.shots = shots
        state.gate_history = []
        state.param_list = []
        state.num_param = 0
        param_all = param.tolist()
        _state = copy.deepcopy(quleaf_state)
        acted_state = _act_gates_to_state(gate_history, _state, param_all)
        expec_val = 0
        for coeff, pauli_str in hamiltonian.pauli_str:
            _state = copy.deepcopy(acted_state)
            expec_val += _expec_val_on_quleaf(_state, coeff, pauli_str, shots)
        expec_val = paddle.to_tensor([expec_val], dtype=param.dtype)
        return expec_val

    @staticmethod
    def backward(ctx: paddle.autograd.PyLayerContext, expec_val_grad: paddle.Tensor) -> paddle.Tensor:
        r"""The backward function which is to compute the gradient of the input parameters.

        Args:
            ctx: To get the variables saved in the forward function.
            expec_val_grad: The gradient of the expectation value.

        Returns:
            The gradient of the parameters for the quantum gates.
        """
        param, = ctx.saved_tensor()
        param_all = param.tolist()
        quleaf_state = ctx.quleaf_state
        gate_history = ctx.gate_history
        hamiltonian = ctx.hamiltonian
        shots = ctx.shots
        param_for_gate_list = _get_param_for_gate_list(gate_history)
        assert len(param_for_gate_list) == len(param_all)

        def expec_val_shift(param_idx: int, param_shift: float) -> float:
            param_temp = copy.deepcopy(param_all)
            param_temp[param_idx] += param_shift
            _state = copy.deepcopy(quleaf_state)
            acted_state = _act_gates_to_state(gate_history, _state, param_temp)
            expec_val = 0
            for coeff, pauli_str in hamiltonian.pauli_str:
                _state = copy.deepcopy(acted_state)
                expec_val += _expec_val_on_quleaf(_state, coeff, pauli_str, shots)
            # expec_val = paddle.to_tensor([expec_val], dtype=expec_val_grad.dtype)
            return expec_val

        def general_param_shift(param_idx: int) -> float:
            gate_name = param_for_gate_list[param_idx]
            if gate_name in ['crx', 'cry', 'crz', 'cu']:
                coeff_list = [
                    1 / (16 * math.pow(math.sin(math.pi / 8), 2)),
                    -1 / (16 * math.pow(math.sin(3 * math.pi / 8), 2)),
                    1 / (16 * math.pow(math.sin(5 * math.pi / 8), 2)),
                    - 1 / (16 * math.pow(math.sin(7 * math.pi / 8), 2)),
                ]
                shift_list = [math.pi / 2, 3 * math.pi / 2, 5 * math.pi / 2, 7 * math.pi / 2]
                grad_terms = map(
                    lambda coeff, shift: coeff * expec_val_shift(param_idx, shift),
                    coeff_list, shift_list
                )
                grad = sum(grad_terms)
            else:
                coeff1 = 0.5
                coeff2 = -0.5
                grad = (
                    coeff1 * expec_val_shift(param_idx, math.pi / 2) +
                    coeff2 * expec_val_shift(param_idx, 3 * math.pi / 2)
                )
            return grad

        param_grad = np.zeros(param.size)
        for idx in range(0, param_grad.size):
            param_grad[idx] = general_param_shift(idx)
        param_grad = np.reshape(param_grad, param.shape)
        param_grad = paddle.to_tensor(param_grad, dtype=param.dtype)
        param_grad = expec_val_grad * param_grad
        return param_grad
