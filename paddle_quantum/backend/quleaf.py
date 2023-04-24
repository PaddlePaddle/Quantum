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
import paddle_quantum as pq
from QCompute import MeasureZ, RX, RY
from typing import Optional, List

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


single_qubit_gates = {
    'S': QCompute.S, 'T': QCompute.T, 'Sdg': QCompute.SDG, 'Tdg': QCompute.TDG,
    'H': QCompute.H, 'X': QCompute.X, 'Y': QCompute.Y, 'Z': QCompute.Z,
    'U3': QCompute.U, 'RX': QCompute.RX, 'RY': QCompute.RY, 'RZ': QCompute.RZ
}
multi_qubits_gates = {
    'CNOT': QCompute.CX, 'CX': QCompute.CX, 'CY': QCompute.CY, 'CZ': QCompute.CZ, 'SWAP': QCompute.SWAP,
    'CU': QCompute.CU, 'CRX': QCompute.CRX, 'CRY': QCompute.CRY, 'CRZ': QCompute.CRZ,
    'CSWAP': QCompute.CSWAP, 'CCX': QCompute.CCX
}
fixed_gates = {'S', 'T', 'Sdg', 'Tdg', 'H', 'X', 'Y', 'Z', 'CNOT', 'CX', 'CY', 'CZ', 'SWAP', 'CSWAP', 'CCX'}
parameterized_gates = {'U3', 'RX', 'RY', 'RZ', 'CU', 'CRX', 'CRY', 'CRZ'}


# TODO: need to check whether the logic is right when the param_sharing is True.
def _act_gates_to_state(
        oper_history: List[dict], quleaf_state: QCompute.QEnv,
        _gate_idx: Optional[int] = None, _depth_idx: Optional[int] = None, _qubits_idx: Optional[int] = None,
        _param_idx: Optional[int] = None, _param_shift_val: Optional[int] = None
) -> QCompute.QEnv:
    r"""The function to act the quantum gate to the quantum state in the QuLeaf backend.

    Args:
        oper_history: The history of quantum gate, which records the type, parameters and qubits index of the gates.
        quleaf_state: The quantum state in QuLeaf.
        param_all: All the parameters in the gates.

    Raises:
        NotImplementedError: Some quantum gate is not supported in QuLeaf yet.

    Returns:
        The acted quantum state.
    """
    for gate_idx, gate_info in enumerate(oper_history):
        gate_name = gate_info['name']
        if gate_name in single_qubit_gates:
            gate_func = single_qubit_gates[gate_name]
            if gate_name in fixed_gates:
                for _ in range(gate_info['depth']):
                    for qubit_idx in gate_info['qubits_idx']:
                        gate_func(quleaf_state.Q[qubit_idx])
            elif gate_name == 'U3':
                for depth_idx in range(gate_info['depth']):
                    for idx, qubit_idx in enumerate(gate_info['qubits_idx']):
                        if gate_info['param_sharing']:
                            param = gate_info['param'][depth_idx].tolist()
                        else:
                            param = gate_info['param'][depth_idx][idx].tolist()
                        if gate_idx == _gate_idx and depth_idx == _depth_idx and idx == _qubits_idx:
                            param[_param_idx] += _param_shift_val
                        gate_func(*param)(quleaf_state.Q[qubit_idx])
            else:
                # Single qubit parameterized quantum gate
                for depth_idx in range(gate_info['depth']):
                    for idx, qubit_idx in enumerate(gate_info['qubits_idx']):
                        if gate_info['param_sharing']:
                            param = gate_info['param'][depth_idx][0].item()
                        else:
                            param = gate_info['param'][depth_idx][idx][0].item()
                        if gate_idx == _gate_idx and depth_idx == _depth_idx and idx == _qubits_idx:
                            param += _param_shift_val
                        gate_func(param)(quleaf_state.Q[qubit_idx])
        elif gate_name in multi_qubits_gates:
            gate_func = multi_qubits_gates[gate_name]
            if gate_name in fixed_gates:
                for _ in range(gate_info['depth']):
                    for qubits_idx in gate_info['qubits_idx']:
                        gate_func(*[quleaf_state.Q[qubit_idx] for qubit_idx in qubits_idx])
            elif gate_name == 'CU':
                for depth_idx in range(gate_info['depth']):
                    for idx, qubits_idx in enumerate(gate_info['qubits_idx']):
                        if gate_info['param_sharing']:
                            param = gate_info['param'][depth_idx].tolist()
                        else:
                            param = gate_info['param'][depth_idx][idx].tolist()
                        if gate_idx == _gate_idx and depth_idx == _depth_idx and idx == _qubits_idx:
                            param[_param_idx] += _param_shift_val
                        gate_func(*param)(*[quleaf_state.Q[qubit_idx] for qubit_idx in qubits_idx])
            else:
                # CRx, CRy, CRz
                for depth_idx in range(gate_info['depth']):
                    for idx, qubits_idx in enumerate(gate_info['qubits_idx']):
                        if gate_info['param_sharing']:
                            param = gate_info['param'][depth_idx][0].item()
                        else:
                            param = gate_info['param'][depth_idx][idx][0].item()
                        if gate_idx == _gate_idx and depth_idx == _depth_idx and idx == _qubits_idx:
                            param += _param_shift_val
                        gate_func(param)(*[quleaf_state.Q[qubit_idx] for qubit_idx in qubits_idx])
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
            RY(-math.pi / 2)(_state.Q[qubit_idx])
        elif pauli_matrix == 'y':
            RX(math.pi / 2)(_state.Q[qubit_idx])
        elif pauli_matrix != 'z':
            raise ValueError("Cannot recognize the pauli words of the hamiltonian.")
    MeasureZ(*_state.Q.toListPair())
    counts = _state.commit(shots, fetchMeasure=True)['counts']
    filtered_counts = [(counts[key], [key[-idx - 1] for idx in observed_qubits]) for key in counts]
    return coeff * sum(
        (
            ((-1) ** key.count('1')) * val / shots
            for val, key in filtered_counts
        )
    )


class ExpecValOp(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
            ctx: paddle.autograd.PyLayerContext,
            state: 'pq.State',
            hamiltonian: 'pq.Hamiltonian',
            shots: paddle.Tensor,
            *parameters
    ) -> paddle.Tensor:
        r"""The forward function to compute the expectation value of the observable in the QuLeaf Backend.

        Args:
            ctx: To save some variables so that they can be used in the backward function.
            param: The parameters in the previous quantum gates.
            state: The quantum state to be measured.
            hamiltonian: The observable.
            shots: The number of measurement shots.
            *parameters: The parameters in the parameterized quantum circuit.

        Returns:
            The expectation value of the observable for the input state.
        """
        dtype = shots.dtype
        shots = int(shots.item())
        ctx.save_for_backward(*parameters)
        quleaf_state = copy.deepcopy(state.data)
        oper_history = state.oper_history
        ctx.quleaf_state = quleaf_state
        ctx.oper_history = oper_history
        ctx.hamiltonian = hamiltonian
        ctx.shots = shots
        ctx.dtype = dtype
        _state = copy.deepcopy(quleaf_state)
        acted_state = _act_gates_to_state(oper_history, _state, parameters)
        expec_val = 0
        for coeff, pauli_str in hamiltonian.pauli_str:
            _state = copy.deepcopy(acted_state)
            expec_val += _expec_val_on_quleaf(_state, coeff, pauli_str, shots)
        expec_val = paddle.to_tensor([expec_val], dtype=dtype)
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
        parameters = ctx.saved_tensor()
        dtype = ctx.dtype
        quleaf_state = ctx.quleaf_state
        oper_history = ctx.oper_history
        hamiltonian = ctx.hamiltonian
        shots = ctx.shots

        def expec_val_shift(
                _gate_idx: int, _depth_idx: int, _qubits_idx: int, _param_idx: int, param_shift_val: float
        ) -> float:
            _state = copy.deepcopy(quleaf_state)
            acted_state = _act_gates_to_state(
                oper_history, _state, _gate_idx, _depth_idx, _qubits_idx, _param_idx, param_shift_val)
            expec_val = 0
            for coeff, pauli_str in hamiltonian.pauli_str:
                _state = copy.deepcopy(acted_state)
                expec_val += _expec_val_on_quleaf(_state, coeff, pauli_str, shots)
            # expec_val = paddle.to_tensor([expec_val], dtype=expec_val_grad.dtype)
            return expec_val

        def general_param_shift(
                gate_name: str, _gate_idx: int, _depth_idx: int, _qubits_idx: int, _param_idx: int
        ) -> float:
            if gate_name in {'CRX', 'CRY', 'CRZ', 'CU'}:
                coeff_list = [
                    1 / (16 * math.pow(math.sin(math.pi / 8), 2)),
                    -1 / (16 * math.pow(math.sin(3 * math.pi / 8), 2)),
                    1 / (16 * math.pow(math.sin(5 * math.pi / 8), 2)),
                    - 1 / (16 * math.pow(math.sin(7 * math.pi / 8), 2)),
                ]
                shift_list = [math.pi / 2, 3 * math.pi / 2, 5 * math.pi / 2, 7 * math.pi / 2]
                grad_terms = map(
                    lambda coeff, shift_val: coeff * expec_val_shift(
                        _gate_idx, _depth_idx, _qubits_idx, _param_idx, shift_val
                    ),
                    coeff_list, shift_list
                )
                grad = sum(grad_terms)
            else:
                coeff1 = 0.5
                coeff2 = -0.5
                grad = (
                    coeff1 * expec_val_shift(_gate_idx, _depth_idx, _qubits_idx, _param_idx, math.pi / 2) +
                    coeff2 * expec_val_shift(_gate_idx, _depth_idx, _qubits_idx, _param_idx, 3 * math.pi / 2)
                )
            return grad

        params_idx = 0
        params_grad = [None]
        for gate_idx, gate_info in enumerate(oper_history):
            gate_name = gate_info['name']
            if gate_name not in parameterized_gates:
                continue
            if parameters[params_idx].stop_gradient is True:
                params_grad.append(None)
                params_idx += 1
                continue
            shape = parameters[params_idx].shape
            param_grad = np.zeros(shape=shape)
            for _depth_idx in range(shape[0]):
                for _qubits_idx in range(shape[1]):
                    for _param_idx in range(shape[2]):
                        grad = general_param_shift(gate_name, gate_idx, _depth_idx, _qubits_idx, _param_idx)
                        param_grad[_depth_idx, _qubits_idx, _param_idx] += grad
            params_grad.append(expec_val_grad * paddle.to_tensor(param_grad, dtype=dtype))
            params_idx += 1
        return params_grad
