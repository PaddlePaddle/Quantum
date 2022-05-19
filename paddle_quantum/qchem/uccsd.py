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
Unitary coupled cluster with singles and doubles for molecular ground state calculation.
"""

from typing import Union
import numpy as np
import openfermion
from openfermion import FermionOperator
from openfermion.transforms import normal_ordered
import paddle_quantum as pq
from paddle_quantum.trotter import construct_trotter_circuit
from .qchem import qubitOperator_to_Hamiltonian


def _get_single_excitation_operator(p: int, q: int) -> FermionOperator:
    r"""
    Args:
        p: index of the unoccupied orbital.
        q: index of the occupied orbital.
    
    Returns:
        FermionOperator,
        .. math:
            \hat{E}_{pq}\equiv\sum_{\sigma}\hat{c}^{\dagger}_{p\sigma}\hat{c}_{q\sigma}
    """

    return FermionOperator(f"{2 * p}^ {2 * q}") + FermionOperator(f"{2 * p + 1}^ {2 * q + 1}")


def _get_double_excitation_operator(p: int, q: int, r: int, s: int) -> FermionOperator:
    r"""
    Args:
        p, r: index of the unoccupied orbital.
        q, s: index of the occupied orbital.
    
    Returns:
        FermionOperator,
        .. math:
            \hat{e}_{pqrs}=\sum_{\sigma\tau}\hat{c}^{\dagger}_{p\sigma}\hat{c}^{\dagger}_{r\tau}\hat{c}_{s\tau}\hat{c}_{q\sigma}
    """

    if p == r or q == s:
        e2 = FermionOperator(f"{2 * p}^ {2 * r + 1}^ {2 * s + 1} {2 * q}") + \
             FermionOperator(f"{2 * p + 1}^ {2 * r}^ {2 * s} {2 * q + 1}")
    else:
        e2 = FermionOperator(f"{2 * p}^ {2 * r}^ {2 * s} {2 * q}") + \
             FermionOperator(f"{2 * p}^ {2 * r + 1}^ {2 * s + 1} {2 * q}") + \
             FermionOperator(f"{2 * p + 1}^ {2 * r}^ {2 * s} {2 * q + 1}") + \
             FermionOperator(f"{2 * p + 1}^ {2 * r + 1}^ {2 * s + 1} {2 * q + 1}")
    return normal_ordered(e2)


def _get_antiH_single_excitation_operator(p: int, q: int) -> FermionOperator:
    r"""
    Args:
        p: index of the unoccupied orbital.
        q: index of the occupied orbital.
    
    Returns:
        FermionOperator, 
        .. math:
            \hat{E}_{pq}-\hat{E}_{qp}
    """

    e1_pq = _get_single_excitation_operator(p, q)
    e1_qp = _get_single_excitation_operator(q, p)
    return e1_pq - e1_qp


def _get_antiH_double_excitation_operator(p: int, q: int, r: int, s: int) -> FermionOperator:
    r"""
    Args:
        p, r: index of the unoccupied orbital.
        q, s: index of the occupied orbital.

    Returns:
        FermionOperator,
        .. math:
            \hat{e}_{pqrs}-\hat{e}_{srqp}
    """

    e2_pqrs = _get_double_excitation_operator(p, q, r, s)
    e2_srqp = _get_double_excitation_operator(s, r, q, p)
    return e2_pqrs - e2_srqp


class UCCSDModel(pq.gate.Gate):
    r"""Unitary Coupled Cluster ansatz for quantum chemistry calculation. 
    
    .. note::
        UCCSD model typically results in REALLY deep quantum circuit. Training UCCSD ansatz for molecules beyond H2 is time consuming and demands better hardware.

    .. math::

        \begin{align}
            U(\theta)&=e^{\hat{T}-\hat{T}^{\dagger}}\\
            \hat{T}&=\hat{T}_1+\hat{T}_2\\
            \hat{T}_1&=\sum_{a\in{\text{virt}}}\sum_{i\in\text{occ}}t_{ai}\sum_{\sigma}\hat{c}^{\dagger}_{a\sigma}\hat{c}_{i\sigma}-h.c.\\
            \hat{T}_2&=\frac{1}{2}\sum_{a,b\in\text{virt}}\sum_{i,j\in\text{occ}}t_{aibj}\sum_{\sigma\tau}\hat{c}^{\dagger}_{a\sigma}\hat{c}^{\dagger}_{b\tau}\hat{c}_{j\tau}\hat{c}_{i\sigma}-h.c. 
        \end{align}

    Args:
        n_qubits: number of qubits used to represent the quantum system.
        n_electrons: number of electrons inside the system.
        n_trotter_steps: number of Trotter steps required to build the UCCSD circuit.
        single_excitation_amplitude: :math:`t_{ai}` in the definition of :math:`\hat{T}_1`.
        double_excitation_amplitude: :math:`t_{aibj}` in the definition of :math:`\hat{T}_2`.
    """
    def __init__(
            self,
            n_qubits: int,
            n_electrons: int,
            n_trotter_steps: int,
            single_excitation_amplitude: Union[np.array, None] = None,
            double_excitation_amplitude: Union[np.array, None] = None) -> None:
        super().__init__(n_qubits, backend=pq.Backend.StateVector)

        n_occupied_mos = n_electrons // 2
        n_mos = n_qubits // 2
        occupied_orbitals = list(range(n_occupied_mos))
        virtual_orbitals = list(range(n_occupied_mos, n_mos))

        if single_excitation_amplitude is None:
            single_excitation_amplitude = np.random.randn(n_qubits, n_qubits)
        if double_excitation_amplitude is None:
            double_excitation_amplitude = np.random.randn(n_qubits, n_qubits, n_qubits, n_qubits)

        ucc_generator = FermionOperator()
        for a in virtual_orbitals:
            for i in occupied_orbitals:
                ucc_generator += single_excitation_amplitude[a, i] * _get_antiH_single_excitation_operator(a, i)
                for b in virtual_orbitals:
                    for j in occupied_orbitals:
                        ucc_generator += double_excitation_amplitude[
                                             a, i, b, j] * _get_antiH_double_excitation_operator(a, i, b, j)

        ucc_hamiltonian = -1j * ucc_generator
        ucc_hamilton_qubit = openfermion.jordan_wigner(ucc_hamiltonian)
        ucc_pqH = qubitOperator_to_Hamiltonian(ucc_hamilton_qubit)
        self.uccsd_hamilton = ucc_pqH

        circuit = pq.ansatz.Circuit()
        tau = 1.0 / n_trotter_steps
        construct_trotter_circuit(circuit, ucc_pqH, tau=tau, steps=n_trotter_steps)
        self.model = circuit

    def forward(self, state: pq.State) -> pq.State:
        return self.model(state)
