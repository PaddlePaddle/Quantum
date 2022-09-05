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
Slater determinant ansatz used in restricted Hartree Fock method.
"""

import logging
from typing import Union, cast
import numpy as np
from scipy.stats import unitary_group
import openfermion
import paddle
import paddle_quantum as pq

__all__ = ["RHFSlaterDeterminantModel"]


class GivensRotationBlock(pq.gate.Gate):
    r"""This is a two-qubit gate performs the Givens rotation.

    .. math:
        U(\theta)=e^{-i\frac{\theta}{2}(Y\otimes X-X\otimes Y)}

    Args:
        pindex, qindex qubits where Givens rotation gate acts on.
        theta: Givens rotation angle.
    """
    def __init__(
            self,
            pindex: int,
            qindex: int,
            theta: float) -> None:
        super().__init__(backend=pq.Backend.StateVector)

        cnot1 = pq.gate.CNOT([qindex, pindex])
        cnot2 = pq.gate.CNOT([pindex, qindex])
        ry_pos = pq.gate.RY(qindex, param=paddle.to_tensor(theta))
        ry_neg = pq.gate.RY(qindex, param=paddle.to_tensor(-theta))
        self.model = pq.ansatz.Sequential(cnot1, cnot2, ry_pos, cnot2, ry_neg, cnot1)

    def forward(self, state: pq.State) -> pq.State:
        return self.model(state)


class RHFSlaterDeterminantModel(pq.gate.Gate):
    r"""Slater determinant ansatz used in Restricted Hartree Fock calculation.
      
    Args:
        n_qubits: number of qubits used to encode the Slater determinant state.
        n_electrons: number of electrons inside the molecule.
        mo_coeff: parameters used to initialize Slater determinant state.
    """
    def __init__(
            self,
            n_qubits: int,
            n_electrons: int,
            mo_coeff: Union[np.array, None] = None) -> None:
        assert (n_qubits % 2 == 0 and n_electrons % 2 == 0), "Restricted Hartree Fock calculation\
             should only be carried out for molecules with even number of electrons"
        super().__init__(n_qubits, backend=pq.Backend.StateVector)
        if mo_coeff is not None:
            assert mo_coeff.shape == (
            n_electrons // 2, n_qubits // 2), f"The shape of `mo_coeff` should be {(n_electrons // 2, n_qubits // 2)},\
                but get {mo_coeff.shape}."
        else:
            logging.info("Will randomly initialize the circuit parameters.")
            U = unitary_group.rvs(n_qubits // 2)
            mo_coeff = U[:, :n_electrons // 2].T

        circuit_description = openfermion.slater_determinant_preparation_circuit(mo_coeff)
        models = []
        for parallel_ops in circuit_description:
            for op in parallel_ops:
                qi, qj, theta, _ = cast((int, int, float, float), op)
                givens_block_spinup = GivensRotationBlock(2 * qi, 2 * qj, theta)
                givens_block_spindown = GivensRotationBlock(2 * qi + 1, 2 * qj + 1, theta)
                models.append(givens_block_spinup)
                models.append(givens_block_spindown)
        self.model = pq.ansatz.Sequential(*models)

    def forward(self, state: pq.State) -> pq.State:
        return self.model(state)
