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
Ansatz for quantum chemistry
"""

from typing import Optional
import numpy as np
from openfermion import InteractionOperator, jordan_wigner
from openfermion.utils import is_hermitian
from ..ansatz import Circuit
from ..hamiltonian import Hamiltonian
from ..trotter import construct_trotter_circuit
from .utils import orb2spinorb

__all__ = ["HardwareEfficient", "UCC", "HartreeFock"]


class HardwareEfficient(Circuit):
    def __init__(
        self,
        num_qubits: int,
        depth: int,
        use_cz: bool = True,
        angles: Optional[np.ndarray] = None,
        rot_type: str = "ry",
    ) -> None:
        r"""
        Args:
            num_qubits (int): number of qubits.
            depth (int): the number of circuit units contained in the circuit. A circuit unit is defined as [Rot, Entangle].
            use_cz (bool): whether use CZ gate in the entangling layer, default is True, will use CNOT if this value is set to False.
            angles (np.ndarray): initial parameters of rotation gates inside the circuit, default is None and angles will
                be initialized randomly.
            rot_type (str): the form of rotation unit, default is None and will use "RY" gate, can be chosen among ["RY", "RX", "RZ", "U3"]
        """
        super().__init__(num_qubits)

        if rot_type not in {"rx", "ry", "rz", "u3"}:
            raise ValueError("`rot_type` needs to be chosen from `rx`, `ry`, `rz` and `u3`.")
        rot_type = rot_type.lower()

        if isinstance(angles, np.ndarray):
            if rot_type in {"rx", "ry"}:
                assert angles.shape == (num_qubits, depth+1)
            elif rot_type == "u3":
                assert angles.shape == (num_qubits, 3*(depth+1))
            else:
                raise ValueError("rot_type should be set to rx, ry or u3")

        entangle_qindex_pair = [(n, n+1) for n in range(num_qubits-1)]
        entangle_type = "cz" if use_cz else "cx"

        for _ in range(depth):
            getattr(self, rot_type)(qubits_idx="full")
            getattr(self, entangle_type)(qubits_idx=entangle_qindex_pair)
        getattr(self, rot_type)(qubits_idx="full")

        if isinstance(angles, np.ndarray):
            self.update_param(angles.flatten())

        self._rot_type = rot_type
        self._entangle_type = entangle_type

    @property
    def rot_type(self):
        r"""Type of rotation gate in the circuit.
        """
        return self._rot_type

    @property
    def entangle_type(self):
        r"""Type of entangling gate in the circuit.
        """
        return self._entangle_type


class UCC(Circuit):
    def __init__(
        self,
        num_qubits: int,
        ucc_order: str = "sd",
        single_ex_amps: Optional[np.ndarray] = None,
        double_ex_amps: Optional[np.ndarray] = None,
        **trotter_kwargs
    ) -> None:
        r"""
        UCCSD ansatz has a form

        .. math::

            e^{-i\hat{H}t},

        where :math:`\hat{H}` is a Hamiltonian operator with upto two-body terms.
        
        Args:
            num_qubits: number of qubits of the circuit.
            ucc_order: the order of UCC ansatz, default to "sd" which means UCCSD, allowed values are "s", "d", "sd".
            single_ex_amps: the amplitude for single excitation operator (or one-body operator), default is None and will be
                generated randomly.
            double_ex_amps: the amplitude for double excitation operator (or two-body operator), default is None and will be 
                generated randomly.
            trotter_kwargs: see ``construct_trotter_circuit`` function for available kwargs.
        """
        super().__init__(num_qubits)
        
        if ucc_order.lower() not in {"s", "d", "sd"}:
            raise ValueError("The allowed `ucc_order` are `s`, `d` and `sd`.")

        # NOTE: we assume the \hat{H} is symmetric w.r.t spin flip
        if isinstance(single_ex_amps, np.ndarray):
            assert single_ex_amps.shape == (num_qubits//2, num_qubits//2), ValueError("shape of `single_ex_amps` mismatches `num_qubits`.")
            np.testing.assert_array_almost_equal(
                single_ex_amps,
                single_ex_amps.T,
                err_msg="The single excitation coefficients provided will lead to a non-Hermitian operator for UCC."
            )
        elif ucc_order.lower() in {"s", "sd"}:
            single_ex_amps = np.random.randn(num_qubits//2, num_qubits//2)
            single_ex_amps += single_ex_amps.T
        else:
            single_ex_amps = np.zeros((num_qubits//2, num_qubits//2))

        if isinstance(double_ex_amps, np.ndarray):
            assert double_ex_amps.shape == (num_qubits//2, num_qubits//2, num_qubits//2, num_qubits//2), ValueError("shape of `double_ex_amps` mismatches `num_qubits`.")
            np.testing.assert_array_almost_equal(
                double_ex_amps, np.transpose(double_ex_amps, (3, 2, 1, 0)),
                err_msg="The double excitation coefficients provided will lead to a non-Hermitian operator for UCC."
            )
        elif ucc_order.lower() in {"d", "sd"}:
            double_ex_amps = np.random.randn(num_qubits//2, num_qubits//2, num_qubits//2, num_qubits//2)
            #NOTE: in order to make the resulting operator Hermitian, A_{prsq}=A_{qsrp}
            double_ex_amps += np.transpose(double_ex_amps, (3, 2, 1, 0))
        else:
            double_ex_amps = np.zeros((num_qubits//2, num_qubits//2, num_qubits//2, num_qubits//2))

        num_modes = num_qubits//2

        # NOTE: We assume the order of Fermion operator in JW transform to be updownupdown...
        single_ex_amps_so, double_ex_amps_so = orb2spinorb(num_modes, single_ex_amps, double_ex_amps)
        self._onebody_tensor = single_ex_amps_so
        self._twobody_tensor = double_ex_amps_so

        uccsd_h = InteractionOperator(0.0, single_ex_amps_so, double_ex_amps_so)
        assert is_hermitian(uccsd_h), "The operator on the exponent of the UCC operator isn't Hermitian."
        
        uccsd_QH = jordan_wigner(uccsd_h)
        pq_H = Hamiltonian.from_qubit_operator(uccsd_QH)
        #BUG: the np.ndarray may not be a parameter in circuit.
        construct_trotter_circuit(self, pq_H, **trotter_kwargs)

    @property
    def onebody_tensor(self):
        r"""
        :math:`T_{pq}` in UCCSD method.
        """
        return self._onebody_tensor

    @property
    def twobody_tensor(self):
        r"""
        :math:`V_{pqrs}` in UCCSD method.
        """
        return self._twobody_tensor


class HartreeFock(Circuit):
    def __init__(
        self,
        num_qubits: int,
        angles: Optional[np.ndarray] = None
    ) -> None:
        r"""
        Hartree-Fock (HF) ansatz.
        The HF ansatz will leads to a Slater determinant encoded by a ``num_qubits`` quantum state.

        Args:
            num_qubits: number of qubits used in the HF ansatz.
            angles: parameters of HF ansatz.

        Note:
            The ``num_qubits`` == ``num_modes`` in case of HF ansatz.
        """
        super().__init__(num_qubits)
        qubit_idx = []
        last_qindex = num_qubits-1
        for first_qindex, _ in enumerate(range(num_qubits-1)):
            qubit_idx.extend(
                [q0, q1] for q0, q1 in zip(
                    range(last_qindex-1, first_qindex-1, -1),
                    range(last_qindex, first_qindex, -1)
                )
            )

        #TODO: further reduce the number of Givens rotations.
        for qubits in qubit_idx:
            self.cnot(qubits[::-1])
            self.cry(qubits)
            self.cnot(qubits[::-1])
            self.rz(qubits[1])
        
        if isinstance(angles, np.ndarray):
            self.update_param(angles.flatten())
