# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Wave function module.
"""

#TODO: wave function, explor the symmetry
#TODO: how to distribute the wave function to cluster

from typing import Union, Optional
import math
import numpy as np
import paddle
from openfermion import InteractionOperator, jordan_wigner
from ..base import get_dtype
from ..backend import Backend
from ..gate.functional.base import simulation
from ..state import State
from ..hamiltonian import Hamiltonian
from ..intrinsic import _get_float_dtype

__all__ = ["WaveFunction"]


class WaveFunction(State):
    _fswap_matrix = paddle.to_tensor(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1]
        ],
        dtype=_get_float_dtype(get_dtype())
    )

    def __init__(
        self,
        data: Union[paddle.Tensor, np.ndarray],
        convention: str = "mixed",
        backend: Optional[Backend] = None,
        dtype: Optional[str] = None,
        override: Optional[bool] = False
    ) -> None:
        r"""
        Quantum state as a Fermionic Fock state :math:`|\Psi\rangle` . The number of qubits of the quantum state equals
        the number of spin-orbitals in :math:`|\Psi\rangle`, 
        each qubit corresponds to a spin orbital state in :math:`|\Psi\rangle` .

        .. math::

            |\Psi\rangle=\sum_{\alpha}C_{\alpha}|\alpha\rangle, C_{\alpha}\in\mathbb{C}. \\
            |\alpha\rangle=\otimes_{i=1}^N|n_i\rangle=\Pi_{i=1}^N \left(\hat{a}^{\dagger}_i\right)^{n_i}|\vec{0}\rangle, n_i\in\{0, 1\}.
        
        Args:
            data: a complex value vector contains coefficients of :math:`|\Psi\rangle` in computational basis.
            convention: labeling of spin orbitals, if set to ``mixed``, then spin up and spin down creation operators are intersecting
                each other, else, if set to ``separated``, spin up and spin down creation operators are belong to different sections.
            backend: paddle_quantum Backend on which to perform simulation, default is statevector backend.
            dtype: data type of wave function, default is ``complex64``.
            override: whether to override settings in wave function, default is False.
        """
        num_qubits = math.floor(math.log2(len(data)))
        assert 2**num_qubits == len(data), f"The input `data` is not a valid quantum state, `len(data)` should be 2**{num_qubits:d}, got {len(data):d}."
        assert num_qubits % 2 == 0, f"The input data is not a valid Fermionic Fock state (spin-orbital form), the `num_qubits` should be a multiple of 2, got {num_qubits:d}."
        
        super().__init__(data, num_qubits, backend, dtype, override)
        
        assert convention in {"mixed", "separated"}, "Only `mixed` and `separated` convention is valid."
        self.convention = convention
    
    def clone(self):
        r"""Return a copy of the wavefunction.

        Returns:
            A new state which is identical to this state.
        """
        return WaveFunction(self.data, self.convention, self.backend, self.dtype, override=True)

    def swap(self, p: int, q: int) -> None:
        r"""
        Switching p-th qubit and q-th qubit.

        Note:
            Since qubit represents fermion state, exchanging them will results in a minus sign.

        Args:
            p: index of the qubit being exchanged.
            q : index of another qubit being exchanged.
        """
        new_data = simulation(self, self._fswap_matrix, [p, q], self.num_qubits, self.backend)
        self.data = new_data

    def to_spin_mixed(self) -> None:
        r"""
        If the wavefunction is in convention "separated", convert it to a "mixed" convention state.
        """
        if self.convention == "mixed":
            print("Already in spin mixed mode, nothing changed.")
        else:
            num_orbs = self.num_qubits//2
            for head, cur in enumerate(range(num_orbs, self.num_qubits)):
                for p, q in zip(range(cur-1, 2*head, -1), range(cur, 2*head+1, -1)):
                    self.swap(p, q)
            self.convention = "mixed"
    
    def to_spin_separated(self) -> None:
        r"""
        If the wavefunction is in convention "mixed", convert it to a "separated" convention state.
        """
        if self.convention == "separated":
            print("Already in spin separated mode, nothing changed.")
        else:
            # 0 represents spin up mode, 1 represents spin down mode.
            spin_mode = [0, 1]*(self.num_qubits//2)
            while True:
                num_switch = 0
                for p in range(self.num_qubits-1):
                    if spin_mode[p] == 1 and spin_mode[p+1] == 0:
                        self.swap(p, p+1)
                        num_switch += 1
                        spin1, spin2 = spin_mode[p], spin_mode[p+1]
                        spin_mode[p] = spin2
                        spin_mode[p+1] = spin1
                if num_switch == 0:
                    break
            self.convention = "separated"
            
    @classmethod
    def slater_determinant_state(
        cls,
        num_qubits: int,
        num_elec: int,
        mz: int,
        backend: Optional[Backend] = None,
        dtype: Optional[str] = None,
    ):
        r"""
        Construct a single Slater determinant state whose length is ``num_qubits`` .
        The number of "1" in the Slater determinant state equals `num_elec`, the difference
        between number of spin up electrons and the number of spin down electrons is ``mz`` .
        The prepared Slater determinant is in mixed spin orbital mode, which is "updownupdown...".

        Args:
            num_qubits: number of qubits used in preparing Slater determinant state.
            num_elec: number of electrons in Slater determinant state.
            mz: :math:`n_{\uparrow}-n_{\downarrow}` .
        
        Returns:
            WaveFunction.
        """
        assert num_qubits >= num_elec, "Need more qubits to hold the Slater determinant state."
        num_alpha = math.floor(0.5*(num_elec + mz))
        num_beta = math.floor(0.5*(num_elec - mz))
        assert num_alpha + num_beta == num_elec
        
        bstr_list = ["1", "1"]*min(num_alpha, num_beta)
        if mz > 0:
            bstr_list.extend(["1", "0"]*abs(mz))
        if mz < 0:
            bstr_list.extend(["0", "1"]*abs(mz))
        bstr_list.extend(["0"]*(num_qubits-2*max(num_alpha, num_beta)))
        bstr = "".join(bstr_list)
        
        psi = np.zeros(2**num_qubits, dtype=np.complex64)
        psi[int(bstr, 2)] = 1.0+0.0j
        return cls(psi, dtype=dtype, backend=backend)

    @classmethod
    def zero_state(cls, num_qubits: int, backend: Optional[Backend] = None, dtype: Optional[str] = None):
        r"""
        Construct a zero state, :math:`|0000....\rangle` .
        """
        psi = np.zeros(2**num_qubits, dtype=np.complex64)
        psi[0] = 1.0+0.0j
        return cls(psi, dtype=dtype, backend=backend)

    def num_elec(self, shots: int = 0) -> float:
        r"""Calculate the total number of electrons in the wave function.

        .. math::

            \langle\Psi|\sum_{i\sigma}\hat{a}_{i\sigma}^{\dagger}\hat{a}_{i\sigma}|\Psi\rangle.

        """
        num_qubits = self.num_qubits
        number_operator = Hamiltonian(
            [(0.5*num_qubits, "I")].extend(
                (-0.5, f"Z{i:d}") for i in range(num_qubits)
            )
        )
        return self.expec_val(number_operator, shots)

    def total_SpinZ(self, shots: int = 0) -> float:
        r"""Calculate the total spin Z component of the wave function.

        .. math::

            \langle\Psi|\sum_{i}\hat{S}_z|\Psi\rangle \\
            \hat{S}_z = 0.5*\sum_{p}(\hat{n}_{p\alpha}-\hat{n}_{p\beta}) \\
            \alpha\equiv\uparrow, \beta\equiv\downarrow, \hat{n}_{p\sigma}=\hat{a}^{\dagger}_{p\sigma}\hat{a}_{p\sigma}

        """
        num_qubits = self.num_qubits
        sz_operator = Hamiltonian(
            [
                (((-1)**(k+1))*0.25, f"Z{k:d}") for k in range(num_qubits)
            ]
        )
        return self.expec_val(sz_operator, shots)

    def total_Spin2(self, shots: int = 0) -> float:
        r"""Calculate the expectation value of :math:`\hat{S}^2` operator on the wave function.

        .. math::

            \langle\Psi|\hat{S}_+\hat{S}_- +\hat{S}_z(\hat{S}_z-1)|\Psi\rangle \\
            \hat{S}_+ = \sum_{p}\hat{a}_{p\alpha}^{\dagger}\hat{a}_{p\beta} \\
            \hat{S}_- = \sum_{p}\hat{a}_{p\beta}^{\dagger}\hat{a}_{p\alpha} \\
            \hat{S}_z = 0.5*\sum_{p}(\hat{n}_{p\alpha}-\hat{n}_{p\beta}) \\
            \alpha\equiv\uparrow, \beta\equiv\downarrow, \hat{n}_{p\sigma}=\hat{a}^{\dagger}_{p\sigma}\hat{a}_{p\sigma}

        """
        # construct \hat{S}^2
        num_modes = self.num_qubits//2
        T = np.zeros([self.num_qubits, self.num_qubits])
        V = np.zeros([self.num_qubits, self.num_qubits, self.num_qubits, self.num_qubits])
        for p in range(num_modes):
            T[2*p, 2*p] = 0.75
            T[2*p+1, 2*p+1] = 0.75
            for q in range(num_modes):
                V[2*p, 2*q+1, 2*p+1, 2*q] = -1.0
                V[2*p, 2*q, 2*q, 2*p] = 0.25
                V[2*p+1, 2*q+1, 2*q+1, 2*p+1] = 0.25
                V[2*p, 2*q+1, 2*q+1, 2*p] = -0.5
        s2 = InteractionOperator(0.0, T, V)
        s2_qubit = jordan_wigner(s2)
        s2_h = Hamiltonian.from_qubit_operator(s2_qubit)
        # calculate expect value
        return self.expec_val(s2_h, shots)
