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
Loss functions for quantum chemistry calculation.
"""

import paddle
import numpy as np
import paddle_quantum as pq

from .qchem import get_molecular_data, spin_hamiltonian
from .density_matrix import get_spinorb_onebody_dm

__all__ = ["MolEnergyLoss", "RHFEnergyLoss"]


class MolEnergyLoss(pq.loss.ExpecVal):
    r"""Loss function for molecular ground state calculation. 

    Args:
        geometry: e.g. "H 0.0 0.0 0.0; H 0.0 0.0 0.74".
        basis: chemical basis, e.g. "sto-3g".
        multiplicity: spin multiplicity.
        charge: charge of the molecule.
    """

    def __init__(
            self,
            geometry: str,
            basis: str,
            multiplicity: int = 1,
            charge: int = 0) -> None:
        geometry_internal = []
        for atom in geometry.split(";"):
            atom = atom.strip()
            atom_list = atom.split(" ")
            atom_symbol = atom_list[0]
            atom_coord = atom_list[1:]
            geometry_internal.append((atom_symbol, [float(x) for x in atom_coord]))

        mol = get_molecular_data(geometry_internal, charge, multiplicity, basis)
        mol_H = spin_hamiltonian(mol)
        super().__init__(mol_H)


class RHFEnergyLoss(pq.Operator):
    r"""Loss function for Restricted Hartree Fock calculation.
        NOTE: This function needs PySCF be installed!

    Args:
        geometry: e.g. "H 0.0 0.0 0.0; H 0.0 0.0 0.74".
        basis: chemical basis, e.g. "sto-3g".
        multiplicity: spin multiplicity.
        charge: charge of the molecule.

    Raises:
        ModuleNotFoundError: `hartree fock` method needs pyscf being installed, please run `pip install -U pyscf`. 
    """
    
    def __init__(
            self,
            geometry: str,
            basis: str,
            multiplicity: int = 1,
            charge: int = 0
    ) -> None:
        super().__init__(backend=pq.Backend.StateVector)

        try:
            import pyscf
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "`hartree fock` method needs pyscf being installed, please run `pip install -U pyscf`.")

        from pyscf.lo import lowdin

        mol = pyscf.gto.Mole(atom=geometry, basis=basis, multiplicity=multiplicity, charge=charge)
        mol.build()
        ovlp = mol.intor_symmetric("int1e_ovlp")
        kin = mol.intor_symmetric("int1e_kin")
        vext = mol.intor_symmetric("int1e_nuc")
        vint = np.transpose(mol.intor("int2e"), (0, 2, 3, 1))
        V = lowdin(ovlp)
        onebody_tensor, twobody_tensor, V_tensor = map(paddle.to_tensor, [kin + vext, 0.5 * vint, V])

        self.energy_nuc = mol.energy_nuc()
        self.onebody = onebody_tensor
        self.twobody = twobody_tensor
        self._V = V_tensor

    def forward(self, state: pq.State) -> paddle.Tensor:
        state_tensor = state.data
        rdm_spinup, _ = get_spinorb_onebody_dm(state.num_qubits, state_tensor)
        rdm = 2 * (self._V @ rdm_spinup @ self._V)
        rhf_energy = self.energy_nuc + paddle.einsum("pq,qp->", self.onebody, rdm) + \
                     paddle.einsum("pqrs,qp,sr->", self.twobody, rdm, rdm) - \
                     0.5 * paddle.einsum("pqrs,sp,qr->", self.twobody, rdm, rdm)
        return rhf_energy
