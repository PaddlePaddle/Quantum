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
The drivers of the classical quantum chemistry calculation.
"""

from typing import List, Tuple, Optional
import sys
import logging
import numpy as np

__all__ = ["Driver", "PySCFDriver"]


class Driver(object):
    def run_scf(self):
        raise NotImplementedError

    @property
    def num_modes(self):
        raise NotImplementedError

    @property
    def energy_nuc(self):
        raise NotImplementedError

    @property
    def mo_coeff(self):
        raise NotImplementedError

    def load_molecule(self):
        raise NotImplementedError

    def get_onebody_tensor(self):
        raise NotImplementedError

    def get_twobody_tensor(self):
        raise NotImplementedError


class PySCFDriver(Driver):
    r"""Use pyscf to perform classical quantum chemistry calculation.
    """
    def __init__(self) -> None:
        if sys.platform not in ["linux", "darwin"]:
            raise ModuleNotFoundError("NOTE your operating system is Windows, PySCF doesn't support Windows yet, sorry....")

        try:
            import pyscf
        except ImportError as e:
            raise ModuleNotFoundError(
                "PySCF doesn't find on the current path, you can install it by `pip install pyscf`."
            ) from e
        self.mol = pyscf.gto.Mole()

    def load_molecule(
        self,
        atom: List[Tuple[str, List[float]]],
        basis: str,
        multiplicity: int,
        charge: int,
        unit: str
    ) -> None:
        r"""construct a pyscf molecule from the given information.

        Args:
            atom: atom symbol and their coordinate, same format
                as in `Molecule`.
            basis: basis set.
            multiplicity: spin multiplicity 2S+1.
            charge: charge of the molecule.
            unit: `Angstrom` or `Bohr`.
        """
        pyscf_atom = "; ".join(
            f"{symbol:s} {x:.8f} {y:.8f} {z:.8f}"
            for symbol, (x, y, z) in atom
        )
        self.mol.build(
            atom=pyscf_atom,
            basis=basis,
            spin=multiplicity-1,
            charge=charge,
            unit=unit
        )
        self._energy_nuc = self.mol.energy_nuc()

    def run_scf(self):
        r"""perform RHF calculation on the molecule.
        """
        from pyscf.scf import RHF

        logging.info("\n#######################################\nSCF Calculation (Classical)\n#######################################")
        logging.info(f"Basis: {self.mol.basis:s}")

        mf = RHF(self.mol)
        mf.kernel()
        self.scf = mf
        self.scf_e_tot = mf.e_tot
        self._mo_coeff = mf.mo_coeff
        self._num_modes = mf.mo_coeff.shape[1]

        logging.info(f"SCF energy: {self.scf_e_tot:.5f}.")
    
    @property
    def energy_nuc(self):
        r"""Potential energy of nuclears.
        """
        return self._energy_nuc
    
    @property
    def mo_coeff(self):
        r"""Transformation matrix between atomic orbitals and molecular orbitals.
        """
        return self._mo_coeff

    @property
    def num_modes(self):
        r"""Number of molecular orbitals.
        """
        return self._num_modes

    def get_onebody_tensor(self, integral_type: Optional[str] = None) -> np.ndarray:
        r"""
        :math:`T[p,q] = \int\phi_p^*(x) f(x) \phi_q(x)dx`

        Args:
            integral_type (str): the type of integral, e.g. "int1e_ovlp", "int1e_kin", etc.,
                see https://github.com/pyscf/pyscf/blob/master/pyscf/gto/moleintor.py for details.

        Returns:
            np.ndarray.
        """
        try:
            mo_coeff = self.mo_coeff
        except AttributeError:
            self.run_scf()
            mo_coeff = self.mo_coeff

        if integral_type is None:
            raise ValueError("An integral type needs to be specified.")

        with self.mol.with_common_orig((0.0, 0.0, 0.0)):
            ao_integral = self.mol.intor(integral_type)
        return np.einsum("...ij,ik,jl->...kl", ao_integral, mo_coeff, mo_coeff)

    def get_twobody_tensor(self) -> np.ndarray:
        r"""
        :math:`V[p,r,s,q] = \int\phi^*_p(x)\phi^*_r(x')(1/|x-x'|)\phi_s(x')\phi_q(x)dxdx'=(pq|rs)` in pyscf.
        """
        from pyscf import ao2mo

        try:
            mo_coeff = self.mo_coeff
        except AttributeError:
            self.run_scf()
            mo_coeff = self.mo_coeff        

        eri = ao2mo.kernel(self.mol, mo_coeff)
        eri = ao2mo.restore(1, eri, mo_coeff.shape[1])
        return np.transpose(eri, (0, 2, 3, 1))
