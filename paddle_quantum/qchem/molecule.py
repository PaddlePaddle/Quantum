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
The module of the molecule.
"""

from typing import List, Tuple, Optional
import logging
import numpy as np
from openfermion import InteractionOperator, jordan_wigner
from ..hamiltonian import Hamiltonian
from .drivers import Driver, PySCFDriver
from .utils import orb2spinorb

__all__ = ["Molecule"]


class Molecule(object):
    def __init__(
        self,
        geometry: Optional[List[Tuple[str, List]]] = None,
        basis: Optional[str] = None,
        multiplicity: Optional[int] = None,
        charge: Optional[int] = None,
        mol_expr: Optional[str] = None,
        use_angstrom: bool = True,
        driver: Optional[Driver] = None
    ) -> None:
        r"""Construct molecule object from given information.

        Args:
            basis: basis set for computation chemistry.
            multiplicity: spin multiplicity of molecule, 2S+1.
            charge: charge of molecule.
            mol_expr: molecular expression, e.g. "CO2", "CH3COOH".
            geometry: atom symbol and their coordinate,
                e.g. ``[("H", [0.0, 0.0, 0.0]), ("H", [0.0, 0.0, 1.4])]`` . Default is None, if it's
                None, ``mol_expr`` should be specified, geometry will be download from internet.
            use_angstrom (bool): the length unit, default is True, if set to False,
                will use Atomic unit.
            driver: classical quantum chemistry calculator, default is None.
        """
        self.basis = "sto-3g" if basis is None else basis
        self.multiplicity = 1 if multiplicity is None else multiplicity
        self.charge = 0 if charge is None else charge
        self._unit = "Angstrom" if use_angstrom else "Bohr"
        if geometry is None and mol_expr is None:
            raise ValueError("One of the `mol_expr` and `geometry` shouldn't be None.")
        elif geometry is None:
            self.geometry = self.load_geometry(mol_expr)
        else:
            self.geometry = geometry
        
        if driver is None:
            raise ValueError("You need to specify a driver to perform classical quantum chemistry calculation.")

        driver.load_molecule(
            self.geometry,
            self.basis,
            self.multiplicity,
            self.charge,
            self.unit
        )
        self.driver = driver
        self._charges = driver.mol.atom_charges()
        self._coords = driver.mol.atom_coords()
        
        if mol_expr is None:
            mol_el: List[str] = driver.mol.elements
            mol_expr = ""
            for el in np.unique(mol_el).tolist():
                num_el = mol_el.count(el)
                mol_expr += f"{el:s}{num_el:d}" if num_el > 1 else f"{el:s}"
        self.mol_expr = mol_expr

    def build(self):
        r"""Use driver to calculate molecular integrals.
        """
        logging.info("\n#######################################\nMolecule\n#######################################")
        logging.info(f"{self.mol_expr:s}")
        logging.info("Geometry:")
        logging.info(
            "\n".join(
                f"{t[0]:s} {t[1][0]:.5f}, {t[1][1]:.5f}, {t[1][2]:.5f}"
                for t in self.geometry
            )
        )
        logging.info(f"Charge: {self.charge:d}")
        logging.info(f"Multiplicity: {self.multiplicity:d}")
        logging.info(f"Unit: {self.unit:s}")

        self.driver.run_scf()
        self._num_qubits = 2*self.driver.num_modes

    @property
    def atom_charges(self) -> np.ndarray:
        r"""Charges on each nuclei.
        """
        return self._charges

    @property
    def atom_coords(self) -> np.ndarray:
        r"""Atom's coordinate.
        """
        return self._coords

    @property
    def num_qubits(self) -> int:
        r"""Number of qubits used to encode the molecular quantum state on a quantum computer.
        """
        try:
            self._num_qubits
        except AttributeError as e:
            raise Exception("You need to run Molecule.build() method in order to access this attribute.").with_traceback(e.__traceback__)
        return self._num_qubits

    @property
    def unit(self):
        r"""Unit used for measuring the spatial distance of atoms in molecule.
        """
        return self._unit

    #TODO: develop load_geometry method that can automatically load from internet.
    def load_geometry(self, mol_expr: str):
        r"""load geometry of a molecule from internet.
        """
        pass

    def get_mo_integral(
        self,
        integral_type: str
    ) -> np.ndarray:
        r"""calculate integrals using chosen driver.

        Args:
            integral_type: type of integral, the name is different for different driver.

        Returns:
            The integrals.
        """
        if self.driver is None:
            raise ValueError("You need a driver to do classical mean field calculation to get molecular orbit.")

        if isinstance(self.driver, PySCFDriver):
            if integral_type.split("_")[0] == "int1e":
                return self.driver.get_onebody_tensor(integral_type)
            elif integral_type == "int2e" or integral_type.split("_")[0] == "int2e":
                return self.driver.get_twobody_tensor()
        else:
            raise NotImplementedError("Only PySCFDriver is currently implemented.")

    def get_molecular_hamiltonian(self) -> Hamiltonian:
        r"""returns the molecular hamiltonian for the given molecule.
        """
        hcore = self.get_mo_integral("int1e_nuc") + self.get_mo_integral("int1e_kin")
        eri = self.get_mo_integral("int2e")
        constant = self.driver.energy_nuc

        hcore_so, eri_so = orb2spinorb(self.driver.num_modes, hcore, eri)

        # set the values in the array that are lower than 1e-16 to zero.
        eps = np.finfo(hcore.dtype).eps
        hcore_so[abs(hcore_so) < eps] = 0.0
        eri_so[abs(eri_so) < eps] = 0.0

        h = InteractionOperator(constant, hcore_so, 0.5*eri_so)
        self._of_h = h # for testing purpose
        return Hamiltonian.from_qubit_operator(jordan_wigner(h))
