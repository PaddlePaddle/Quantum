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
The function for quantum chemistry.
"""

import os
import re
import string
from typing import Optional
import numpy as np
import openfermion
from openfermion import MolecularData, transforms
from openfermion.ops import general_basis_change
from paddle_quantum.hamiltonian import Hamiltonian


__all__ = [
    "qubitOperator_to_Hamiltonian",
    "geometry",
    "get_molecular_data",
    "active_space",
    "fermionic_hamiltonian",
    "spin_hamiltonian"
]


def qubitOperator_to_Hamiltonian(spin_h: openfermion.ops.operators.qubit_operator.QubitOperator, tol: Optional[float] = 1e-8) -> Hamiltonian:
    r"""Transfer openfermion form to Paddle Quantum Hamiltonian form.

    Args:
        spin_h: Hamiltonian in openfermion form.
        tol: Value less than tol will be ignored. Defaults to 1e-8.

    Returns:
        Hamiltonian in Paddle Quantum form.
    """
    terms = spin_h.__str__().split('+\n')
    spin_h.compress(abs_tol=tol)
    pauli_str = []
    for term in terms:
        decomposed_term = re.match(r"(.*) \[(.*)\].*", term).groups()
        if decomposed_term[1] == '':
            try:
                pauli_str.append([float(decomposed_term[0]), 'I'])
            except ValueError:
                if complex(decomposed_term[0]).real > 0:
                    pauli_str.append([abs(complex(decomposed_term[0])), 'I'])
                else:
                    pauli_str.append([-abs(complex(decomposed_term[0])), 'I'])
        else:
            term_str = ', '.join(re.split(r' ', decomposed_term[1]))
            try:
                pauli_str.append([float(decomposed_term[0]), term_str])
            except ValueError:
                if complex(decomposed_term[0]).real > 0:
                    pauli_str.append([abs(complex(decomposed_term[0])), term_str])
                else:
                    pauli_str.append([-abs(complex(decomposed_term[0])), term_str])
    return Hamiltonian(pauli_str)


def _geo_str(geometry: list) -> str:
    r"""String of molecular geometry information

    Args:
        geometry: contains the geometry of the molecule, for example, the H2 molecule
            [['H', [-1.68666, 1.79811, 0.0]], ['H', [-1.12017, 1.37343, 0.0]]]

    Returns:
        String of molecular geometry information
    """
    geo_str = ''
    for item in geometry:
        atom_symbol = item[0]
        position = item[1]
        line = '{} {} {} {}'.format(atom_symbol,
                                    position[0],
                                    position[1],
                                    position[2])
        if len(geo_str) > 0:
            geo_str += '\n'
        geo_str += line
    geo_str += '\nsymmetry c1'
    return geo_str


def _run_psi4(
        molecule: MolecularData,
        charge: int,
        multiplicity: int,
        method: str,
        basis: str,
        if_print: bool,
        if_save: bool
) -> None:
    r"""The necessary information to calculate molecules, including one-body integrations and two-body integrations, as well as the energy of ground states by scf and fci methods.

    Args:
        molecule: Class containing all information about a molecule.
        charge: Charge of the molecule.
        multiplicity: The multiplicity of the molecule.
        method: Method used to calculate the ground state energy, including 'scf' and 'fci'.
        basis: Most common used basis are ‘sto-3g’, ‘6-31g’. For more basis options, please refer to
            https://psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement. 
        if_print: If or not the base state energy of the molecule calculated by the selected method should be printed.
        if_save: If the molecular information needs to be stored as an .hdf5 file.
    """

    import psi4
    psi4.set_memory('500 MB')
    psi4.set_options({'soscf': 'false',
                      'scf_type': 'pk'})
    geo = molecule.geometry
    mol = psi4.geometry(_geo_str(geo))
    mol.set_multiplicity(multiplicity)
    mol.set_molecular_charge(charge)

    if molecule.multiplicity == 1:
        psi4.set_options({'reference': 'rhf',
                          'guess': 'sad'})
    else:
        psi4.set_options({'reference': 'rohf',
                          'guess': 'gwh'})

    # HF calculation
    hf_energy, hf_wfn = psi4.energy('scf/' + basis, molecule=mol, return_wfn='on')
    # Get orbitals and Fock matrix.
    molecule.hf_energy = hf_energy
    molecule.nuclear_repulsion = mol.nuclear_repulsion_energy()
    molecule.canonical_orbitals = np.asarray(hf_wfn.Ca())
    molecule.overlap_integrals = np.asarray(hf_wfn.S())
    molecule.n_orbitals = molecule.canonical_orbitals.shape[0]
    molecule.n_qubits = 2 * molecule.n_orbitals
    molecule.orbital_energies = np.asarray(hf_wfn.epsilon_a())
    molecule.fock_matrix = np.asarray(hf_wfn.Fa())

    # Get integrals using MintsHelper.
    mints = psi4.core.MintsHelper(hf_wfn.basisset())

    molecule.one_body_integrals = general_basis_change(
        np.asarray(mints.ao_kinetic()), molecule.canonical_orbitals, (1, 0))
    molecule.one_body_integrals += general_basis_change(
        np.asarray(mints.ao_potential()), molecule.canonical_orbitals, (1, 0))
    two_body_integrals = np.asarray(mints.ao_eri())
    two_body_integrals.reshape((molecule.n_orbitals, molecule.n_orbitals,
                                molecule.n_orbitals, molecule.n_orbitals))
    two_body_integrals = np.einsum('psqr', two_body_integrals)
    two_body_integrals = general_basis_change(
        two_body_integrals, molecule.canonical_orbitals, (1, 1, 0, 0))
    molecule.two_body_integrals = two_body_integrals

    # FCI calculation
    psi4.set_options({'qc_module': 'detci'})
    fci_energy, fci_wfn = psi4.energy('fci/' + basis, molecule=mol, return_wfn='on')
    molecule.fci_energy = fci_energy

    if if_save is True:
        molecule.save()

    if if_print is True:
        if method == 'scf':
            print('Hartree-Fock energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, hf_energy))

        elif method == 'fci':
            print('FCI energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, fci_energy))
        elif method == '':
            print('Calculation is done')


def geometry(structure: Optional[str] = None, file: Optional[str] = None) -> str:
    r"""Read molecular geometry information.

    Args:
        structure: Including molecular geometry information in string, take H2 as an example 
            ``[['H', [-1.68666, 1.79811, 0.0]], ['H', [-1.12017, 1.37343, 0.0]]]``. Defaults to None.
        file: The path of .xyz file. Defaults to None.

    Raises:
        AssertionError:  The two optional input cannot be None simultaneously.

    Returns:
        Molecular geometry information.
    """
    if structure is None and file is None:
        raise AssertionError('Input must be structure or .xyz file')
    elif file is None:
        shape = np.array(structure).shape
        assert shape[1] == 2, 'The shape of structure must be (n, 2)'
        for i in range(shape[0]):
            assert isinstance(np.array(structure)[:, 0][i], str), 'The first position must be element symbol'
            assert len(np.array(structure)[:, 1][i]) == 3, 'The second position represents coordinate ' \
                                                           'of particle: x, y, z'
        geo = structure
    elif structure is None:
        assert file[-4:] == '.xyz', 'The file is supposed to be .xyz'
        geo = []
        with open(file) as f:
            for line in f.readlines()[2:]:
                one_geo = []

                symbol, x, y, z = line.split()
                one_geo.append(symbol)
                one_geo.append([float(x), float(y), float(z)])
                geo.append(one_geo)

    return geo


def get_molecular_data(
        geometry: str,
        charge: int=0,
        multiplicity: int=1,
        basis: str='sto-3g',
        method: str='scf',
        if_save: bool=True,
        if_print: bool=True,
        name: str="",
        file_path: str="."
) -> MolecularData:
    r"""Calculate necessary values of molecule, including one-body integrations, two-body integrations, and the ground state energy calculated by a chosen method

    Args:
        geometry: Molecular geometry information.
        charge: Molecular charge. Defaults to 0.
        multiplicity: Molecular multiplicity. Defaults to 1.
        basis: Most common used basis are ‘sto-3g’, ‘6-31g’. For more basis options, please refer to
            https://psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement. Defaults to 'sto-3g'.
        method: Method to calculate ground state energy, including ``scf``, ``fci``. Defaults to ``scf``.
        if_save: If need to save molecule information as .hdf5 file. Defaults to True.
        if_print: If need to print ground state energy calculated by chosen method. Defaults to True.
        name: The name of the file to save. Defaults to "".
        file_path: The path of the file to save. Defaults to ".".

    Returns:
        A class contains information of the molecule.
    """
    methods = ['scf', 'fci']
    assert method in methods, 'We provide 2 methods: scf and fci'

    if if_save is True:
        path = file_path + '/qchem_data/'
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        if name == "":
            elements = np.array(geometry)[:, 0]
            symbol, counts = np.unique(elements, return_counts=True)
            filename = path
            for i in range(len(symbol)):
                filename += symbol[i]+str(counts[i])+'_'
            filename += basis + '_' + method + '.hdf5'
        else:
            if name[-5:] == '.hdf5':
                filename = name
            else:
                filename = name + '.hdf5'

    molecule = MolecularData(
        geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        filename=filename
    )

    _run_psi4(
        molecule,
        charge,
        multiplicity,
        method,
        basis,
        if_print,
        if_save
    )

    return molecule


def active_space(
        electrons: int,
        orbitals: int,
        multiplicity: int=1,
        active_electrons: int=None,
        active_orbitals: int=None
) -> tuple:
    r"""Calculate active space by nominating the number of active electrons and active orbitals.

    Args:
        electrons: Number of total electrons.
        orbitals: Number of total orbitals.
        multiplicity: Spin multiplicity. Defaults to 1.
        active_electrons: Number of active electrons, default to the case that all electrons are active.
        active_orbitals: Number of active orbitals, default to the case that all orbitals are active.

    Returns:
        Index for core orbitals and active orbitals.
    """
    assert type(electrons) == int and electrons > 0, 'Number of electrons must be positive integer.'
    assert type(orbitals) == int and orbitals > 0, 'Number of orbitals must be positive integer.'
    assert type(multiplicity) == int and multiplicity >= 0, 'The multiplicity must be non-negative integer.'

    if active_electrons is None:
        no_core_orbitals = 0
        core_orbitals = []
    else:
        assert type(active_electrons) == int, 'The number of active electrons must be integer.'
        assert active_electrons > 0, 'The number of active electrons must be greater than 0.'
        assert electrons >= active_electrons, 'The number of electrons should more than or equal ' \
                                              'to the number of active electrons.'
        assert active_electrons >= multiplicity - 1, 'The number of active electrons should greater than ' \
                                                     'or equal to multiplicity - 1.'
        assert multiplicity % 2 != active_electrons % 2, 'Mulitiplicity and active electrons should be one odd ' \
                                                         'and the other one even.'

        no_core_orbitals = (electrons - active_electrons) // 2
        core_orbitals = list(np.arange(0, no_core_orbitals))

    if active_orbitals is None:
        active_orbitals = list(np.arange(no_core_orbitals, orbitals))
    else:
        assert type(active_orbitals) == int, 'The number of active orbitals must be integer.'
        assert active_orbitals > 0, 'The number of active orbitals must be greater than 0.'
        assert no_core_orbitals + active_orbitals <= orbitals, 'The summation of core orbitals and active ' \
                                                               'orbitals should be smaller than orbitals.'
        assert no_core_orbitals + active_orbitals > (electrons + multiplicity - 1) / 2, \
            'The summation of core orbitals and active orbitals should be greater than ' \
            '(electrons + multiplicity - 1)/2.'

        active_orbitals = list(np.arange(no_core_orbitals, no_core_orbitals + active_orbitals))

    return core_orbitals, active_orbitals


def fermionic_hamiltonian(
        molecule: MolecularData,
        filename: str=None,
        multiplicity: int=1,
        active_electrons: int=None,
        active_orbitals: int=None
) -> openfermion.ops.operators.qubit_operator.QubitOperator:
    r"""Calculate the fermionic hamiltonian of the given molecule.

    Args:
        molecule: A class contains information of the molecule.
        filename: Path of .hdf5 file of molecule. Defaults to None.
        multiplicity: Spin multiplicity. Defaults to 1.
        active_electrons: Number of active electrons, default to the case that all electrons are active.
        active_orbitals: Number of active orbitals, default to the case that all orbitals are active.

    Returns:
        Hamiltonian in openfermion form.
    """
    if molecule is None:
        assert type(filename) == str, 'Please provide the path of .hdf5 file.'
        assert filename[-5:] == '.hdf5', 'The filename is supposed to be .hdf5 file'
        molecule = MolecularData(filename=filename)
    core_orbitals, active_orbitals = active_space(
        molecule.n_electrons,
        molecule.n_orbitals,
        multiplicity,
        active_electrons,
        active_orbitals)
    terms_molecular_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=core_orbitals, active_indices=active_orbitals
    )
    fermionic_hamiltonian = openfermion.transforms.get_fermion_operator(terms_molecular_hamiltonian)

    return fermionic_hamiltonian


def spin_hamiltonian(
        molecule: openfermion.ops.operators.qubit_operator.QubitOperator ,
        filename: str=None,
        multiplicity: int=1,
        mapping_method: str='jordan_wigner',
        active_electrons: int=None,
        active_orbitals: int=None
) -> Hamiltonian:
    r"""Generate Hamiltonian in Paddle Quantum form.

    Args:
        molecule: Hamiltonian in openfermion form.
        filename: Path of .hdf5 file of molecule. Defaults to None.
        multiplicity: Spin multiplicity. Defaults to 1.
        mapping_method: Transformation method, default to ``jordan_wigner``, besides, ``bravyi_kitaev`` is supported. Defaults to ``jordan_wigner``.
        active_electrons: Number of active electrons, default to the case that all electrons are active.
        active_orbitals: Number of active orbitals, default to the case that all orbitals are active.

    Returns:
        Hamiltonian in Paddle Quantum form
    """
    assert mapping_method in ['jordan_wigner', 'bravyi_kitaev'], "Please choose the mapping " \
                                                                 "in ['jordan_wigner', 'bravyi_kitaev']."
    fermionic_h = fermionic_hamiltonian(molecule,
                                        filename,
                                        multiplicity,
                                        active_electrons,
                                        active_orbitals)

    if mapping_method == 'jordan_wigner':
        spin_h = transforms.jordan_wigner(fermionic_h)
    elif mapping_method == 'bravyi_kitaev':
        spin_h = transforms.bravyi_kitaev(fermionic_h)
    return qubitOperator_to_Hamiltonian(spin_h, tol=1e-8)
