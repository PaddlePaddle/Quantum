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

"""
Quantum chemistry module
"""

import os
import re
import numpy as np
import psi4
import openfermion
from openfermion import MolecularData, transforms
from openfermion.ops import general_basis_change
from paddle_quantum.utils import Hamiltonian

__all__ = [
    "geometry",
    "get_molecular_data",
    "active_space",
    "fermionic_hamiltonian",
    "spin_hamiltonian"
]


def _hamiltonian_transformation(spin_h, tol=1e-8):
    r"""将哈密顿量从 openfermion 格式转换成 Paddle Quantum 格式。

    Warning:
        输入的哈密顿量必须为埃尔米特的，输入的哈密顿中虚数的系数会和实数一起转换成他们的范数 (norm)。

    Args:
        spin_h (openfermion.ops.operators.qubit_operator.QubitOperator): openfermion 格式的哈密顿量
        tol (float, optional): 系数小于 tol 的值将被忽略掉，默认为 1e-8

    Returns:
        paddle_quantum.Hamiltonian object: Paddle Quantum 格式的哈密顿量
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


def _geo_str(geometry):
    r"""创建分子几何信息的字符串。

    Args:
        geometry (list): 包含了分子的几何信息，以 H2 分子为例
        [['H', [-1.68666, 1.79811, 0.0]], ['H', [-1.12017, 1.37343, 0.0]]]。

    Returns:
        str: 分子几何信息的字符串。
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
    molecule,
    charge,
    multiplicity,
    method,
    basis,
    if_print,
    if_save
):
    r"""计算分子的必要信息，包括单体积分 (one-body integrations) 和双体积分 (two-body integrations)，
    以及用 scf 和 fci 的方法计算基态的能量。

    Args:
        molecule (MolecularData object): 包含分子所有信息的类 (class)。
        charge (int): 分子的电荷。
        multiplicity (int): 分子的多重度。
        method (str): 用于计算基态能量的方法，包括 'scf'和 'fci'。
        basis (str): 常用的基组是 'sto-3g', '6-31g'等。更多的基组选择可以参考网站。
        https://psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement。
        if_print (Boolean): 是否需要打印出选定方法 (method) 计算出的分子基态能量。
        if_save (Boolean): 是否需要将分子信息存储成 .hdf5 文件。
    """
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


def geometry(structure=None, file=None):
    r"""读取分子的几何信息。

    Args:
        structure (string, optional): 分子几何信息的字符串形式，以 H2 分子为例
            ``[['H', [-1.68666, 1.79811, 0.0]], ['H', [-1.12017, 1.37343, 0.0]]]``
        file (string, optional): .xyz 文件的路径

    Returns:
        str: 分子的几何信息

    Raises:
        AssertionError: 两个输入参数不可以同时为 ``None`` 。
    """
    if ((structure is None) and (file is None)):
        raise AssertionError('Input must be structure or .xyz file')
    elif file is None:
        shape = np.array(structure).shape
        assert shape[1] == 2, 'The shape of structure must be (n, 2)'
        for i in range(shape[0]):
            assert type(np.array(structure)[:, 0][i]) == str, 'The first position must be element symbol'
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
    geometry,
    charge=0,
    multiplicity=1,
    basis='sto-3g',
    method='scf',
    if_save=True,
    if_print=True,
    name="",
    file_path="."
):
    r"""计算分子的必要信息，包括单体积分（one-body integrations）和双体积分（two-body integrations），
    以及用选定的方法计算基态的能量。

    Args:
        geometry (str): 分子的几何信息
        charge (int, optional): 分子的电荷，默认值为 0
        multiplicity (int, optional): 分子的多重度，默认值为 1
        basis (str, optional): 常用的基组是 ``'sto-3g'`` 、 ``'6-31g'`` 等，默认的基组是 ``'sto-3g'``，更多的基组选择可以参考网站
            https://psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement
        method (str, optional): 用于计算基态能量的方法，包括 ``'scf'`` 和 ``'fci'`` ，默认方法为 ``'scf'``
        if_save (bool, optional): 是否需要将分子信息存储成 .hdf5 文件，默认为 ``True``
        if_print (bool, optional): 是否需要打印出选定方法 (method) 计算出的分子基态能量，默认为 ``True``
        name (str, optional): 命名储存的文件，默认为 ``""``
        file_path (str, optional): 文件的储存路径，默认为 ``"."``

    Returns:
        MolecularData: 包含分子所有信息的类
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

    molecule = MolecularData(geometry,
                             basis=basis,
                             multiplicity=multiplicity,
                             charge=charge,
                             filename=filename)

    _run_psi4(molecule,
              charge,
              multiplicity,
              method,
              basis,
              if_print,
              if_save)

    return molecule


def active_space(electrons,
                 orbitals,
                 multiplicity=1,
                 active_electrons=None,
                 active_orbitals=None):
    r"""对于给定的活跃电子和活跃轨道计算相应的活跃空间（active space）。

    Args:
        electrons (int): 电子数
        orbitals (int): 轨道数
        multiplicity (int, optional): 自旋多重度
        active_electrons (int, optional): 活跃 (active) 电子数，默认情况为所有电子均为活跃电子
        active_orbitals (int, optional): 活跃 (active) 轨道数，默认情况为所有轨道均为活跃轨道

    Returns:
        tuple: 核心轨道和活跃轨道的索引
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


def fermionic_hamiltonian(molecule,
                          filename=None,
                          multiplicity=1,
                          active_electrons=None,
                          active_orbitals=None):
    r"""计算给定分子的费米哈密顿量。

    Args:
        molecule (MolecularData): 包含分子所有信息的类
        filename (str, optional): 分子的 .hdf5 文件的路径
        multiplicity (int, optional): 自旋多重度
        active_electrons (int, optional): 活跃 (active) 电子数，默认情况为所有电子均为活跃电子
        active_orbitals (int, optional): 活跃 (active) 轨道数，默认情况为所有轨道均为活跃轨道

    Returns:
        openfermion.ops.operators.qubit_operator.QubitOperator: openfermion 格式的哈密顿量
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


def spin_hamiltonian(molecule,
                     filename=None,
                     multiplicity=1,
                     mapping_method='jordan_wigner',
                     active_electrons=None,
                     active_orbitals=None):
    r"""生成 Paddle Quantum 格式的哈密顿量

    Args:
        molecule (openfermion.ops.operators.qubit_operator.QubitOperator): openfermion 格式的哈密顿量
        filename (str, optional): 分子的 .hdf5 文件的路径
        multiplicity (int, optional): 自旋多重度
        mapping_method (str, optional): 映射方法，这里默认为 ``'jordan_wigner'`` ，此外还提供 ``'bravyi_kitaev'``
        active_electrons (int, optional): 活跃 (active) 电子数，默认情况为所有电子均为活跃电子
        active_orbitals (int, optional): 活跃 (active) 轨道数默认情况为所有轨道均为活跃轨道

    Returns:
        paddle_quantum.utils.Hamiltonian: Paddle Quantum 格式的哈密顿量
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
    return _hamiltonian_transformation(spin_h, tol=1e-8)
