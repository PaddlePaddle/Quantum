# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

from typing import Dict
import time
import logging
import argparse
import toml
import paddle.optimizer as optim
from paddle_quantum import qchem
from paddle_quantum.qchem import Molecule
from paddle_quantum.qchem import PySCFDriver
from paddle_quantum.qchem import GroundStateSolver
from paddle_quantum.qchem import energy, dipole_moment

#BUG: basicConfig changed in python3.7
logging.basicConfig(filename="log", filemode="w", format="%(message)s", level=logging.INFO)


def main(args):
    time_start = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    logging.info(f"Job start at {time_start:s}")

    parsed_configs: Dict = toml.load(args.config)

    # create molecule
    atom_symbols = parsed_configs["molecule"]["symbols"]
    basis = parsed_configs["molecule"].get("basis", "sto-3g")
    multiplicity = parsed_configs["molecule"].get("multiplicity")
    charge = parsed_configs["molecule"].get("charge")
    use_angstrom = parsed_configs["molecule"].get("use_angstrom", True)

    if parsed_configs.get("driver") is None or parsed_configs["driver"]["name"] == "pyscf":
        driver = PySCFDriver()
    else:
        raise NotImplementedError("Drivers other than PySCFDriver are not implemented yet.")

    if isinstance(atom_symbols, str):
        raise NotImplementedError("`load_geometry` function is not implemented yet.")
    elif isinstance(atom_symbols, list):
        atom_coords = parsed_configs["molecule"]["coords"]
        geometry = list(zip(atom_symbols, atom_coords))
        mol = Molecule(geometry, basis, multiplicity, charge, use_angstrom=use_angstrom, driver=driver)
    else:
        raise ValueError("Symbols can only be string or list, e.g. 'LiH' or ['H', 'Li']")
    mol.build()

    # create ansatz
    num_qubits = mol.num_qubits
    ansatz_settings = parsed_configs["ansatz"]
    ansatz_name = list(ansatz_settings.keys())[0]
    ansatz_class = getattr(qchem, ansatz_name)
    ansatz = ansatz_class(num_qubits, **ansatz_settings[ansatz_name])

    # load optimizer
    if parsed_configs.get("optimizer") is None:
        optimizer_name = "Adam"
        optimizer_settings = {
            "Adam": {
                "learning_rate": 0.4
            }
        }
        optimizer = optim.Adam
    else:
        optimizer_settings = parsed_configs["optimizer"]
        optimizer_name = list(optimizer_settings.keys())[0]
        optimizer = getattr(optim, optimizer_name)

    # calculate properties
    if parsed_configs.get("VQE") is None:
        vqe_settings = {
            "num_iterations": 100,
            "tol": 1e-5,
            "save_every": 10
        }
    else:
        vqe_settings = parsed_configs["VQE"]
    solver = GroundStateSolver(optimizer, **vqe_settings)
    _, psi = solver.solve(ansatz, mol=mol, **optimizer_settings[optimizer_name])
    e = energy(psi, mol)
    d = dipole_moment(psi, mol)

    logging.info("\n#######################################\nSummary\n#######################################")
    logging.info(f"Ground state energy={e:.5f}")
    logging.info(f"dipole moment=({d[0]:.5f}, {d[1]:.5f}, {d[2]:.5f}).")

    time_stop = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    logging.info(f"\nJob end at {time_stop:s}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum chemistry task with paddle quantum.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    main(parser.parse_args())
