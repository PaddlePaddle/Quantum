# !/usr/bin/env python3
# Copyright (c) 2023 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

from typing import Dict
import warnings
import time
import logging
import argparse
import toml
import numpy as np
import paddle.optimizer as optim
from paddle_quantum.gate import RY, X
from paddle_quantum.qchem import GroundStateSolver, HartreeFock
from utils import DeuteronHamiltonian
logging.basicConfig(filename="log", filemode="w", format="%(message)s", level=logging.INFO)


def main(args):
    time_start = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    logging.info(f"Job start at {time_start:s}")

    parsed_configs: Dict = toml.load(args.config)

    # build deuteron Hamiltonian
    num_qubits = parsed_configs["N"]
    omega = parsed_configs["hbar_omega"]
    V0 = parsed_configs["V0"]
    deuham = DeuteronHamiltonian(omega, V0)
    deuhN = deuham.get_hamiltonian(num_qubits)

    # build HartreeFock circuit
    cir = HartreeFock(num_qubits)
    cir.insert(0, X(0))
    cir.insert(1, RY(range(1, num_qubits)))

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
    e, _ = solver.solve(cir, ham=deuhN, **optimizer_settings[optimizer_name])   

    logging.info("\n#######################################\nSummary\n#######################################")
    logging.info(f"Binding energy={e:.5f}")

    if parsed_configs.get("calc_exact") is True:
        warnings.warn(f"Calculate exact binding energy will diagonalize {2**num_qubits:d}*{2**num_qubits} matrix, shouldn't be used if `N` is large.")
        exact_e = np.linalg.eigvalsh(deuhN.construct_h_matrix())[0]
        logging.info(f"Exact binding energy={exact_e:.5f}")
        logging.info(f"Relative error={abs(e-exact_e)/abs(exact_e):.5f}")
    
    time_stop = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    logging.info(f"\nJob end at {time_stop:s}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate deuteron binding energy.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    main(parser.parse_args())