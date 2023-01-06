# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

import argparse
import toml
import time
import os
import warnings
import logging
from paddle import optimizer as paddle_optimizer
from paddle_quantum.ansatz import Circuit
from paddle_quantum.biocomputing import Protein
from paddle_quantum.biocomputing import ProteinFoldingSolver
from paddle_quantum.biocomputing import visualize_protein_structure

warnings.filterwarnings('ignore')
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
logging.basicConfig(filename="log", filemode="w", level=logging.INFO, format="%(message)s")


def circuit(num_qubits: int, depth: int) -> Circuit:
    r"""Ansatz used in protein folding VQE.
    """
    cir = Circuit(num_qubits)
    cir.superposition_layer()
    for _ in range(depth):
        cir.ry()
        cir.cx()
    cir.ry()
    return cir


def main(args):
    time_start = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    logging.info(f"Job start at {time_start:s}")

    # construct the protein
    parsed_configs = toml.load(args.config)
    aa_seq = parsed_configs["amino_acids"]
    contact_pairs = parsed_configs["possible_contactions"]
    num_aa = len(aa_seq)
    protein = Protein("".join(aa_seq), {(0, 1): 1, (1, 2): 0, (num_aa-2, num_aa-1): 3}, contact_pairs)

    # build the solver
    cir_depth = parsed_configs["depth"]
    cir = circuit(protein.num_qubits, cir_depth)

    penalty_factors = [10.0, 10.0]
    alpha = 0.5
    optimizer = paddle_optimizer.Adam
    num_iterations = parsed_configs["num_iterations"]
    tol = parsed_configs["tol"]
    save_every = parsed_configs["save_every"]
    learning_rate = parsed_configs["learning_rate"]
    problem = ProteinFoldingSolver(penalty_factors, alpha, optimizer, num_iterations, tol, save_every)
    _, protein_str = problem.solve(protein, cir, learning_rate=learning_rate)

    # parse results & plot the 3d structure of protein
    num_config_qubits = protein.num_config_qubits
    bond_directions = [1, 0]
    bond_directions.extend(int(protein_str[slice(i, i + 2)], 2) for i in range(0, num_config_qubits, 2))
    bond_directions.append(3)
    visualize_protein_structure(aa_seq, bond_directions)

    logging.info("\n#######################################\nSummary\n#######################################")
    logging.info(f"Protein bonds direction: {bond_directions}.")
    time_stop = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
    logging.info(f"\nJob end at {time_stop:s}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein folding task with paddle quantum.")
    parser.add_argument("--config", type=str, help="Input the config file with toml format.")
    main(parser.parse_args())
