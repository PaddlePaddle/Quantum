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
Loads the energy matrix from the Miyazawa-Jernigan potential file.
"""

import os
from typing import Tuple, List
import logging
import numpy as np

__all__ = ["load_energy_matrix_file"]

MJ_POTENTIAL_FILE_PATH = os.path.realpath(
    os.path.dirname(__file__)
)


def load_energy_matrix_file() -> Tuple[np.ndarray, List[str]]:
    r"""Returns the energy matrix from the Miyazawa-Jernigan potential file.

    Note:
        This is an internal function, user does not intended to call it directly.
    """
    logging.info("! Use Miyazawa-Jernigan potential for interaction between amino acides in protein")
    matrix = np.loadtxt(f"{MJ_POTENTIAL_FILE_PATH}/mj_matrix.txt", dtype=str)
    energy_matrix = _parse_energy_matrix(matrix)
    symbols = list(matrix[0, :])
    return energy_matrix, symbols


def _parse_energy_matrix(matrix: np.ndarray) -> np.ndarray:
    r"""
    Parses a matrix loaded from the Miyazawa-Jernigan potential file.

    Note:
        This is an internal function, user does not intended to call it directly.
    """
    energy_matrix = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
    for row in range(1, np.shape(matrix)[0]):
        for col in range(row - 1, np.shape(matrix)[1]):
            energy_matrix[row, col] = float(matrix[row, col])
    energy_matrix = energy_matrix[
        1:,
    ]
    return energy_matrix
