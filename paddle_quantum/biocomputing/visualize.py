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
visualize the protein structure
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["visualize_protein_structure"]

COORDINATE = (1.0/np.sqrt(3)) * np.asarray(
    [[-1, 1, 1], [1, 1, -1], [-1, -1, -1], [1, -1, 1]]
)


def visualize_protein_structure(
    aa_seq: List[str],
    bond_directions: List[int],
    view_angles: Optional[List[float]] = None
):
    r"""
    Args:
        aa_seq: Amino acides sequence.
        bond_directions: Direction of bonds connect neighboring amino acides.
        view_angles: horizontal and azimuthal angles for the final output image.
    """
    if view_angles is None:
        view_angles = [0.0, 0.0]

    num_aa = len(aa_seq)
    relative_coords = np.zeros((num_aa, 3))
    for i in range(1, num_aa):
        relative_coords[i, :] += (-1)**i * COORDINATE[bond_directions[i-1]]
    aa_coords = relative_coords.cumsum(axis=0)

    x_coords = aa_coords[:, 0]
    y_coords = aa_coords[:, 1]
    z_coords = aa_coords[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect([1, 1, 1])
    for i, aa_label in enumerate(aa_seq):
        ax.text(
            x_coords[i],
            y_coords[i],
            z_coords[i],
            aa_label,
            size=10,
            zorder=10,
            color="k"
        )
    ax.plot(x_coords, y_coords, z_coords)
    ax.scatter(x_coords, y_coords, z_coords, s=500)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D structure of protein")
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    fig.savefig(f"{''.join(aa_seq)}_3d_structure.jpg")
