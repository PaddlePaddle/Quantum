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
The visualization function in paddle quantum.
"""

import paddle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors as mplcolors
import paddle_quantum
from paddle_quantum.qinfo import partial_trace_discontiguous
from math import sqrt
from typing import Optional, Union, Tuple, List
import os


def plot_state_in_bloch_sphere(
    state: List[paddle_quantum.State],
    show_arrow: Optional[bool] = False,
    save_gif: Optional[bool] = False,
    filename: Optional[str] = None,
    view_angle: Optional[Union[tuple, list]]=None,
    view_dist: Optional[int] = None,
    set_color: Optional[str] =None
) -> None:
    r"""Plot the input quantum state on the Bloch sphere.

    Args:
        state:  List of the input quantum states in the state vector form or the density matrix form.
        show_arrow: Whether to show an arrow for each vector. Defaults to ``False``.
        save_gif: Whether to store the gif. Defaults to ``False``.
        filename:  The name of the gif file to be stored. Defaults to ``None``.
        view_angle:  View angle. The first element is the angle [0-360] to the x-y plane, and the second element is the angle [0-360] to the x-z plane. Defaults to ``(30, 45)``.
        view_dist: View distance. Defaults to ``7``.
        set_color:To set the specified color, consult the ``cmap`` table. Defaults to ``"red to black gradient"``.
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert (
        type(state) == list
        or type(state) == paddle.Tensor
        or type(state) == np.ndarray
        or type(state) == paddle_quantum.State
    ), 'the type of "state" must be "list" or "paddle.Tensor" or "np.ndarray" or "paddle_quantum.State".'
    if type(state) == paddle_quantum.State:
        state = [state.data]
    if type(state) == paddle.Tensor or type(state) == np.ndarray:
        state = [state]
    state_len = len(state)
    assert state_len >= 1, '"state" is NULL.'
    for i in range(state_len):
        assert type(state[i]) == paddle.Tensor or type(state[i]) == np.ndarray or type(state[i]) == paddle_quantum.State, \
            'the type of "state[i]" should be "paddle.Tensor" or "numpy.ndarray" or "paddle_quantum.State".'
    if set_color is not None:
        assert type(set_color) == str, 'the type of "set_color" should be "str".'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'state_in_bloch_sphere.gif'
    if view_angle is None:
        view_angle = (30, 45)
    if view_dist is None:
        view_dist = 7

    # Convert Tensor to numpy
    for i in range(state_len):
        if type(state[i]) == paddle.Tensor:
            state[i] = state[i].numpy()
        if type(state[i]) == paddle_quantum.State:
            state[i] = state[i].data.numpy()

    # Convert state_vector to density_matrix
    for i in range(state_len):
        if state[i].size == 2:
            state_vector = state[i]
            state[i] = np.outer(state_vector, np.conj(state_vector))

    # Calc the bloch_vectors
    bloch_vector_list = []
    for i in range(state_len):
        bloch_vector_tmp = __density_matrix_convert_to_bloch_vector(state[i])
        bloch_vector_tmp = bloch_vector_tmp.numpy()
        bloch_vector_list.append(bloch_vector_tmp)

    # List must be converted to array for slicing.
    bloch_vectors = np.array(bloch_vector_list)

    # A update function for animation class
    def update(frame):
        view_rotating_angle = 5
        new_view_angle = [view_angle[0], view_angle[1] + view_rotating_angle * frame]
        __plot_bloch_sphere(
            ax,
            bloch_vectors,
            show_arrow,
            clear_plt=True,
            view_angle=new_view_angle,
            view_dist=view_dist,
            set_color=set_color,
        )

    # Dynamic update and save
    if save_gif:
        # Helper function to plot vectors on a sphere.
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, projection='3d')

        frames_num = 7
        anim = animation.FuncAnimation(
            fig, update, frames=frames_num, interval=600, repeat=False
        )
        anim.save(filename, dpi=100, writer='pillow')
        # close the plt
        plt.close(fig)

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')

    __plot_bloch_sphere(
        ax,
        bloch_vectors,
        show_arrow,
        clear_plt=True,
        view_angle=view_angle,
        view_dist=view_dist,
        set_color=set_color,
    )

    plt.show()


def plot_multi_qubits_state_in_bloch_sphere(
    state: paddle_quantum.State,
    which_qubits: Optional[List[int]]=None,
    show_arrow: Optional[bool] = False,
    save_gif: Optional[bool] = False,
    save_pic: Optional[bool] = True,
    filename: Optional[str] = None,
    view_angle: Optional[Union[tuple, list]]=None,
    view_dist: Optional[int] = None,
    set_color: Optional[str] = '#0000FF'
) -> None:
    r"""Displaying the quantum state on the Bloch sphere which has multi qubits.

    Args:
        state: List of the input quantum states in the state vector form or the density matrix form.
        which_qubits: Index of qubits to display, default to be fully displayed.
        show_arrow: Whether to show an arrow for each vector. Default is ``False``.
        save_gif:  Whether to store the gif. Default is ``False``.
        save_pic: Whether to store the picture. Default is ``True``.
        filename: The name of the picture to be stored. Defaults to ``None``.
        view_angle: View angle. The first element is the angle [0-360] to the x-y plane, and the second element is the angle [0-360] to the x-z plane. Defaults to ``(30, 45)``.
        view_dist: View distance. Default is ``7``.
        set_color: To set the specified color, consult the ``cmap`` table. Default is ``"blue"``.
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert (
        type(state) == paddle.Tensor
        or type(state) == np.ndarray
        or type(state) == paddle_quantum.State
    ), 'the type of "state" must be "paddle.Tensor" or "np.ndarray" or "paddle_quantum.State".'
    assert type(set_color) == str, 'the type of "set_color" should be "str".'

    if type(state) == paddle_quantum.State:
        state = state.data.numpy()

    n_qubits = int(np.log2(state.shape[0]))

    if which_qubits is None:
        which_qubits = list(range(n_qubits))
    else:
        assert (
            type(which_qubits) == list
        ), 'the type of which_qubits should be None or list'
        assert 1 <= len(which_qubits) <= n_qubits, '展示的量子数量需要小于n_qubits'
        for i in range(len(which_qubits)):
            assert 0 <= which_qubits[i] < n_qubits, '0<which_qubits[i]<n_qubits'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'state_in_bloch_sphere.gif'
    if view_angle is None:
        view_angle = (30, 45)
    if view_dist is None:
        view_dist = 7

    # Convert Tensor to numpy
    if type(state) == paddle.Tensor:
        state = state.numpy()

    # state_vector to density matrix
    if state.shape[0] >= 2 and state.size == state.shape[0]:
        state_vector = state
        state = np.outer(state_vector, np.conj(state_vector))

    # multi qubits state decompose
    if state.shape[0] > 2:
        rho = paddle.to_tensor(state)
        tmp_s = []
        for q in which_qubits:
            tmp_s.append(partial_trace_discontiguous(rho, [q]))
        state = tmp_s
    else:
        state = [state]
    state_len = len(state)

    # Calc the bloch_vectors
    bloch_vector_list = []
    for i in range(state_len):
        bloch_vector_tmp = __density_matrix_convert_to_bloch_vector(state[i])
        bloch_vector_tmp = bloch_vector_tmp.numpy()
        bloch_vector_list.append(bloch_vector_tmp)

    # List must be converted to array for slicing.
    bloch_vectors = np.array(bloch_vector_list)

    # A update function for animation class
    def update(frame):
        view_rotating_angle = 5
        new_view_angle = [view_angle[0], view_angle[1] + view_rotating_angle * frame]
        __plot_bloch_sphere(
            ax,
            bloch_vectors,
            show_arrow,
            clear_plt=True,
            view_angle=new_view_angle,
            view_dist=view_dist,
            set_color=set_color,
        )

    # Dynamic update and save
    if save_gif:
        # Helper function to plot vectors on a sphere.
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, projection='3d')

        frames_num = 7
        anim = animation.FuncAnimation(
            fig, update, frames=frames_num, interval=600, repeat=False
        )
        anim.save(filename, dpi=100, writer='pillow')
        # close the plt
        plt.close(fig)

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    dim = np.ceil(sqrt(len(which_qubits))).astype("int")
    for i in range(1, len(which_qubits) + 1):
        ax = fig.add_subplot(dim, dim, i, projection='3d')
        bloch_vector = np.array([bloch_vectors[i - 1]])
        __plot_bloch_sphere(
            ax,
            bloch_vector,
            show_arrow,
            clear_plt=True,
            view_angle=view_angle,
            view_dist=view_dist,
            set_color=set_color,
        )
    if save_pic:
        plt.savefig('n_qubit_state_in_bloch.png', bbox_inches='tight')
    plt.show()


def plot_rotation_in_bloch_sphere(
        init_state: paddle_quantum.State,
        rotating_angle: List[paddle.Tensor],
        show_arrow: Optional[bool] = False,
        save_gif: Optional[bool] = False,
        filename: Optional[str] = None,
        view_angle: Optional[Union[list, tuple]] = None,
        view_dist: Optional[int] = None,
        color_scheme: Optional[List[str]] = None
) -> None:
    r"""Plot the rotation starting from the initial quantum state on the Bloch sphere.

    Args:
        init_state: Initial quantum state in the state vector form or the density matrix form.
        rotating_angle: Rotation angle ``[theta, phi, lam]``.
        show_arrow: Whether to show an arrow for each vector. Defaults to ``False``.
        save_gif: Whether to store the gif. Defaults to ``False``.
        filename: The name of the gif file to be stored. Defaults to ``None``.
        view_angle: The first element is the angle [0-360] to the x-y plane, and the second element is the angle [0-360] to the x-z plane. Defaults to ``None``.
        view_dist: View distance. Default is ``7``.
        color_scheme:initial color, trace color, and end color, respectively. To set the specified color, consult the ``cmap`` table. Default is ``“red”``.
    """
    # Check input data
    __input_args_dtype_check(show_arrow, save_gif, filename, view_angle, view_dist)

    assert (
        type(init_state) == paddle.Tensor
        or type(init_state) == np.ndarray
        or type(init_state) == paddle_quantum.State
    ), 'the type of input data should be "paddle.Tensor" or "numpy.ndarray" or "paddle_quantum.State".'
    assert (
        type(rotating_angle) == tuple or type(rotating_angle) == list
    ), 'the type of rotating_angle should be "tuple" or "list".'
    assert (
        len(rotating_angle) == 3
    ), 'the rotating_angle must include [theta=paddle.Tensor, phi=paddle.Tensor, lam=paddle.Tensor].'
    for i in range(3):
        assert (
            type(rotating_angle[i]) == paddle.Tensor or type(rotating_angle[i]) == float
        ), 'the rotating_angle must include [theta=paddle.Tensor, phi=paddle.Tensor, lam=paddle.Tensor].'
    if color_scheme is not None:
        assert type(color_scheme) == list and len(color_scheme) <= 3, (
            'the type of "color_scheme" should be "list" and '
            'the length of "color_scheme" should be less than or equal to "3".'
        )
        for i in range(len(color_scheme)):
            assert (
                type(color_scheme[i]) == str
            ), 'the type of "color_scheme[i] should be "str".'

    # Assign a value to an empty variable
    if filename is None:
        filename = 'rotation_in_bloch_sphere.gif'

    # Assign colors to bloch vectors
    color_list = ['orangered', 'lightsalmon', 'darkred']
    if color_scheme is not None:
        for i in range(len(color_scheme)):
            color_list[i] = color_scheme[i]
    set_init_color, set_trac_color, set_end_color = color_list

    for i in range(len(rotating_angle)):
        if type(rotating_angle[i]) == paddle.Tensor:
            rotating_angle[i] = float(rotating_angle[i].numpy()[0])

    theta, phi, lam = rotating_angle
    
    # Convert Tensor to numpy
    if type(init_state) == paddle_quantum.State:
        init_state = init_state.data.numpy()

    if type(init_state) == paddle.Tensor:
        init_state = init_state.numpy()

    # Convert state_vector to density_matrix
    if init_state.size == 2:
        state_vector = init_state
        init_state = np.outer(state_vector, np.conj(state_vector))

    def u_gate_matrix(params):
        theta, phi, lam = params

        if (isinstance(theta, paddle.Tensor) and
                isinstance(phi, paddle.Tensor) and
                isinstance(lam, paddle.Tensor)):
            re_a = paddle.cos(theta / 2)
            re_b = - paddle.cos(lam) * paddle.sin(theta / 2)
            re_c = paddle.cos(phi) * paddle.sin(theta / 2)
            re_d = paddle.cos(phi + lam) * paddle.cos(theta / 2)
            im_a = paddle.zeros([1], 'float64')
            im_b = - paddle.sin(lam) * paddle.sin(theta / 2)
            im_c = paddle.sin(phi) * paddle.sin(theta / 2)
            im_d = paddle.sin(phi + lam) * paddle.cos(theta / 2)

            re = paddle.reshape(paddle.concat([re_a, re_b, re_c, re_d]), [2, 2])
            im = paddle.reshape(paddle.concat([im_a, im_b, im_c, im_d]), [2, 2])

            return re + im * paddle.to_tensor([1j])
        elif (type(theta) is float and
            type(phi) is float and
            type(lam) is float):
            return np.array([[np.cos(theta / 2),
                            -np.exp(1j * lam) * np.sin(theta / 2)],
                            [np.exp(1j * phi) * np.sin(theta / 2),
                            np.exp(1j * phi + 1j * lam) * np.cos(theta / 2)]])
        else:
            assert False
    # Rotating angle
    def rotating_operation(rotating_angle_each):
        gate_matrix = u_gate_matrix(rotating_angle_each)
        if type(gate_matrix) == paddle.Tensor:
            gate_matrix = gate_matrix.numpy()
        return np.matmul(np.matmul(gate_matrix, init_state), gate_matrix.conj().T)

    # Rotating angle division
    rotating_frame = 50
    rotating_angle_list = []
    state = []
    for i in range(rotating_frame + 1):
        angle_each = [
            theta / rotating_frame * i,
            phi / rotating_frame * i,
            lam / rotating_frame * i,
        ]
        rotating_angle_list.append(angle_each)
        state.append(rotating_operation(angle_each))

    state_len = len(state)
    # Calc the bloch_vectors
    bloch_vector_list = []
    for i in range(state_len):
        bloch_vector_tmp = __density_matrix_convert_to_bloch_vector(state[i])
        bloch_vector_tmp = bloch_vector_tmp.numpy()
        bloch_vector_list.append(bloch_vector_tmp)

    # List must be converted to array for slicing.
    bloch_vectors = np.array(bloch_vector_list)

    # A update function for animation class
    def update(frame):
        frame = frame + 2
        if frame <= len(bloch_vectors) - 1:
            __plot_bloch_sphere(
                ax,
                bloch_vectors[1:frame],
                show_arrow=show_arrow,
                clear_plt=True,
                rotating_angle_list=rotating_angle_list,
                view_angle=view_angle,
                view_dist=view_dist,
                set_color=set_trac_color,
            )

            # The starting and ending bloch vector has to be shown
            # show starting vector
            __plot_bloch_sphere(
                ax,
                bloch_vectors[:1],
                show_arrow=True,
                clear_plt=False,
                view_angle=view_angle,
                view_dist=view_dist,
                set_color=set_init_color,
            )

        # Show ending vector
        if frame == len(bloch_vectors):
            __plot_bloch_sphere(
                ax,
                bloch_vectors[frame - 1 : frame],
                show_arrow=True,
                clear_plt=False,
                view_angle=view_angle,
                view_dist=view_dist,
                set_color=set_end_color,
            )

    if save_gif:
        # Helper function to plot vectors on a sphere.
        fig = plt.figure(figsize=(8, 8), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = fig.add_subplot(111, projection='3d')

        # Dynamic update and save
        stop_frames = 15
        frames_num = len(bloch_vectors) - 2 + stop_frames
        anim = animation.FuncAnimation(
            fig, update, frames=frames_num, interval=100, repeat=False
        )
        anim.save(filename, dpi=100, writer='pillow')
        # close the plt
        plt.close(fig)

    # Helper function to plot vectors on a sphere.
    fig = plt.figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, projection='3d')

    # Draw the penultimate bloch vector
    update(len(bloch_vectors) - 3)
    # Draw the last bloch vector
    update(len(bloch_vectors) - 2)

    plt.show()


def plot_density_matrix_graph(density_matrix: paddle_quantum.State, size: Optional[float] = 0.3) -> None:
    r"""Density matrix visualization tools

    Args:
        density_matrix: The state vector or density matrix of quantum state with multi qubits, requiring the number of qubits greater than 1
        size: Bar width, between 0 and 1, default is ``0.3``.

    Raises:
        TypeError: Expected density_matrix to be np.ndarray or paddle.Tensor or paddle_quantum.State
        ValueError: Expected density matrix dim0 equal to dim1
    """
    if not isinstance(
        density_matrix, (np.ndarray, paddle.Tensor, paddle_quantum.State)
    ):
        msg = f'Expected density_matrix to be np.ndarray or paddle.Tensor or paddle_quantum.State, but got {type(density_matrix)}'
        raise TypeError(msg)
    if isinstance(density_matrix, paddle_quantum.State):
        density_matrix = density_matrix.data.numpy()
    if isinstance(density_matrix, paddle.Tensor):
        density_matrix = density_matrix.numpy()
    if density_matrix.shape[0] != density_matrix.shape[1]:
        msg = f'Expected density matrix dim0 equal to dim1, but got dim0={density_matrix.shape[0]}, dim1={density_matrix.shape[1]}'
        raise ValueError(msg)

    real = density_matrix.real
    imag = density_matrix.imag

    figure = plt.figure()
    ax_real = figure.add_subplot(121, projection='3d', title="real")
    ax_imag = figure.add_subplot(122, projection='3d', title="imag")

    xx, yy = np.meshgrid(list(range(real.shape[0])), list(range(real.shape[1])))
    xx, yy = xx.ravel(), yy.ravel()
    real = real.reshape(-1)
    imag = imag.reshape(-1)

    ax_real.bar3d(xx, yy, np.zeros_like(real), size, size, np.abs(real))
    ax_imag.bar3d(xx, yy, np.zeros_like(imag), size, size, np.abs(imag))
    plt.show()

    return


def __plot_bloch_sphere(
        ax,
        bloch_vectors=None,
        show_arrow=False,
        clear_plt=True,
        rotating_angle_list=None,
        view_angle=None,
        view_dist=None,
        set_color=None,
) -> None:
    # Assign a value to an empty variable
    if view_angle is None:
        view_angle = (30, 45)
    if view_dist is None:
        view_dist = 7
    # Define my_color
    if set_color is None:
        color = 'rainbow'
        black_code = '#000000'
        red_code = '#F24A29'
        if bloch_vectors is not None:
            black_to_red = mplcolors.LinearSegmentedColormap.from_list(
                'my_color', [(0, black_code), (1, red_code)], N=len(bloch_vectors[:, 4])
            )
            map_vir = plt.get_cmap(black_to_red)
            color = map_vir(bloch_vectors[:, 4])
    else:
        color = set_color

    # Set the view angle and view distance
    ax.view_init(view_angle[0], view_angle[1])
    ax.dist = view_dist

    # Draw the general frame
    def draw_general_frame():

        # Do not show the grid and original axes
        ax.grid(False)
        ax.set_axis_off()
        ax.view_init(view_angle[0], view_angle[1])
        ax.dist = view_dist

        # Set the lower limit and upper limit of each axis
        # To make the bloch_ball look less flat, the default is relatively flat
        ax.set_xlim3d(xmin=-1.5, xmax=1.5)
        ax.set_ylim3d(ymin=-1.5, ymax=1.5)
        ax.set_zlim3d(zmin=-1, zmax=1.3)

        # Draw a new axes
        coordinate_start_x, coordinate_start_y, coordinate_start_z = np.array(
            [[-1.5, 0, 0], [0, -1.5, 0], [0, 0, -1.5]]
        )
        coordinate_end_x, coordinate_end_y, coordinate_end_z = np.array(
            [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
        )
        ax.quiver(
            coordinate_start_x,
            coordinate_start_y,
            coordinate_start_z,
            coordinate_end_x,
            coordinate_end_y,
            coordinate_end_z,
            arrow_length_ratio=0.03,
            color="black",
            linewidth=0.5,
        )
        ax.text(0, 0, 1.7, r"|0⟩", color="black", fontsize=16)
        ax.text(0, 0, -1.9, r"|1⟩", color="black", fontsize=16)
        ax.text(1.9, 0, 0, r"|+⟩", color="black", fontsize=16)
        ax.text(-1.7, 0, 0, r"|–⟩", color="black", fontsize=16)
        ax.text(0, 1.7, 0, r"|i+⟩", color="black", fontsize=16)
        ax.text(0, -1.9, 0, r"|i–⟩", color="black", fontsize=16)

        # Draw a surface
        horizontal_angle = np.linspace(0, 2 * np.pi, 80)
        vertical_angle = np.linspace(0, np.pi, 80)
        surface_point_x = np.outer(np.cos(horizontal_angle), np.sin(vertical_angle))
        surface_point_y = np.outer(np.sin(horizontal_angle), np.sin(vertical_angle))
        surface_point_z = np.outer(
            np.ones(np.size(horizontal_angle)), np.cos(vertical_angle)
        )
        ax.plot_surface(
            surface_point_x,
            surface_point_y,
            surface_point_z,
            rstride=1,
            cstride=1,
            color="black",
            linewidth=0.05,
            alpha=0.03,
        )

        # Draw circle
        def draw_circle(
            circle_horizon_angle, circle_vertical_angle, linewidth=0.5, alpha=0.2
        ):
            r = 1
            circle_point_x = (
                r * np.cos(circle_vertical_angle) * np.cos(circle_horizon_angle)
            )
            circle_point_y = (
                r * np.cos(circle_vertical_angle) * np.sin(circle_horizon_angle)
            )
            circle_point_z = r * np.sin(circle_vertical_angle)
            ax.plot(
                circle_point_x,
                circle_point_y,
                circle_point_z,
                color="black",
                linewidth=linewidth,
                alpha=alpha,
            )

        # draw longitude and latitude
        def draw_longitude_and_latitude():
            # Draw longitude
            num = 3
            theta = np.linspace(0, 0, 100)
            psi = np.linspace(0, 2 * np.pi, 100)
            for i in range(num):
                theta = theta + np.pi / num
                draw_circle(theta, psi)

            # Draw latitude
            num = 6
            theta = np.linspace(0, 2 * np.pi, 100)
            psi = np.linspace(-np.pi / 2, -np.pi / 2, 100)
            for i in range(num):
                psi = psi + np.pi / num
                draw_circle(theta, psi)

            # Draw equator
            theta = np.linspace(0, 2 * np.pi, 100)
            psi = np.linspace(0, 0, 100)
            draw_circle(theta, psi, linewidth=0.5, alpha=0.2)

            # Draw prime meridian
            theta = np.linspace(0, 0, 100)
            psi = np.linspace(0, 2 * np.pi, 100)
            draw_circle(theta, psi, linewidth=0.5, alpha=0.2)

        # If the number of data points exceeds 20, no longitude and latitude lines will be drawn.
        if bloch_vectors is not None and len(bloch_vectors) < 52:
            draw_longitude_and_latitude()
        elif bloch_vectors is None:
            draw_longitude_and_latitude()

        # Draw three invisible points
        invisible_points = np.array(
            [
                [0.03440399, 0.30279721, 0.95243384],
                [0.70776026, 0.57712403, 0.40743499],
                [0.46991358, -0.63717908, 0.61088792],
            ]
        )
        ax.scatter(
            invisible_points[:, 0],
            invisible_points[:, 1],
            invisible_points[:, 2],
            c='w',
            alpha=0.01,
        )

    # clean plt
    if clear_plt:
        ax.cla()
        draw_general_frame()

    # Draw the data points
    if bloch_vectors is not None:
        ax.scatter(
            bloch_vectors[:, 0],
            bloch_vectors[:, 1],
            bloch_vectors[:, 2],
            c=color,
            alpha=1.0,
        )

    # if show the rotating angle
    if rotating_angle_list is not None:
        bloch_num = len(bloch_vectors)
        (
            rotating_angle_theta,
            rotating_angle_phi,
            rotating_angle_lam,
        ) = rotating_angle_list[bloch_num - 1]
        rotating_angle_theta = round(rotating_angle_theta, 6)
        rotating_angle_phi = round(rotating_angle_phi, 6)
        rotating_angle_lam = round(rotating_angle_lam, 6)

        # Shown at the top right of the perspective
        display_text_angle = [-(view_angle[0] - 10), (view_angle[1] + 10)]
        text_point_x = 2 * np.cos(display_text_angle[0]) * np.cos(display_text_angle[1])
        text_point_y = (
            2 * np.cos(display_text_angle[0]) * np.sin(-display_text_angle[1])
        )
        text_point_z = 2 * np.sin(-display_text_angle[0])
        ax.text(
            text_point_x,
            text_point_y,
            text_point_z,
            r'$\theta=' + str(rotating_angle_theta) + r'$',
            color="black",
            fontsize=14,
        )
        ax.text(
            text_point_x,
            text_point_y,
            text_point_z - 0.1,
            r'$\phi=' + str(rotating_angle_phi) + r'$',
            color="black",
            fontsize=14,
        )
        ax.text(
            text_point_x,
            text_point_y,
            text_point_z - 0.2,
            r'$\lambda=' + str(rotating_angle_lam) + r'$',
            color="black",
            fontsize=14,
        )

    # If show the bloch_vector
    if show_arrow:
        ax.quiver(
            0,
            0,
            0,
            bloch_vectors[:, 0],
            bloch_vectors[:, 1],
            bloch_vectors[:, 2],
            arrow_length_ratio=0.05,
            color=color,
            alpha=1.0,
        )


def __density_matrix_convert_to_bloch_vector(
    density_matrix: paddle_quantum.State,
) -> paddle.Tensor:
    assert (
        type(density_matrix) == np.ndarray
        or type(density_matrix) == paddle_quantum.State
    ), 'the type of "state" must be "np.ndarray" or "paddle_quantum.State".'
    if isinstance(density_matrix, paddle_quantum.State):
        density_matrix = density_matrix.data.numpy()

    # Pauli Matrix
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])

    # Convert a density matrix to a Bloch vector.
    ax = np.trace(np.dot(density_matrix, pauli_x)).real
    ay = np.trace(np.dot(density_matrix, pauli_y)).real
    az = np.trace(np.dot(density_matrix, pauli_z)).real

    # Calc the length of bloch vector
    length = ax ** 2 + ay ** 2 + az ** 2
    length = sqrt(length)
    if length > 1.0:
        length = 1.0

    # Calc the color of bloch vector, the value of the color is proportional to the length
    color = length

    bloch_vector = [ax, ay, az, length, color]

    # You must use an array, which is followed by slicing and taking a column
    bloch_vector = np.array(bloch_vector)

    return paddle.to_tensor(bloch_vector)


def __input_args_dtype_check(show_arrow: bool, save_gif: bool, filename: str, view_angle: Union[tuple, list], view_dist: int) -> None:
    if show_arrow is not None:
        assert type(show_arrow) == bool, 'the type of "show_arrow" should be "bool".'
    if save_gif is not None:
        assert type(save_gif) == bool, 'the type of "save_gif" should be "bool".'
    if save_gif:
        if filename is not None:
            assert type(filename) == str, 'the type of "filename" should be "str".'
            other, ext = os.path.splitext(filename)
            assert ext == '.gif', 'The suffix of the file name must be "gif".'
            # If it does not exist, create a folder
            path, file = os.path.split(filename)
            if not os.path.exists(path):
                os.makedirs(path)
    if view_angle is not None:
        assert (
            type(view_angle) == list or type(view_angle) == tuple
        ), 'the type of "view_angle" should be "list" or "tuple".'
        for i in range(2):
            assert (
                type(view_angle[i]) == int
            ), 'the type of "view_angle[0]" and "view_angle[1]" should be "int".'
    if view_dist is not None:
        assert type(view_dist) == int, 'the type of "view_dist" should be "int".'
