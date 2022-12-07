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
The visualization function of ``paddle_quantum.gate.Gate`` for display
in ``paddle_quantum.ansatz.Circuit`` .
"""

from typing import Optional, Union, List, Any
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

import paddle_quantum as pq

# the default parameters of plot
__CIRCUIT_PLOT_PARAM = {
    'scale': 1.0,                 # scale flag
    'circuit_height': 0.55,       # the height of per line
    'offset': 0.12,               # the head and tail offset of horizontal circuit lines
    'line_width': 0.8,
    'block_margin': 0.07,         # the horizontal and vertical margin of a rectangle
    'node_width': 0.2,            # the block width of patches
    'fontsize': 9.5,
    'circle_radius': 0.08,
    'cross_radius': 0.06,
    'dot_radius': 0.03,
    'margin': 0.02,               # the proportion of figure margin
    'tex': False,
}


def scale_circuit_plot_param(scale: float) -> None:
    r'''The scale function of ``__CIRCUIT_PLOT_PARAM`` dictionary

    Args:
        scale: the scalar for scaling the elements in the figure 
    '''
    for key in __CIRCUIT_PLOT_PARAM:
        if isinstance(__CIRCUIT_PLOT_PARAM[key], (int, float)):
            __CIRCUIT_PLOT_PARAM[key] = scale * __CIRCUIT_PLOT_PARAM[key]


def set_circuit_plot_param(**kwargs: Any) -> None:
    r'''The set function of ``__CIRCUIT_PLOT_PARAM`` dictionary

    Args:
        kwargs: parameters to update the ``__CIRCUIT_PLOT_PARAM`` dictionary
    '''
    __CIRCUIT_PLOT_PARAM.update(kwargs)


def get_circuit_plot_param() -> dict:
    r'''The output function of ``__CIRCUIT_PLOT_PARAM`` dictionary

    Returns:
        a copy of ``__CIRCUIT_PLOT_PARAM``
    '''
    return __CIRCUIT_PLOT_PARAM.copy()


def reset_circuit_plot_param() -> None:
    r'''The reset function of ``__CIRCUIT_PLOT_PARAM`` dictionary
    '''
    global __CIRCUIT_PLOT_PARAM
    __CIRCUIT_PLOT_PARAM = {
        'scale': 1.0,
        'circuit_height': 0.55,
        'offset': 0.12,
        'line_width': 0.8,
        'block_margin': 0.07,
        'node_width': 0.2,
        'fontsize': 9.5,
        'circle_radius': 0.08,
        'cross_radius': 0.06,
        'dot_radius': 0.03,
        'margin': 0.02,
        'tex': False,
    }


def _circuit_plot(
    circuit,
    dpi: Optional[int] = 100,
    scale: Optional[float] = 1.0,
    tex: Optional[bool] = False,
) -> Union[None, matplotlib.figure.Figure]:
    r'''display the circuit using matplotlib

    Args:
        save_path: the save path of image
        dpi: dots per inches, here is resolution ratio
        show: whether execute ``plt.show()``
        output: whether return the ``matplotlib.figure.Figure`` instance
        scale: scale coefficient of figure

    Returns:
        a ``matplotlib.figure.Figure`` instance or ``None`` depends on ``output``

    Note:
        Using ``plt.show()`` may cause a distortion, but it will not happen in the figure saved.
        If the depth is too long, there will be some patches unable to display.
    '''
    if scale != __CIRCUIT_PLOT_PARAM['scale']:   # scale
        scale_circuit_plot_param(scale)
    set_circuit_plot_param(tex=tex)

    height = __CIRCUIT_PLOT_PARAM['circuit_height']
    lw = __CIRCUIT_PLOT_PARAM['line_width']
    offset = __CIRCUIT_PLOT_PARAM['offset']
    margin = __CIRCUIT_PLOT_PARAM['margin']

    _x = 0  # initial postion of horizontal
    _y = 0  # initial postion of vertical

    _fig = plt.figure(facecolor='w', edgecolor='w', dpi=dpi)
    _axes = _fig.add_subplot(1, 1, 1,)

    for gate in circuit.sublayers():
        if not isinstance(gate, pq.gate.Gate):
            raise NotImplementedError
        _x += gate.display_in_circuit(_axes, _x)

    for _ in range(circuit.num_qubits):     # plot horizontal lines for all qubits
        line = Line2D((-offset, _x + offset), (_y, _y), lw=lw, color='k', zorder=0)
        _axes.add_line(line)
        _y += height

    # set the figure size and pattern
    _axes.set_axis_off()
    _axes.set_xlim(- offset, _x + offset)
    _axes.set_ylim(_y - height * 0.5,  - height * 0.5)
    _axes.set_aspect('equal')
    _fig.set_figwidth(_x + offset * 2)
    _fig.set_figheight((circuit.num_qubits)*height)
    plt.subplots_adjust(top=1-margin, bottom=margin, right=1-margin, left=margin,)

    return _fig


def _single_qubit_gate_display(ax: matplotlib.axes.Axes, x: float, y: float, h: float, w: float,
                               tex_name: Optional[str] = None,) -> None:
    r'''Add a rectangle gate with name.

    Args:
        ax: matplotlib.axes.Axes instance
        x: the start horizontal position in the figure
        y: the center of vertical postion
        h: height per line
        w: width for one block
        tex_name: the name written in latex to print
    '''
    margin = __CIRCUIT_PLOT_PARAM['block_margin']
    fontsize = __CIRCUIT_PLOT_PARAM['fontsize']
    lw = __CIRCUIT_PLOT_PARAM['line_width']
    tex_ = __CIRCUIT_PLOT_PARAM['tex']

    rect = Rectangle((x+margin, y-h*0.5+margin), height=h-margin*2, width=w-margin*2,
                     lw=lw, fc='w', ec='k', zorder=1)
    ax.add_patch(rect)
    if tex_name is not None:
        ax.text(x+w*0.5, y, s=tex_name, size=fontsize, zorder=2,
                color='k', ha='center', va='center', rasterized=True, usetex=tex_,)


def _multiple_qubit_gate_display(ax: matplotlib.axes.Axes, x: float, y1: float, y2: float, h: float, w: float,
                                 tex_name: Optional[str] = None,) -> None:
    r'''Add a rectangle gate acting on multiple continuous gates with name.

    Args:
        ax: matplotlib.axes.Axes instance
        x: the start horizontal position in the figure
        y1: the center of vertical postion of minimum qubits
        y2: the center of vertical postion of maximum qubits
        h: height per line
        w: width for one block
        tex_name: the name written in latex to print
    '''

    margin = __CIRCUIT_PLOT_PARAM['block_margin']
    fontsize = __CIRCUIT_PLOT_PARAM['fontsize']
    lw = __CIRCUIT_PLOT_PARAM['line_width']
    tex_ = __CIRCUIT_PLOT_PARAM['tex']
    total_h = y2-y1+h

    rect = Rectangle((x+margin, y1-h*0.5+margin),
                     height=total_h-margin*2, width=w-2*margin, fc='w', ec='k', lw=lw, zorder=1)
    ax.add_patch(rect)
    ax.text(x+w*0.5, (y1+y2)*0.5, s=tex_name, size=fontsize,
            color='k', ha='center', va='center', rasterized=True, usetex=tex_)


def _patch_display(ax: matplotlib.axes.Axes, x: float, y: float, mode: str,) -> None:
    r'''Add a patch

    Args:
        ax: matplotlib.axes.Axes instance
        x: the central horizontal position in the block
        y: the center of vertical postion
        mode: '+' is for controlled object, 'x' is a cross used `SWAP`, '.' is for controlling
    '''
    lw = __CIRCUIT_PLOT_PARAM['line_width']

    if mode == '+':
        crcl_r = __CIRCUIT_PLOT_PARAM['circle_radius']
        reverse_dot = Circle((x, y), crcl_r, fc='none', ec='k', lw=lw, zorder=1)
        line_v = Line2D((x, x), (y-crcl_r, y+crcl_r), zorder=1, c='k', lw=lw,)  # verticle line
        ax.add_patch(reverse_dot)
        ax.add_line(line_v)

    elif mode == '.':
        dot_r = __CIRCUIT_PLOT_PARAM['dot_radius']
        dot = Circle((x, y), dot_r, fc='k', ec='k', lw=lw, zorder=1)
        ax.add_patch(dot)

    elif mode == 'x':
        crs_r = __CIRCUIT_PLOT_PARAM['cross_radius']
        line_left = Line2D((x-crs_r, x+crs_r), (y-crs_r, y+crs_r), c='k', lw=lw, zorder=1)
        line_right = Line2D((x-crs_r, x+crs_r), (y+crs_r, y-crs_r), c='k', lw=lw, zorder=1)
        ax.add_line(line_left)
        ax.add_line(line_right)


def _vertical_line_display(ax: matplotlib.axes.Axes, x: float, y1: float, y2: float,) -> None:
    r'''Add a patch

    Args:
        ax: matplotlib.axes.Axes instance
        x: the central horizontal position in the block
        y1: the vertical postion of one end
        y2: the vertical postion of the other end
    '''
    lw = __CIRCUIT_PLOT_PARAM['line_width']
    line_ = Line2D((x, x), (y1, y2), lw=lw, c='k', zorder=0)
    ax.add_line(line_)


def _param_tex_name_(tex_name: str, theta: Union[float, List[float]]) -> str:
    r'''Combine latex name and its parameters

    Args:
        tex_name: latex name
        theta: parameters to plot
    '''
    if isinstance(theta, float):
        return f'{tex_name[:-1]}(' + f'{theta:.2f}' + ')$'
    param = ''.join(f'{float(value):.2f},' for value in theta)
    return f'{tex_name[:-1]}({param[:-1]})$'


def _is_continuous_list(qubits_idx: List[int]) -> bool:
    r'''Check whether the list is continuous

    Args:
        qubits_idx: a list with different elements
    '''
    return len(qubits_idx) == max(qubits_idx) - min(qubits_idx) + 1


def _not_exist_intersection(list_a: List[float], list_b: List[float]) -> bool:
    r'''Check whether there is an overlap in ``List_a`` and ``List_b``.
    Args:
        List_a: a list with two elements
        List_b: a list with two elements
    '''
    min_a = min(list_a)
    max_a = max(list_a)
    min_b = min(list_b)
    max_b = max(list_b)
    min_ab = min(min_a, min_b)
    max_ab = max(max_a, max_b)
    return max_a+max_b-min_a-min_b < max_ab-min_ab


def _base_gate_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for single qubit base gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']

    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:   # get vertical position
            _single_qubit_gate_display(ax, x, act_qubits*h, h, w, tex_name)
        x += w  # next layer
    return x - x_start


def _base_param_gate_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for single qubit paramgate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']

    for depth in range(gate.depth):
        for param_idx, act_qubits in enumerate(gate.qubits_idx):
            if gate.param_sharing:    # get parameters depending on this flag
                theta = gate.theta[depth]
            else:
                theta = gate.theta[depth, param_idx]
            _single_qubit_gate_display(ax, x, act_qubits*h, h, w, _param_tex_name_(tex_name, theta))
        x += w
    return x - x_start


def _cx_like_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``cx`` like gate .

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']

    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:
            x_c = x + 0.5 * w       # the center of block
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _single_qubit_gate_display(ax, x, act_qubits[1]*h, h, w, tex_name)
            _vertical_line_display(ax, x_c, act_qubits[0]*h, act_qubits[1]*h)
            x += w
    return x - x_start


def _crx_like_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``crx`` like gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']

    for depth in range(gate.depth):
        for param_idx, act_qubits in enumerate(gate.qubits_idx):
            if gate.param_sharing:
                theta = gate.theta[depth]
            else:
                theta = gate.theta[depth, param_idx]
            x_c = x + 0.5 * w        # the center of block
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _single_qubit_gate_display(ax, x, act_qubits[1]*h, h, w, _param_tex_name_(tex_name, theta))
            _vertical_line_display(ax, x_c, act_qubits[0]*h, act_qubits[1]*h)
            x += w
    return x - x_start


def _oracle_like_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``oracle`` like gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']

    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:
            if isinstance(act_qubits, (int, float)):
                _single_qubit_gate_display(ax, x, act_qubits*h, h, w, tex_name)
            else:
                assert _is_continuous_list(act_qubits), 'Discontinuous oracle cannot be plotted.'
                _multiple_qubit_gate_display(ax, x, min(act_qubits)*h, max(act_qubits)*h, h, w, tex_name)
            x += w
    return x - x_start


def _c_oracle_like_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``control oracle`` like gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']

    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:
            assert _is_continuous_list(act_qubits[1:]), 'Discontinuous oracle cannot be plotted.'
            min_ = min(act_qubits[1:])
            max_ = max(act_qubits[1:])
            if act_qubits[0] <= max_ and act_qubits[0] >= min_:
                raise RuntimeError('Invalid input of control oracle. ')

            x_c = x + 0.5 * w
            _patch_display(ax, x_c, h*act_qubits[0], mode='.')
            _multiple_qubit_gate_display(ax, x, min_*h, max_*h, h, w, tex_name)
            _vertical_line_display(ax, x_c, h*act_qubits[0], 0.5*(min_*h+max_*h))
            x += w
    return x - x_start


def _rxx_like_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``rxx`` like gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']

    for depth in range(gate.depth):
        for param_idx, act_qubits in enumerate(gate.qubits_idx):
            assert _is_continuous_list(act_qubits), 'Discontinuous oracle cannot be plotted.'
            if gate.param_sharing:
                theta = gate.theta[depth]
            else:
                theta = gate.theta[depth, param_idx]
            _multiple_qubit_gate_display(ax, x, min(act_qubits)*h, max(act_qubits)*h,
                                         h, w, _param_tex_name_(tex_name, theta))
            x += w
    return x - x_start


def _cnot_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``cnot`` like gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:
            x_c = x + 0.5 * w
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _patch_display(ax, x_c, act_qubits[1]*h, mode='+')
            _vertical_line_display(ax, x_c, act_qubits[0]*h, act_qubits[1]*h)
            x += w
    return x - x_start


def _swap_display(gate, ax: matplotlib.axes.Axes,  x: float,) -> float:
    r'''The display function for ``swap`` like gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:
            x_c = x + 0.5 * w
            _patch_display(ax, x_c, act_qubits[0]*h, mode='x')
            _patch_display(ax, x_c, act_qubits[1]*h, mode='x')
            _vertical_line_display(ax, x_c, act_qubits[0]*h, act_qubits[1]*h)
            x += w
    return x - x_start


def _cswap_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``cswap`` like gate.

     Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']

    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:
            x_c = x + 0.5 * w
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _patch_display(ax, x_c, act_qubits[1]*h, mode='x')
            _patch_display(ax, x_c, act_qubits[2]*h, mode='x')
            _vertical_line_display(ax, x_c, min(act_qubits)*h, max(act_qubits)*h)
            x += w
    return x - x_start


def _tofolli_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``tofolli`` like gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']

    for _ in range(gate.depth):
        for act_qubits in gate.qubits_idx:
            x_c = x + 0.5 * w
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _patch_display(ax, x_c, act_qubits[1]*h, mode='.')
            _patch_display(ax, x_c, act_qubits[2]*h, mode='+')
            _vertical_line_display(ax, x_c, min(act_qubits)*h, max(act_qubits)*h)
            x += w
    return x - x_start


def _linear_entangled_layer_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``linear entangled layer`` gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    block_w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    dot_w = __CIRCUIT_PLOT_PARAM['node_width']

    for depth in range(gate.depth):
        for param_idx, act_qubits in enumerate(gate.qubits_idx):   # `ry` layer
            _single_qubit_gate_display(ax, x, act_qubits*h, h, block_w,
                                       f'$R_y({float(gate.theta[depth][param_idx][0]):.2f})$')
        x += block_w

        for idx in range(len(gate.qubits_idx)-1):   # `cnot` layer
            act_qubits = [gate.qubits_idx[idx], gate.qubits_idx[idx + 1]]
            x_c = x + 0.5 * dot_w
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _patch_display(ax, x_c, act_qubits[1]*h, mode='+')
            _vertical_line_display(ax, x_c, act_qubits[0]*h, act_qubits[1]*h)
            x += dot_w

        for param_idx, act_qubits in enumerate(gate.qubits_idx):   # `rz` layer
            _single_qubit_gate_display(ax, x, act_qubits*h, h, block_w,
                                       f'$R_z({float(gate.theta[depth][param_idx][1]):.2f})$')
        x += block_w

        for idx in range(len(gate.qubits_idx)-1):  # `cnot` layer
            act_qubits = [gate.qubits_idx[idx], gate.qubits_idx[idx + 1]]
            x_c = x + 0.5 * dot_w
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _patch_display(ax, x_c, act_qubits[1]*h, mode='+')
            _vertical_line_display(ax, x_c, act_qubits[0]*h, act_qubits[1]*h)
            x += dot_w
    return x - x_start


def _cr_entangled_layer_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``complex or real entangled layer`` gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    tex_name = gate.gate_info['texname']
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    block_w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    dot_w = __CIRCUIT_PLOT_PARAM['node_width']

    if tex_name == '$REntLayer$':   # define the tex name generator for deferent gates
        def _name_generator(theta):
            return f'$R_y({float(theta):.2f})$'
    elif tex_name == '$CEntLayer$':
        def _name_generator(theta):
            name = '$U('
            for value in theta:
                name += f'{float(value):.2f},'
            name = name.strip(',')
            name += ')$'
            return name

    for depth in range(gate.depth):
        for param_idx, act_qubits in enumerate(gate.qubits_idx):
            _single_qubit_gate_display(ax, x, act_qubits*h, h, block_w,
                                       _name_generator(gate.theta[depth][param_idx]))
        x += block_w
        for idx in range(len(gate.qubits_idx)):
            act_qubits = [gate.qubits_idx[idx], gate.qubits_idx[(idx + 1) % len(gate.qubits_idx)]]
            x_c = x + 0.5 * dot_w
            _patch_display(ax, x_c, act_qubits[0]*h, mode='.')
            _patch_display(ax, x_c, act_qubits[1]*h, mode='+')
            _vertical_line_display(ax, x_c, act_qubits[0]*h, act_qubits[1]*h)
            x += dot_w
    return x - x_start


def _cr_block_layer_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``complex or real block layer`` gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''
    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    block_w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    tex_name = gate.gate_info['texname']
    dot_w = __CIRCUIT_PLOT_PARAM['node_width']

    if tex_name == '$RBLayer$':
        def _add_block(ax, x, theta, pos):  # add a block including 4 `ry` and 1 `cnot`
            _single_qubit_gate_display(ax, x, pos[0]*h, h, block_w,
                                       tex_name=f'$R_y({float(theta[0]):.2f})$')
            _single_qubit_gate_display(ax, x, pos[1]*h, h, block_w,
                                       tex_name=f'$R_y({float(theta[1]):.2f})$')
            _single_qubit_gate_display(ax, x+block_w+dot_w, pos[0]*h, h, block_w,
                                       tex_name=f'$R_y({float(theta[2]):.2f})$')
            _single_qubit_gate_display(ax, x+block_w+dot_w, pos[1]*h, h, block_w,
                                       tex_name=f'$R_y({float(theta[3]):.2f})$')
            _patch_display(ax, x+block_w+0.5*dot_w, pos[0]*h, mode='.')
            _patch_display(ax, x+block_w+0.5*dot_w, pos[1]*h, mode='+')
            _vertical_line_display(ax, x+block_w+0.5*dot_w, pos[0]*h, pos[1]*h)

    elif tex_name == '$CBLayer$':
        def _add_block(ax, x, theta, pos,):     # add a block including 4 `u3` and 1 `cnot`
            _single_qubit_gate_display(ax, x, pos[0]*h, h, block_w,
                                       tex_name=f'$U({float(theta[0]):.2f},\
                                        {float(theta[1]):.2f},{float(theta[2]):.2f})$')
            _single_qubit_gate_display(ax, x, pos[1]*h, h, block_w,
                                       tex_name=f'$U({float(theta[3]):.2f},\
                                        {float(theta[4]):.2f},{float(theta[5]):.2f})$')
            _single_qubit_gate_display(ax, x+block_w+dot_w, pos[0]*h, h, block_w,
                                       tex_name=f'$U({float(theta[6]):.2f},\
                                        {float(theta[7]):.2f},{float(theta[8]):.2f})$')
            _single_qubit_gate_display(ax, x+block_w+dot_w, pos[1]*h, h, block_w,
                                       tex_name=f'$U({float(theta[9]):.2f},\
                                        {float(theta[10]):.2f},{float(theta[11]):.2f})$')
            _patch_display(ax, x+block_w+0.5*dot_w, pos[0]*h, mode='.')
            _patch_display(ax, x+block_w+0.5*dot_w, pos[1]*h, mode='+')
            _vertical_line_display(ax, x+block_w+0.5*dot_w, pos[0]*h, pos[1]*h)

    def _get_display_layer():  # arrange the order of block and compress
        display_layer = []
        num_theta_layer = [0, 0]
        act_qubits = gate.qubits_idx
        for layer in range(2):
            indices = []
            for i in range(1, len(act_qubits), 2):
                indices.append([act_qubits[i-1], act_qubits[i]])
            while len(indices) > 0:
                indcs_list = [indices.pop(0)]
                num_theta_layer[layer] += 1
                for _ in range(len(indices)):
                    candidate = indices.pop(0)
                    if all(_not_exist_intersection(indcs, candidate) for indcs in indcs_list):
                        indcs_list.append(candidate)
                    else:
                        indices.append(candidate)
                display_layer.append(indcs_list)
            act_qubits = act_qubits[1:]
        return display_layer, num_theta_layer

    def _add_layer(ax, x, theta, pos_list,):  # add block into layer
        plotted_list = []
        for pos in pos_list:
            _add_block(ax, x, theta[int((gate.qubits_idx.index(pos[0])) / 2)], pos,)
            plotted_list.extend(list(range(min(pos), max(pos)+1)))

    display_layer, num_theta_layer = _get_display_layer()    # get layers
    for depth in range(gate.depth):
        for num_theta, layer in enumerate(display_layer):
            if num_theta < num_theta_layer[0]:   # get parameters
                _add_layer(ax, x, gate.theta[depth, :int((len(gate.qubits_idx)) / 2)], layer)
            else:
                _add_layer(ax, x, gate.theta[depth, int((len(gate.qubits_idx)) / 2):], layer)
            x += 2 * block_w + dot_w
    return x - x_start


def _qaoa_layer_display(gate, ax: matplotlib.axes.Axes, x: float,) -> float:
    r'''The display function for ``qaoa layer`` gate.

    Args:
        gate: the ``paddle_quantum.gate.Gate`` instance
        ax: the ``matplotlib.axes.Axes`` instance
        x: the start horizontal position

    Returns:
        the total width occupied
    '''

    x_start = x
    h = __CIRCUIT_PLOT_PARAM['circuit_height']
    block_w = __CIRCUIT_PLOT_PARAM['scale'] * gate.gate_info['plot_width']
    dot_w = __CIRCUIT_PLOT_PARAM['node_width']

    def _get_display_layer():    # arrange the order of block and compress
        indices = gate.edges
        display_layer = []
        while len(indices) > 0:
            indcs_list = [indices.pop(0)]
            for _ in range(len(indices)):
                candidate = indices.pop(0)
                if all(_not_exist_intersection(indcs, candidate) for indcs in indcs_list):
                    indcs_list.append(candidate)
                else:
                    indices.append(candidate)
            display_layer.append(indcs_list)
        return display_layer

    def _add_block(ax, x, theta, pos):  # add a block including 2 `cnot` and 1 `rz`
        _single_qubit_gate_display(ax, x+dot_w, pos[1]*h, h, block_w, tex_name=f'$R_z({float(theta):.2f})$')
        _patch_display(ax, x+0.5*dot_w, pos[0]*h, mode='.')
        _patch_display(ax, x+0.5*dot_w, pos[1]*h, mode='+')
        _vertical_line_display(ax, x+0.5*dot_w, pos[0]*h, pos[1]*h)
        _patch_display(ax, x+block_w+1.5*dot_w, pos[0]*h, mode='.')
        _patch_display(ax, x+block_w+1.5*dot_w, pos[1]*h, mode='+')
        _vertical_line_display(ax, x+block_w+1.5*dot_w, pos[0]*h, pos[1]*h)

    def _add_layer(axis, x, theta, pos_list):  # add block into layer
        plotted_list = []
        for pos in pos_list:
            _add_block(axis, x, theta, pos)
            plotted_list.extend(list(range(min(pos), max(pos)+1)))

    x_start = x
    gamma = gate.theta[:gate.depth]
    beta = gate.theta[gate.depth:]
    display_layer = _get_display_layer()

    for depth in range(gate.depth):
        for layer in display_layer:
            _add_layer(ax, x, gamma[depth], layer)
            x += block_w + 2 * dot_w
        for y in gate.nodes:
            _single_qubit_gate_display(ax, x, y*h, h, block_w, tex_name=f'$R_x({float(beta[depth]):.2f})$')
        x += block_w
    return x - x_start
