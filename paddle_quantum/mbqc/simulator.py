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
This module contains the commonly used class and simulation tools in MBQC.
"""

from numpy import random, pi
from networkx import Graph, spring_layout, draw_networkx
import matplotlib.pyplot as plt
from paddle import t, to_tensor, matmul, conj, real, reshape, multiply
from paddle_quantum.mbqc.utils import plus_state, cz_gate, pauli_gate
from paddle_quantum.mbqc.utils import basis, kron, div_str_to_float
from paddle_quantum.mbqc.utils import permute_to_front, permute_systems, print_progress, plot_results
from paddle_quantum.mbqc.qobject import State, Pattern
from paddle_quantum.mbqc.transpiler import transpile

__all__ = [
    "MBQC",
    "simulate_by_mbqc"
]


class MBQC:
    r""" Define a ``MBQC`` class used for measurement based quantum computation.

    The users can define their MBQC models by instantiate the objects of this class.
    """
    def __init__(self):
        self.__graph = None  # Graph in a MBQC model
        self.__pattern = None  # Measurement pattern in a MBQC model

        self.__bg_state = State()  # Background state of computation
        self.__history = [self.__bg_state]  # History of background states
        self.__status = self.__history[-1] if self.__history != [] else None  # latest history item

        self.vertex = None  # Vertex class to maintain all the vertices
        self.__outcome = {}  # Dictionary to store all measurement outcomes
        self.max_active = 0  # Maximum number of active vertices so far

        self.__draw = False  # Switch to draw the dynamical running process
        self.__track = False  # Switch to track the running progress
        self.__pause_time = None  # Pause time for drawing
        self.__pos = None  # Position for drawing

    class Vertex:
        r"""Define the list of maintenance point used for instantiate a ``Vertex`` object.

        Device the nodes in MBQC to three classes and maintain them dynamically.

        Note:
            This is an internal class and the users don't need to use it directly.

        Args:
            total (list): a list contains all the nodes in MBQC diagram, irrespective of the computation process.
            pending (list): a list contains the nodes to be activated. The number of nodes decreases as the operations
            are performed.
            active (list): a list contains the active nodes directly related to the current measurement operations.
            measured (list): a list contains the nodes which have been measured. The number of nodes decreases as the
            operations are performed.
            
        :meta private:
        """
        def __init__(self, total=None, pending=None, active=None, measured=None):
            self.total = [] if total is None else total
            self.pending = [] if pending is None else pending
            self.active = [] if active is None else active
            self.measured = [] if measured is None else measured

    def set_graph(self, graph):
        r"""Set the graphs in MBQC model.

        The users can use this function to transport their own graphs to the ``MBQC`` object.
        
        Args:
            graph: The graphs in MBQC model. The list takes the form ``[V, E]``, where ``V`` are nodes, and ``E`` are edges.
        """
        vertices, edges = graph

        vertices_of_edges = set([vertex for edge in edges for vertex in list(edge)])
        assert vertices_of_edges.issubset(vertices), "edge must be between the graph vertices."

        self.__graph = Graph()
        self.__graph.add_nodes_from(vertices)
        self.__graph.add_edges_from(edges)

        self.vertex = self.Vertex(total=vertices, pending=vertices, active=[], measured=[])

    def get_graph(self):
        r"""Get the information of graphs.

        Returns:
            nx.Graph: graph
        """
        return self.__graph

    def set_pattern(self, pattern):
        r"""Set the measurement patterns of the MBQC model.

        This function is used for transport the measurement patterns to ``MBQC`` object. The measurement patterns
        are acquired by either translation from the quantum circuit or the construction of users.

        Warning:
            The input pattern parameter is of type ``Pattern``, in which the command list contains ``EMC`` commands.

        Args:
            pattern (Pattern): The measurement patterns corresponding to the MBQC algorithms.
        """
        assert isinstance(pattern, Pattern), "please input a pattern of type 'Pattern'."

        self.__pattern = pattern
        cmds = self.__pattern.commands[:]

        # Check if the pattern is a standard EMC form
        cmd_map = {"E": 1, "M": 2, "X": 3, "Z": 4, "S": 5}
        cmd_num_wild = [cmd_map[cmd.name] for cmd in cmds]
        cmd_num_standard = cmd_num_wild[:]
        cmd_num_standard.sort(reverse=False)
        assert cmd_num_wild == cmd_num_standard, "input pattern is not a standard EMC form."

        # Set graph by entanglement commands
        edges = [tuple(cmd.which_qubits) for cmd in cmds if cmd.name == "E"]
        vertices = list(set([vertex for edge in edges for vertex in list(edge)]))
        graph = [vertices, edges]
        self.set_graph(graph)

    def get_pattern(self):
        r"""Get the information of the measurement patterns.

        Returns:
            Pattern: measurement pattern
        """
        return self.__pattern

    def set_input_state(self, state=None):
        r"""Set the input state to be replaced.

        Warning:
            Unlike the circuit model, the initial state of MBQC model is |+> state. If the users don't use
            this method to initialize the quantum state, the initial state will be |+> state.
            If the users run the MBQC model in measurement mode, the system labels of the input state here
            will be restricted to natural number(start from 0) of ``int`` type.

        Args:
            state: the input state to be replaced, the default is |+> state.
        """
        assert self.__graph is not None, "please set 'graph' or 'pattern' before calling 'set_input_state'."
        assert isinstance(state, State) or state is None, "please input a state of type 'State'."
        vertices = list(self.__graph.nodes)

        if state is None:
            vector = plus_state()
            system = [vertices[0]]  # Activate the first vertex, system should be a list
        else:
            vector = state.vector
            # If a pattern is set, map the input state system to the pattern's input
            if self.__pattern is not None:
                assert all(isinstance(label, int) for label in state.system), "please input system labels of type 'int'"
                assert all(label >= 0 for label in state.system), "please input system labels with non-negative values"

                system = [label for label in self.__pattern.input_ if int(div_str_to_float(label[0])) in state.system]
            else:
                system = state.system
        assert set(system).issubset(vertices), "input system labels must be a subset of graph vertices."

        self.__bg_state = State(vector, system)
        self.__history = [self.__bg_state]
        self.__status = self.__history[-1]
        self.vertex = self.Vertex(total=vertices,
                                  pending=list(set(vertices).difference(system)),
                                  active=system,
                                  measured=[])
        self.max_active = len(self.vertex.active)

    def __set_position(self, pos):
        r"""Set the coordinates of the nodes during the dynamical plotting process.

        Note:
            This is an internal method and the users don't need to use it directly.

        Args:
            pos (dict or bool, optional): the dict of the nodes' labels or built-in choice of coordinates.
            The built-in choice of coordinates are: ``True`` is the coordinates of the measurement patterns,
            ``False`` is the ``spring_layout`` coordinates.
        """
        assert isinstance(pos, bool) or isinstance(pos, dict), "'pos' should be either bool or dict."
        if isinstance(pos, dict):
            self.__pos = pos
        elif pos:
            assert self.__pattern is not None, "'pos=True' must be chosen after a pattern is set."
            self.__pos = {v: [div_str_to_float(v[1]), - div_str_to_float(v[0])] for v in list(self.__graph.nodes)}
        else:
            self.__pos = spring_layout(self.__graph)  # Use 'spring_layout' otherwise

    def __draw_process(self, which_process, which_qubit):
        r""" Draw pictures according to the status of the nodes. This method is used for showing the
            computation process of the MBQC model.

        Note:
            This is an internal method and the users don't need to use directly.

        Args:
            which_process (str): The status of MBQC model, "measuring", "active" or "measured"
            which_qubit (any): the current focal node. Any type(e.g. ``str``, ``tuple``) can be input, but should
            match the type of the labels of the graph.
        """
        if self.__draw:
            assert which_process in ["measuring", "active", "measured"]
            assert which_qubit in self.vertex.total, "'which_qubit' must be in the graph."

            vertex_sets = []
            # Find where the 'which_qubit' is
            if which_qubit in self.vertex.pending:
                pending = self.vertex.pending[:]
                pending.remove(which_qubit)
                vertex_sets = [pending, self.vertex.active, [which_qubit], self.vertex.measured]
            elif which_qubit in self.vertex.active:
                active = self.vertex.active[:]
                active.remove(which_qubit)
                vertex_sets = [self.vertex.pending, active, [which_qubit], self.vertex.measured]
            elif which_qubit in self.vertex.measured:
                vertex_sets = [self.vertex.pending, self.vertex.active, [], self.vertex.measured]

            # Indentify ancilla vertices
            ancilla_qubits = []
            if self.__pattern is not None:
                for vertex in list(self.__graph.nodes):
                    row_coordinate = div_str_to_float(vertex[0])
                    col_coordinate = div_str_to_float(vertex[1])
                    # Ancilla vertices do not have integer coordinates
                    if abs(col_coordinate - int(col_coordinate)) >= 1e-15 \
                            or abs(row_coordinate - int(row_coordinate)) >= 1e-15:
                        ancilla_qubits.append(vertex)

            plt.cla()
            plt.title("MBQC Running Process", fontsize=15)
            plt.xlabel("Measuring (RED)  Active (GREEN)  Pending (BLUE)  Measured (GRAY)", fontsize=12)
            plt.grid()
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(500, 100, 800, 600)
            colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:gray']
            for j in range(4):
                for vertex in vertex_sets[j]:
                    options = {
                        "nodelist": [vertex],
                        "node_color": colors[j],
                        "node_shape": '8' if vertex in ancilla_qubits else 'o',
                        "with_labels": False,
                        "width": 3,
                    }
                    draw_networkx(self.__graph, self.__pos, **options)
                    ax = plt.gca()
                    ax.margins(0.20)
                    plt.axis("on")
                    ax.set_axisbelow(True)
            plt.pause(self.__pause_time)

    def draw_process(self, draw=True, pos=False, pause_time=0.5):
        r"""Dynamically plot the computation process of the MBQC model.

        Args:
            draw (bool, optional): The boolean switch of whether plot the computation process.
            pos (bool or dict, optional): the dict of the nodes' labels or built-in choice of coordinates. 
                The built-in choice of coordinates are: ``True`` is the coordinates of the measurement patterns, 
                ``False`` is the ``spring_layout`` coordinates.
            pause_time (float, optional): The time step for updating the picture.
        """
        assert self.__graph is not None, "please set 'graph' or 'pattern' before calling 'draw_process'."
        assert isinstance(draw, bool), "'draw' must be bool."
        assert isinstance(pos, bool) or isinstance(pos, dict), "'pos' should be either bool or dict."
        assert pause_time > 0, "'pause_time' must be strictly larger than 0."

        self.__draw = draw
        self.__pause_time = pause_time

        if self.__draw:
            plt.figure()
            plt.ion()
            self.__set_position(pos)

    def track_progress(self, track=True):
        r""" A switch for whether showing the progress bar of MBQC computation process.

        Args:
            track (bool, optional): ``True`` open the progress bar,  ``False`` close the progress bar.
        """
        assert isinstance(track, bool), "the parameter 'track' must be bool."
        self.__track = track

    def __apply_cz(self, which_qubits_list):
        r"""Apply a controlled-Z gate to given two qubits.

        Note:
            This is an internal method and the users don't use it directly.

        Warning:
            The two qubits that the CZ gate applies to must be active.

        Args:
            which_qubits_list (list): A list contains the qubits that the CZ gate applies to.
            e.g.``[(1, 2), (3, 4),...]``
        """
        for which_qubits in which_qubits_list:
            assert set(which_qubits).issubset(self.vertex.active), \
                "vertices in 'which_qubits_list' must be activated first."
            assert which_qubits[0] != which_qubits[1], \
                'the control and target qubits must not be the same.'

            # Find the control and target qubits and permute them to the front
            self.__bg_state = permute_to_front(self.__bg_state, which_qubits[0])
            self.__bg_state = permute_to_front(self.__bg_state, which_qubits[1])

            new_state = self.__bg_state
            new_state_len = new_state.length
            qua_length = int(new_state_len / 4)
            cz = cz_gate()
            # Reshape the state, apply CZ and reshape it back
            new_state.vector = reshape(matmul(cz, reshape(new_state.vector, [4, qua_length])), [new_state_len, 1])

            # Update the order of active vertices and the background state
            self.vertex.active = new_state.system
            self.__bg_state = State(new_state.vector, new_state.system)

    def __apply_pauli_gate(self, gate, which_qubit):
        r"""Apply a Pauli gate to the given single qubit.

        Note:
            This is an internal method and the users don't use it directly.

        Args:
            gate (str): Pauli gate, "I", "X", "Y", "Z". Use "X" and "Z" gate when correcting the byproduct.
            which_qubit (any): The label of the system that the Pauli gate applies to. Any type
            (e.g. ``str``, ``tuple`` ) can be used, as long as the type matches the type of labels of the
            nodes in MBQC model.
        """
        new_state = permute_to_front(self.__bg_state, which_qubit)
        new_state_len = new_state.length
        half_length = int(new_state_len / 2)
        gate_mat = pauli_gate(gate)
        # Reshape the state, apply X and reshape it back
        new_state.vector = reshape(matmul(gate_mat, reshape(new_state.vector, [2, half_length])), [new_state_len, 1])
        # Update the order of active vertices and the background state
        self.vertex.active = new_state.system
        self.__bg_state = State(new_state.vector, new_state.system)

    def __create_graph_state(self, which_qubit):
        r""" Generate the minimal graph state for measuring the current node.

        Note:
            This is an internal method and the users don't use it directly.

        Args:
            which_qubit (any): The system labels of the qubit to be measured. Any type
            (e.g. ``str``, ``tuple`` ) can be used, as long as the type matches the type of labels of the
            nodes in MBQC model.
        """
        # Find the neighbors of 'which_qubit'
        which_qubit_neighbors = set(self.__graph.neighbors(which_qubit))
        # Exclude the qubits already measured
        neighbors_not_measured = which_qubit_neighbors.difference(set(self.vertex.measured))
        # Create a list of system labels that will be applied to cz gates
        cz_list = [(which_qubit, qubit) for qubit in neighbors_not_measured]
        # Get the qubits to be activated
        append_qubits = {which_qubit}.union(neighbors_not_measured).difference(set(self.vertex.active))
        # Update active and pending lists
        self.vertex.active += list(append_qubits)
        self.vertex.pending = list(set(self.vertex.pending).difference(self.vertex.active))

        # Compute the new background state vector
        new_bg_state_vector = kron([self.__bg_state.vector] + [plus_state() for _ in append_qubits])

        # Update the background state and apply cz
        self.__bg_state = State(new_bg_state_vector, self.vertex.active)
        self.__apply_cz(cz_list)
        self.__draw_process("active", which_qubit)

    def __update(self):
        r"""Update the history and the quantum states' information.
        """
        self.__history.append(self.__bg_state)
        self.__status = self.__history[-1]

    def measure(self, which_qubit, basis_list):
        r"""Measure given qubits with given measurement basis.

        Note:
            This is one of the most common methods we use after instantiating an MBQC object.
            Here we optimize the single bit measurement simulation to the maximum extent.
            Once use this method, the MBQC class will automatically activate the related nodes, generate
            the corresponding graph state, measure specific qubits and store the results of the numerical
            simulations.

        Warning:
            If and only if the users use this method, the MBQC model carries out the computation.

        Args:
            which_qubit (any): The system labels of the qubit to be measured. Any type
                (e.g. ``str``, ``tuple`` ) can be used, as long as the type matches the type of labels 
                of the nodes in MBQC model.
            basis_list (list): a list composed of measurement basis, the elements are column vectors
                of type ``Tensor``.

        Code example:

        .. code-block:: python

            from paddle_quantum.mbqc.simulator import MBQC
            from paddle_quantum.mbqc.qobject import State
            from paddle_quantum.mbqc.utils import zero_state, basis

            G = [['1', '2', '3'], [('1', '2'), ('2', '3')]]
            mbqc = MBQC()
            mbqc.set_graph(G)
            state = State(zero_state(), ['1'])
            mbqc.set_input_state(state)
            mbqc.measure('1', basis('X'))
            mbqc.measure('2', basis('X'))
            print("Measurement outcomes: ", mbqc.get_classical_output())

        ::

            Measurement outcomes:  {'1': 0, '2': 1}
        """
        self.__draw_process("measuring", which_qubit)
        self.__create_graph_state(which_qubit)
        assert which_qubit in self.vertex.active, 'the qubit to be measured must be activated first.'

        new_bg_state = permute_to_front(self.__bg_state, which_qubit)
        self.vertex.active = new_bg_state.system
        half_length = int(new_bg_state.length / 2)

        eps = 10 ** (-10)
        prob = [0, 0]
        state_unnorm = [0, 0]

        # Calculate the probability and post-measurement states
        for result in [0, 1]:
            basis_dagger = t(conj(basis_list[result]))
            # Reshape the state, multiply the basis and reshape it back
            state_unnorm[result] = reshape(matmul(basis_dagger,
                                                  reshape(new_bg_state.vector, [2, half_length])), [half_length, 1])
            probability = matmul(t(conj(state_unnorm[result])), state_unnorm[result])
            is_complex128 = probability.dtype == to_tensor([], dtype='complex128').dtype
            prob[result] = real(probability) if is_complex128 else probability

        # Randomly choose a result and its corresponding post-measurement state
        if prob[0].numpy().item() < eps:
            result = 1
            post_state_vector = state_unnorm[1]
        elif prob[1].numpy().item() < eps:
            result = 0
            post_state_vector = state_unnorm[0]
        else:  # Take a random choice of outcome
            result = random.choice(2, 1, p=[prob[0].numpy().item(), prob[1].numpy().item()]).item()
            # Normalize the post-measurement state
            post_state_vector = state_unnorm[result] / prob[result].sqrt()

        # Write the measurement result into the dict
        self.__outcome.update({which_qubit: int(result)})
        # Update measured, active lists
        self.vertex.measured.append(which_qubit)
        self.max_active = max(len(self.vertex.active), self.max_active)
        self.vertex.active.remove(which_qubit)

        # Update the background state and history list
        self.__bg_state = State(post_state_vector, self.vertex.active)
        self.__update()

        self.__draw_process("measured", which_qubit)

    def sum_outcomes(self, which_qubits, start=0):
        r"""Based on the input system labels, find the corresponding measurement results in the dict and sum over.


        Note:
            When correcting the byproduct or defining the angle of adaptive measurement, one
            can use this method to sum over the measurement results of given qubits.
        
        Args:
            which_qubits (list):the list contains the system labels of the nodes whose measurement
                results should be find out and summed over.
            start (int): an extra integer added to the results.

        Returns:
            int: The sum of the measurement results of given qubit.

        Code example:

        .. code-block:: python

            from paddle_quantum.mbqc.simulator import MBQC
            from paddle_quantum.mbqc.qobject import State
            from paddle_quantum.mbqc.utils import zero_state, basis

            G = [['1', '2', '3'], [('1', '2'), ('2', '3')]]
            mbqc = MBQC()
            mbqc.set_graph(G)
            input_state = State(zero_state(), ['1'])
            mbqc.set_input_state(input_state)
            mbqc.measure('1', basis('X'))
            mbqc.measure('2', basis('X'))
            mbqc.measure('3', basis('X'))
            print("All measurement outcomes: ", mbqc.get_classical_output())
            print("Sum of outcomes of qubits '1' and '2': ", mbqc.sum_outcomes(['1', '2']))
            print("Sum of outcomes of qubits '1', '2' and '3' with an extra 1: ", mbqc.sum_outcomes(['1', '2', '3'], 1))

        ::

            All measurement outcomes:  {'1': 0, '2': 0, '3': 1}
            Sum of outcomes of qubits '1' and '2':  0
            Sum of outcomes of qubits '1', '2' and '3' with an extra 1:  2
        """
        assert isinstance(start, int), "'start' must be of type int."

        return sum([self.__outcome[label] for label in which_qubits], start)

    def correct_byproduct(self, gate, which_qubit, power):
        r"""Correct the byproduct of the measured quantum state.

        Note:
            This is a commonly used method after the measurement process of the MBQC model.

        Args:
            gate (str): ``'X'`` or ``'Z'``, representing Pauli X or Pauli Z correction respectively.
                which_qubit (any): The system labels of the qubit to be processed. Any type
                (e.g. ``str``, ``tuple`` ) can be used, as long as the type matches the type of labels of the
                nodes in MBQC model.
            power (int): the index of the byproduct correcting.

        Code example:
            Here is an example of quantum teleportation in MBQC framework.

        .. code-block:: python

            from paddle_quantum.mbqc.simulator import MBQC
            from paddle_quantum.mbqc.qobject import State
            from paddle_quantum.mbqc.utils import random_state_vector, basis, compare_by_vector

            G = [['1', '2', '3'], [('1', '2'), ('2', '3')]]
            state = State(random_state_vector(1), ['1'])
            mbqc = MBQC()
            mbqc.set_graph(G)
            mbqc.set_input_state(state)
            mbqc.measure('1', basis('X'))
            mbqc.measure('2', basis('X'))
            outcome = mbqc.get_classical_output()
            mbqc.correct_byproduct('Z', '3', outcome['1'])
            mbqc.correct_byproduct('X', '3', outcome['2'])
            state_out = mbqc.get_quantum_output()
            state_std = State(state.vector, ['3'])
            compare_by_vector(state_out, state_std)

        ::

            Norm difference of the given states is:
             0.0
            They are exactly the same states.
        """
        assert gate in ['X', 'Z'], "'gate' must be 'X' or 'Z'."
        assert isinstance(power, int), "'power' must be of type 'int'."

        if power % 2 == 1:
            self.__apply_pauli_gate(gate, which_qubit)
        self.__update()

    def __run_cmd(self, cmd):
        r"""Carry out the command of measurement or byproduct correction.

        Args:
            cmd (Pattern.CommandM / Pattern.CommandX / Pattern.CommandZ): the command of measurement
            or byproduct correction.
        """
        assert cmd.name in ["M", "X", "Z"], "the input 'cmd' must be CommandM, CommandX or CommandZ."
        if cmd.name == "M":  # Execute measurement commands
            signal_s = self.sum_outcomes(cmd.domain_s)
            signal_t = self.sum_outcomes(cmd.domain_t)
            # The adaptive angle is (-1)^{signal_s} * angle + {signal_t} * pi
            adaptive_angle = multiply(to_tensor([(-1) ** signal_s], dtype="float64"), cmd.angle) \
                             + to_tensor([signal_t * pi], dtype="float64")
            self.measure(cmd.which_qubit, basis(cmd.plane, adaptive_angle))
        else:  # Execute byproduct correction commands
            power = self.sum_outcomes(cmd.domain)
            self.correct_byproduct(cmd.name, cmd.which_qubit, power)

    def __run_cmd_lst(self, cmd_lst, bar_start, bar_end):
        r"""Carry out a list of commands, including measurement and byproduct correction.

        Args:
            cmd_lst (list): the list of commands, including measurement and byproduct correction.
            bar_start (int): the starting point of the progress bar.
            bar_end (int): the end point of the progress bar.
        """
        for i in range(len(cmd_lst)):
            cmd = cmd_lst[i]
            self.__run_cmd(cmd)
            print_progress((bar_start + i + 1) / bar_end, "Pattern Running Progress", self.__track)

    def __kron_unmeasured_qubits(self):
        r"""This method initialize the nodes without being applied by CZ gate to |+> state, and take the tensor
        product between these nodes and the current quantum state.

        Warning:
            This method is used when the users input the measurement patterns. When the users input a graph,
            if the nodes are not activated, we deem the users do nothing to the node by default.
        """
        # Turn off the plot switch
        self.__draw = False
        # As the create_graph_state function would change the measured qubits list, we need to record it
        measured_qubits = self.vertex.measured[:]

        for qubit in list(self.__graph.nodes):
            if qubit not in self.vertex.measured:
                self.__create_graph_state(qubit)
                # Update vertices and backgrounds
                self.vertex.measured.append(qubit)
                self.max_active = max(len(self.vertex.active), self.max_active)
                self.__bg_state = State(self.__bg_state.vector, self.vertex.active)

        # Restore the measured qubits
        self.vertex.measured = measured_qubits

    def run_pattern(self):
        r"""Run the MBQC model by the measurement patterns set before.

        Warning:
            This method must be used after ``set_pattern``.
        """
        assert self.__pattern is not None, "please use this method after calling 'set_pattern'!"

        # Execute measurement commands and correction commands
        cmd_m_lst = [cmd for cmd in self.__pattern.commands if cmd.name == "M"]
        cmd_c_lst = [cmd for cmd in self.__pattern.commands if cmd.name in ["X", "Z"]]
        bar_end = len(cmd_m_lst + cmd_c_lst)

        self.__run_cmd_lst(cmd_m_lst, 0, bar_end)
        # Activate unmeasured qubits before byproduct corrections
        self.__kron_unmeasured_qubits()
        self.__run_cmd_lst(cmd_c_lst, len(cmd_m_lst), bar_end)

        # The output state's label is messy (e.g. [(2, 0), (0, 1), (1, 3)...]),
        # so we permute the systems in order
        q_output = self.__pattern.output_[1]
        self.__bg_state = permute_systems(self.__status, q_output)
        self.__update()

    @staticmethod
    def __map_qubit_to_row(out_lst):
        r"""Construct a map between the labels of output qubits and the number of rows.

        Returns:
            dict: return a dict representing the correspondence between labels and number of rows.
        """
        return {int(div_str_to_float(qubit[0])): qubit for qubit in out_lst}

    def get_classical_output(self):
        r"""Get the classical output of the MBQC model.

        Returns:
            str or dict: if the users input the measurement patterns, the method returns the bit strings of the
            measurement results of the output qubits which is the same as the results of the circuit based model. The
            qubits haven't been measured fills "?". If the input is graph, return the measurement results of all nodes.
        """
        # If the input is pattern, return the equivalent result as the circuit model
        if self.__pattern is not None:
            width = len(self.__pattern.input_)
            c_output = self.__pattern.output_[0]
            q_output = self.__pattern.output_[1]
            # Acquire the relationship between row number and corresponding output qubit label
            output_lst = c_output + q_output
            row_and_qubit = self.__map_qubit_to_row(output_lst)

            # Obtain the string, with classical outputs denoted as their measurement outcomes
            # and quantum outputs denoted as "?"
            bit_str = [str(self.__outcome[row_and_qubit[i]])
                       if row_and_qubit[i] in c_output else '?'
                       for i in range(width)]
            string = "".join(bit_str)
            return string

        # If the input is graph, return the outcome dictionary
        else:
            return self.__outcome

    def get_history(self):
        r"""Get the information during the MBQC computation process.

        Returns:
            list: the list of the results, including generate graph state, measurement and byproduct correction.
        """
        return self.__history

    def get_quantum_output(self):
        r"""Get the quantum state output of the MBQC model.

        Returns:
            State: the quantum state output of the MBQC model.
        """
        return self.__status


def simulate_by_mbqc(circuit, input_state=None):
    r"""Simulate the quantum circuit by equivalent MBQC model.

    This function transform the quantum circuit to equivalent MBQC models and acquire output equivalent to the circuit
    based model.

    Warning:
        Unlike the ``UAnsatz``, the input ``circuit`` here contains the measurement operations.
        By the way, if the users set ``input_state=None``, the initial state of the MBQC is |+> state.

    Args:
        circuit (Circuit): quantum circuit
        input_state (State, optional): the initial state of quantum circuit, default to :math:`|+\rangle`.

    Returns:
        tuple: contains two elements:

            - str: classical output
            - State: quantum output
    """
    if input_state is not None:
        assert isinstance(input_state, State), "the 'input_state' must be of type 'State'."

    pattern = transpile(circuit)
    mbqc = MBQC()
    mbqc.set_pattern(pattern)
    mbqc.set_input_state(input_state)
    mbqc.run_pattern()
    c_output = mbqc.get_classical_output()
    q_output = mbqc.get_quantum_output()

    # Return the classical and quantum outputs
    return c_output, q_output


def __get_sample_dict(bit_num, mea_bits, samples):
    r"""Make statistics of the sampling results based on the number of qubits and the list of the
        index of the measured qubits.

    Args:
        bit_num (int): number of qubits.
        mea_bits (list): the list of the measured qubits.
        samples (list): the list of the measurement results.

    Returns:
        dict: the statistical results
    """
    sample_dict = {}
    for i in range(2 ** len(mea_bits)):
        str_of_order = bin(i)[2:].zfill(len(mea_bits))
        bit_str = []
        idx = 0
        for j in range(bit_num):
            if j in mea_bits:
                bit_str.append(str_of_order[idx])
                idx += 1
            else:
                bit_str.append('?')
        string = "".join(bit_str)
        sample_dict[string] = 0

    # Count sampling results
    for string in list(set(samples)):
        sample_dict[string] += samples.count(string)
    return sample_dict


def sample_by_mbqc(circuit, input_state=None, plot=False, shots=1024, print_or_not=True):
    r""" Repeatedly run the MBQC model and acquire the distributions of the results.
   
    Warning:
        Unlike the ``UAnsatz``, the input ``circuit`` here contains the measurement operations.
        By the way, if the users set ``input_state=None``, the initial state of the MBQC is |+> state.

    Args:
        circuit (Circuit): quantum circuit
        input_state (State, optional): the initial state of quantum circuit, default to |+>
        plot (bool, optional): the boolean switch of whether plotting the histogram of the sampling results,
        default to ``False``.
        shots (int, optional): the number of samples, default to 1024.
        print_or_not (bool, optional): boolean switch of whether print the sampling results and the progress bar,
        default to open.

    Returns:
        dict: the frequency dict composed of the classical results.
        list: the list contains all the sampling results(both quantum and classical).
    """
    # Initialize
    if shots == 1:
        print_or_not = False
    if print_or_not:
        print("Sampling " + str(shots) + " times." + "\nWill return the sampling results.\r\n")
    width = circuit.get_width()
    mea_bits = circuit.get_measured_qubits()

    # Sampling for "shots" times
    samples = []
    all_outputs = []
    for shot in range(shots):
        if print_or_not:
            print_progress((shot + 1) / shots, "Current Sampling Progress")
        c_output, q_output = simulate_by_mbqc(circuit, input_state)
        samples.append(c_output)
        all_outputs.append([c_output, q_output])

    sample_dict = __get_sample_dict(width, mea_bits, samples)
    if print_or_not:
        print("Sample count " + "(" + str(shots) + " shots)" + " : " + str(sample_dict))
    if plot:
        dict_lst = [sample_dict]
        bar_labels = ["MBQC sample outcomes"]
        title = 'Sampling results (MBQC)'
        xlabel = "Measurement outcomes"
        ylabel = "Distribution"
        plot_results(dict_lst, bar_labels, title, xlabel, ylabel)

    return sample_dict, all_outputs
