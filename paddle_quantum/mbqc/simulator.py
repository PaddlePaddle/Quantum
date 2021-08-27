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
此模块包含构造 MBQC 模型的常用类和配套的运算模拟工具。
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
    r"""定义基于测量的量子计算模型 ``MBQC`` 类。

    用户可以通过实例化该类来定义自己的 MBQC 模型。
    """

    def __init__(self):
        r"""MBQC 类的构造函数，用于实例化一个 ``MBQC`` 对象。
        """
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
        r"""定义维护点列表，用于实例化一个 ``Vertex`` 对象。

        将 MBQC 算法中图的节点分为三类，并进行动态维护。

        Note:
            这是内部类，用户不需要直接调用到该类。

        Attributes:
            total (list): MBQC 算法中图上的全部节点，不随运算而改变
            pending (list): 待激活的节点，随着运算的执行而逐渐减少
            active (list): 激活的节点，与当前测量步骤直接相关的节点
            measured (list): 已被测量过的节点，随着运算的执行而逐渐增加
        """

        def __init__(self, total=None, pending=None, active=None, measured=None):
            r"""``Vertex`` 类的构造函数，用于实例化一个 ``Vertex`` 对象。

            Args:
                total (list): MBQC 算法中图上的全部节点，不随运算而改变
                pending (list): 待激活的节点，随着运算的执行而逐渐减少
                active (list): 激活的节点，与当前测量步骤直接相关的节点
                measured (list): 已被测量过的节点，随着运算的执行而逐渐增加
            """
            self.total = [] if total is None else total
            self.pending = [] if pending is None else pending
            self.active = [] if active is None else active
            self.measured = [] if measured is None else measured

    def set_graph(self, graph):
        r"""设置 MBQC 模型中的图。

        该函数用于将用户自己构造的图传递给 ``MBQC`` 实例。
        
        Args:
            graph (list): MBQC 模型中的图，由列表 ``[V, E]`` 给出， 其中 ``V`` 为节点列表，``E`` 为边列表
        """
        vertices, edges = graph

        vertices_of_edges = set([vertex for edge in edges for vertex in list(edge)])
        assert vertices_of_edges.issubset(vertices), "edge must be between the graph vertices."

        self.__graph = Graph()
        self.__graph.add_nodes_from(vertices)
        self.__graph.add_edges_from(edges)

        self.vertex = self.Vertex(total=vertices, pending=vertices, active=[], measured=[])

    def get_graph(self):
        r"""获取图的信息。

        Returns:
            nx.Graph: 图
        """
        return self.__graph

    def set_pattern(self, pattern):
        r"""设置 MBQC 模型的测量模式。

        该函数用于将用户由电路图翻译得到或自己构造的测量模式传递给 ``MBQC`` 实例。

        Warning:
            输入的 pattern 参数是 ``Pattern`` 类型，其中命令列表为标准 ``EMC`` 命令。

        Args:
            pattern (Pattern): MBQC 算法对应的测量模式
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
        r"""获取测量模式的信息。

        Returns:
            Pattern: 测量模式
        """
        return self.__pattern

    def set_input_state(self, state=None):
        r"""设置需要替换的输入量子态。

        Warning:
            与电路模型不同，MBQC 模型通常默认初始态为加态。如果用户不调用此方法设置初始量子态，则默认为加态。
            如果用户以测量模式运行 MBQC，则此处输入量子态的系统标签会被限制为从零开始的自然数，类型为整型。

        Args:
            state (State): 需要替换的量子态，默认为加态
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
        r"""设置动态过程图绘制时节点的位置坐标。

        Note:
            这是内部方法，用户并不需要直接调用到该方法。

        Args:
            pos (dict or bool, optional): 节点坐标的字典数据或者内置的坐标选择，
                                          内置的坐标选择有：``True`` 为测量模式自带的坐标，``False`` 为 ``spring_layout`` 坐标
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
        r"""根据当前节点状态绘图，用以实时展示 MBQC 模型的模拟计算过程。

        Note:
            这是内部方法，用户并不需要直接调用到该方法。

        Args:
            which_process (str): MBQC 执行的阶段，"measuring", "active" 或者 "measured"
            which_qubit (any): 当前关注的节点，可以是 ``str``, ``tuple`` 等任意数据类型，但需要和图的标签类型匹配
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
        r"""动态过程图绘制，用以实时展示 MBQC 模型的模拟计算过程。

        Args:
            draw (bool, optional): 是否绘制动态过程图的布尔开关
            pos (bool or dict, optional): 节点坐标的字典数据或者内置的坐标选择，内置的坐标选择有：
                                            ``True`` 为测量模式自带的坐标，``False`` 为 `spring_layout` 坐标
            pause_time (float, optional): 绘制动态过程图时每次更新的停顿时间
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
        r""" 显示 MBQC 模型运行进度的开关。

        Args:
            track (bool, optional): ``True`` 打开进度条显示功能， ``False`` 关闭进度条显示功能
        """
        assert isinstance(track, bool), "the parameter 'track' must be bool."
        self.__track = track

    def __apply_cz(self, which_qubits_list):
        r"""对给定的两个比特作用控制 Z 门。

        Note:
            这是内部方法，用户并不需要直接调用到该方法。

        Warning:
            作用控制 Z 门的两个比特一定是被激活的。

        Args:
            which_qubits_list (list): 作用控制 Z 门的比特对标签列表，例如 ``[(1, 2), (3, 4),...]``
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
        r"""对给定的单比特作用 Pauli 门。

        Note:
            这是内部方法，用户并不需要直接调用到该方法。

        Args:
            gate (str): Pauli 门的索引字符，"I", "X", "Y", "Z" 分别表示对应的门，在副产品处理时用 "X" 和 "Z" 门
            which_qubit (any): 作用 Pauli 门的系统标签，
                               可以是 ``str``, ``tuple`` 等任意数据类型，但需要和 MBQC 模型中节点的标签类型匹配
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
        r"""以待测量的比特为输入参数，生成测量当前节点所需要的最小的量子图态。

        Note:
            这是内部方法，用户并不需要直接调用到该方法。

        Args:
            which_qubit (any): 待测量比特的系统标签。
                                可以是 ``str``, ``tuple`` 等任意数据类型，但需要和 MBQC 模型中节点的标签类型匹配
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
        r"""更新历史列表和量子态信息。
        """
        self.__history.append(self.__bg_state)
        self.__status = self.__history[-1]

    def measure(self, which_qubit, basis_list):
        r"""以待测量的比特和测量基为输入参数，对该比特进行测量。

        Note:
            这是用户在实例化 MBQC 类之后最常调用的方法之一，此处我们对单比特测量模拟进行了最大程度的优化，
            随着用户对该函数的调用，MBQC 类将自动完成激活相关节点、生成所需的图态以及对特定比特进行测量的全过程，
            并记录测量结果和对应测量后的量子态。用户每调用一次该函数，就完成一次对单比特的测量操作。

        Warning:
            当且仅当用户调用 ``measure`` 类方法时，MBQC 模型才真正进行运算。

        Args:
            which_qubit (any): 待测量量子比特的系统标签，
                                可以是 ``str``, ``tuple`` 等任意数据类型，但需要和 MBQC 模型的图上标签匹配
            basis_list (list): 测量基向量构成的列表，列表元素为 ``Tensor`` 类型的列向量

        代码示例：

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
        r"""根据输入的量子系统标签，在存储测量结果的字典中找到对应的测量结果，并进行求和。

        Note:
            在进行副产品纠正操作和定义适应性测量角度时，用户可以调用该方法对特定比特的测量结果求和。

        Args:
            which_qubits (list): 需要查找测量结果并求和的比特的系统标签列表
            start (int): 对结果进行求和后需要额外相加的整数

        Returns:
            int: 指定比特的测量结果的和

        代码示例：

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
        r"""对测量后的量子态进行副产品纠正。

        Note:
            这是用户在实例化 MBQC 类并完成测量后，经常需要调用的一个方法。

        Args:
            gate (str): ``'X'`` 或者 ``'Z'``，分别表示 Pauli X 或 Z 门修正
            which_qubit (any): 待操作的量子比特的系统标签，可以是 ``str``, ``tuple`` 等任意数据类型，但需要和 MBQC 中图的标签类型匹配
            power (int): 副产品纠正算符的指数

        代码示例：

            此处展示的是 MBQC 模型下实现隐形传态的一个例子。

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
        r"""执行测量或副产品处理命令。

        Args:
            cmd (Pattern.CommandM / Pattern.CommandX / Pattern.CommandZ): 测量或副产品处理命令
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
        r"""对列表执行测量或副产品处理命令。

        Args:
            cmd_lst (list): 命令列表，包含测量或副产品处理命令
            bar_start (int): 进度条的开始点
            bar_end (int): 进度条的结束点
        """
        for i in range(len(cmd_lst)):
            cmd = cmd_lst[i]
            self.__run_cmd(cmd)
            print_progress((bar_start + i + 1) / bar_end, "Pattern Running Progress", self.__track)

    def __kron_unmeasured_qubits(self):
        r"""该方法将没有被作用 CZ 纠缠的节点初始化为 |+> 态，并与当前的量子态做张量积。

        Warning:
            该方法仅在用户输入测量模式时调用，当用户输入图时，如果节点没有被激活，我们默认用户没有对该节点进行任何操作。
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
        r"""按照设置的测量模式对 MBQC 模型进行模拟。

        Warning:
            该方法必须在 ``set_pattern`` 调用后调用。
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
        r"""将输出比特的标签与行数对应起来，便于查找其对应关系。

        Returns:
            dict: 返回字典，代表行数与标签的对应关系
        """
        return {int(div_str_to_float(qubit[0])): qubit for qubit in out_lst}

    def get_classical_output(self):
        r"""获取 MBQC 模型运行后的经典输出结果。

        Returns:
            str or dict: 如果用户输入是测量模式，则返回测量输出节点得到的比特串，与原电路的测量结果相一致，没有被测量的比特位填充 "？"，如果用户输入是图，则返回所有节点的测量结果
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
        r"""获取 MBQC 计算模拟时的中间步骤信息。

        Returns:
            list: 生成图态、进行测量、纠正副产品后运算结果构成的列表
        """
        return self.__history

    def get_quantum_output(self):
        r"""获取 MBQC 模型运行后的量子态输出结果。

        Returns:
            State: MBQC 模型运行后的量子态
        """
        return self.__status


def simulate_by_mbqc(circuit, input_state=None):
    r"""使用等价的 MBQC 模型模拟量子电路。

    该函数通过将量子电路转化为等价的 MBQC 模型并运行，从而获得等价于原始量子电路的输出结果。

    Warning:
        与 ``UAnsatz`` 不同，此处输入的 ``circuit`` 参数包含了测量操作。
        另，MBQC 模型默认初始态为加态，因此，如果用户不输入参数 ``input_state`` 设置初始量子态，则默认为加态。

    Args:
        circuit (Circuit): 量子电路图
        input_state (State, optional): 量子电路的初始量子态，默认为 :math:`|+\rangle` 态

    Returns:
        tuple: 包含如下两个元素:

            - str: 经典输出
            - State: 量子输出
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
    r"""根据比特数和测量比特索引的列表，统计采样结果。

    Args:
        bit_num (int): 比特数
        mea_bits (list): 测量的比特列表
        samples (list): 采样结果

    Returns:
        dict: 统计得到的采样结果
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
    r"""将 MBQC 模型重复运行多次，获得经典结果的统计分布。
   
    Warning:
        与 ``UAnsatz`` 不同，此处输入的 circuit 参数包含了测量操作。
        另，MBQC 模型默认初始态为加态，因此，如果用户不输入参数 `input_state` 设置初始量子态，则默认为加态。

    Args:
        circuit (Circuit): 量子电路图
        input_state (State, optional): 量子电路的初始量子态，默认为加态
        plot (bool, optional): 绘制经典采样结果的柱状图开关，默认为关闭状态
        shots (int, optional): 采样次数，默认为 1024 次
        print_or_not (bool, optional): 是否打印采样结果和绘制采样进度，默认为开启状态

    Returns:
        dict: 经典结果构成的频率字典
        list: 经典测量结果和所有采样结果（包括经典输出和量子输出）的列表
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
