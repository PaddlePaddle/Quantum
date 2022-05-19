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
Use MBQC to simulate Google quantum GRCS circuits and compare with Qiskit simulator.
To prove our simulation advantage in quantum shallow circuit simulation
"""

import os
from time import perf_counter
from paddle_quantum.mbqc.utils import plot_results, write_running_data, read_running_data
from paddle_quantum.mbqc.qobject import Circuit
from paddle_quantum.mbqc.simulator import simulate_by_mbqc
from paddle import to_tensor
from numpy import pi
from qiskit import QuantumCircuit, transpile, Aer

__all__ = [
    "grcs_circuit",
    "cir_mbqc",
    "cir_qiskit",
    "grcs",
    "compare_time"
]


def grcs_circuit(input_cir, cir):
    r"""定义 GRCS 电路。

    Args:
        input_cir (list): 根据 ``".txt"`` 文件处理后得到的输入电路的列表，列表中记录了电路信息
        cir (Circuit or QuantumCircuit): Circuit 为 MBQC 电路模块中的类，QuantumCircuit 为 "Qiskit" 电路模型中的类，
                                        二者都具有类似的量子门输入方式

    Returns:
        float: MBQC 模型下运行电路需要的时间
    """
    half_pi = pi / 2
    half_pi_tensor = to_tensor([pi / 2], dtype='float64')
    for gate in input_cir[1:]:

        # Qiskit input
        if not isinstance(cir, Circuit):
            if gate[1] == 'h':
                cir.h(int(gate[2]))
            elif gate[1] == 'cz':
                cir.cz(int(gate[2]), int(gate[3]))
            elif gate[1] == 't':
                cir.t(int(gate[2]))
            elif gate[1] == 'x_1_2':
                cir.rx(half_pi, int(gate[2]))
            elif gate[1] == 'y_1_2':
                cir.ry(half_pi, int(gate[2]))

        # MBQC input
        # Note: MBQC model start from plus state by default, so we can omit the first layer of Hadamard gates
        # in the circuit; We can also merge the last layer of Hadamard gates with the final Z measurements
        # and replace them by X measurements.
        else:
            if gate[1] == 'cz':
                cir.cz([int(gate[2]), int(gate[3])])
            elif gate[1] == 't':
                cir.t(int(gate[2]))
            elif gate[1] == 'x_1_2':
                cir.rx(half_pi_tensor, int(gate[2]))
            elif gate[1] == 'y_1_2':
                cir.ry(half_pi_tensor, int(gate[2]))
    return cir


def cir_mbqc(input_cir):
    r"""运用 MBQC 模型模拟谷歌量子霸权电路并运算计时。

    Args:
        input_cir (list): 根据 ``".txt"`` 文件处理后得到的输入电路的列表，列表中记录了电路信息

    Returns:
        float: MBQC 模型下运行电路需要的时间
    """
    qubit_number = int(input_cir[0][0])
    cir = Circuit(qubit_number)
    cir = grcs_circuit(input_cir, cir)

    x_measurement = [to_tensor([0], dtype='float64'), 'XY', [], []]
    for output_qubit in range(qubit_number):
        cir.measure(output_qubit, x_measurement)

    mbqc_start_time = perf_counter()
    simulate_by_mbqc(cir)
    mbqc_end_time = perf_counter()
    return mbqc_end_time - mbqc_start_time


def cir_qiskit(input_cir):
    r""" 运用 Qiskit 模拟器模拟谷歌量子霸权电路并运算计时。

    Args:
        input_cir (list): 根据 ``".txt"`` 文件处理后得到的输入电路的列表，列表中记录了电路信息

    Returns:
        float: Qiskit 模拟谷歌量子霸权电路需要的时间
    """
    qubit_number = int(input_cir[0][0])
    cir = QuantumCircuit(qubit_number, qubit_number)
    cir = grcs_circuit(input_cir, cir)
    cir.measure(list(range(qubit_number)), list(range(qubit_number)))

    # Set the simulator
    # Note: We can choose either "aer_simulator_statevector" or "aer_simulator_matrix_product_state" as simulator
    # and choose the minimum time of Qiskit simulation to compare with MBQC

    # We use "aer_simulator_statevector"  for those circuits with qubits number lower than 25.
    if qubit_number <= 25:
        which_simulator = 'aer_simulator_statevector'
    # And use "aer_simulator_matrix_product_state" for for those circuits with qubits number higher than 25
    else:
        which_simulator = 'aer_simulator_matrix_product_state'

    simulator = Aer.get_backend(which_simulator)
    compiled_circuit = transpile(cir, simulator)

    # Run with Qiskit
    qiskit_start_time = perf_counter()
    job = simulator.run(compiled_circuit, shots=1)
    result = job.result()
    result.get_counts(0)
    qiskit_end_time = perf_counter()

    return qiskit_end_time - qiskit_start_time


def grcs():
    r"""定义 GRCS 主函数。

    主函数分别调用两个模拟方式来模拟测试用例，并写入运行时间。

    Note:
        我们选取的电路为谷歌量子霸权电路图 [https://github.com/sboixo/GRCS] 中的部分浅层量子电路，
        对于深层的量子电路，MBQC 的模拟思路依然存在计算瓶颈。
        在使用 Qiskit 进行模拟时，我们选取了 "aer_simulator_statevector" 和 "aer_simulator_matrix_product_state" 中时间较短
        的模拟器作为 Qiskit 对电路的运行时间，与 MBQC 模拟器运行时间对比。
    """
    # Initialize
    CRED = '\033[91m'
    CEND = '\033[0m'
    egs = os.listdir(os.path.dirname(__file__) + "/example/rectangular/depth10")
    time_text = open("record_time.txt", 'w')

    # Start examples
    for eg in egs:

        input_cir = []
        with open(os.path.dirname(__file__) + "/example/rectangular/depth10/" + eg, 'r') as file:
            for line in file:
                input_cir.append(list(line.strip('\n').split(' ')))
        width = input_cir[0][0]

        mbqc_time = cir_mbqc(input_cir)
        qiskit_time = cir_qiskit(input_cir)

        print(CRED + "The current example is:" + CEND, eg)
        print("The qubit number is: " + width)
        print(CRED + "MBQC running time is:" + CEND, mbqc_time, "s")
        print(CRED + "Qiskit running time is:" + CEND, qiskit_time, "s")
        print("--------------------------------------------------------------------------------------------")

        write_running_data(time_text, eg, width, mbqc_time, qiskit_time)

    time_text.close()


def compare_time():
    r"""定义画图函数。

    画图函数读取测试用例的时间，并用 matplotlib 画出来。
    """
    time_comparison = read_running_data("record_time.txt")
    bar_labels = ['MBQC', 'Qiskit']
    title = "GRCS Example: Time comparison between MBQC and Qiskit"
    xlabel = "Index of test example"
    ylabel = "Running time (s)"
    plot_results(time_comparison, bar_labels, title, xlabel, ylabel)
