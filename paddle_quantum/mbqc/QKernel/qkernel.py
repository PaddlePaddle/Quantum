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
Use MBQC to simulate the circuit in quantum Kernel method and compare with the circuit model
"""

from paddle import to_tensor
from paddle_quantum.mbqc.utils import print_progress, plot_results
from paddle_quantum.mbqc.utils import write_running_data, read_running_data
from time import perf_counter
from paddle_quantum.mbqc.simulator import simulate_by_mbqc, sample_by_mbqc
from numpy import random
import paddle_quantum
from paddle_quantum.mbqc.qobject import Circuit

__all__ = [
    "qkernel_circuit",
    "cir_uansatz",
    "cir_mbqc",
    "qkernel",
    "compare_time",
    "compare_result"
]


def qkernel_circuit(alpha, cir):
    r"""输入量子核方法的电路。

    Args:
        alpha (Tensor): 旋转门的角度
        cir (Circuit or UAnsatz): Circuit 或者 UAnsatz 的实例

    Returns:
        Circuit / UAnsatz: 构造的量子电路
    """
    qubit_number = alpha.shape[0]

    if isinstance(cir, Circuit):
        # U
        for i in range(qubit_number):
            if not isinstance(cir, Circuit):
                cir.h(i)
            cir.rx(alpha[i, 1], i)
            cir.rz(alpha[i, 2], i)
            cir.rx(alpha[i, 3], i)
        # cz
        for i in range(qubit_number - 1):
            cir.h(i + 1)
            cir.cnot([i, i + 1])
            cir.h(i + 1)

        # U^{\dagger}
        for i in range(qubit_number):
            cir.rx(alpha[i, 5], i)
            cir.rz(alpha[i, 6], i)
            cir.rx(alpha[i, 7], i)
            cir.h(i)
        cir.measure()
    else:
        # U
        for i in range(qubit_number):
            if not isinstance(cir, Circuit):
                cir.h(i)
            cir.rx(i, param=alpha[i, 1])
            cir.rz(i, param=alpha[i, 2])
            cir.rx(i, param=alpha[i, 3])
        # cz
        for i in range(qubit_number - 1):
            cir.h(i + 1)
            cir.cnot([i, i + 1])
            cir.h(i + 1)

        # U^{\dagger}
        for i in range(qubit_number):
            cir.rx(i, param=alpha[i, 5])
            cir.rz(i, param=alpha[i, 6])
            cir.rx(i, param=alpha[i, 7])
            cir.h(i)

    return cir


def cir_uansatz(alpha, shots=1):
    r"""定义量子核方法的 UAnsatz 电路模型。

    Args:
        alpha (Tensor): 旋转门的角度
        shots (int): 重复次数

    Returns:
        list: 列表中的第一项为测量结果，第二项为 UAnsatz 电路模型模拟方式运行量子核方法需要的时间
    """
    qubit_number = alpha.shape[0]

    # If bit_num >= 25, the UAnsatz could not support it
    if qubit_number >= 25:
        raise ValueError("the UAnsatz model could not support qubit number larger than 25 on a laptop.")

    else:
        cir = paddle_quantum.ansatz.Circuit(qubit_number)
        cir = qkernel_circuit(alpha, cir)

        uansatz_start_time = perf_counter()
        state = cir()
        outcome = state.measure(shots=shots)
        uansatz_end_time = perf_counter()

        # As the outcome dictionary is in a messy order, we need to reorder the outcome
        outcome_in_order = {bin(i)[2:].zfill(qubit_number): outcome[bin(i)[2:].zfill(qubit_number)]
        if bin(i)[2:].zfill(qubit_number) in list(outcome.keys())
        else 0 for i in range(2 ** qubit_number)}

        result_and_time = [outcome_in_order, uansatz_end_time - uansatz_start_time]
        return result_and_time


def cir_mbqc(alpha, shots=1):
    r"""定义量子核方法的 MBQC 翻译电路模型。

    Args:
        alpha (Tensor): 旋转门的角度
        shots (int): 重复次数

    Returns:
        list: 列表中的第一项为测量结果，第二项为依据行序优先算法优化翻译 MBQC 模型模拟量子核方法电路需要的时间
    """
    # Input circuit
    width = alpha.shape[0]
    cir = Circuit(width)
    cir = qkernel_circuit(alpha, cir)
    # Start running
    start_time = perf_counter()
    # No sampling
    if shots == 1:
        simulate_by_mbqc(cir)
        sample_outcomes = {}
    # Sampling
    else:
        sample_outcomes, all_output = sample_by_mbqc(cir, shots=shots, print_or_not=True)
    end_time = perf_counter()
    result_and_time = [sample_outcomes, end_time - start_time]
    return result_and_time


def qkernel(start_width, end_width):
    r"""定义量子核方法的主函数。

    在主函数中，我们比较了在不同宽度下 UAnsatz 电路模型模拟量子核方法和基于行序优先原则算法翻译为 MBQC 模型模拟量子核方法的时间。

    Args:
        start_width (int): 电路的起始宽度
        end_width (int): 电路的终止宽度
    """
    # Initialize
    depth = 9  # Set the depth of circuit to 9
    time_text = open("record_time.txt", 'w')
    all_width = list(range(start_width, end_width))

    # Start running Kernel under different qubit numbers
    counter = 0
    for width in all_width:
        eg = "Quantum kernel method with " + str(width) + " X " + str(depth) + " size."
        print_progress((counter + 1) / len(all_width), "Current Plot Progress")
        counter += 1

        alpha = random.randn(width, depth)
        alpha_tensor = to_tensor(alpha, dtype='float64')
        mbqc_result, mbqc_time = cir_mbqc(alpha_tensor)
        uansatz_result, uansatz_time = cir_uansatz(alpha_tensor)

        write_running_data(time_text, eg, width, mbqc_time, uansatz_time)

    time_text.close()


def compare_time(start_width, end_width):
    r"""定义量子核方法的画图函数。

    此函数用于将 UAnsatz 电路模型模拟运行量子核方法的时间成本和基于行序优先原则算法翻译为 MBQC 模型模拟的时间画出来。
    """
    time_comparison = read_running_data("record_time.txt")
    bar_labels = ["MBQC", "UAnsatz"]
    xticklabels = list(range(start_width, end_width))
    title = "Kernel Example: Time comparison between MBQC and UAnsatz"
    xlabel = "Circuit width"
    ylabel = "Running time (s)"
    plot_results(time_comparison, bar_labels, title, xlabel, ylabel, xticklabels)


def compare_result(qubit_number, shots=1024):
    r"""定义量子核方法获得最终比特串的函数。

    此函数的调用是为了获得量子核方法输出的最终比特串对应的字符。
    对于量子电路模型，只需要获得演化后的量子态，并重复对该量子态进行多次测量，就可以获得输出比特串的概率分布。
    MBQC 模型则与之不同，前文中所有 MBQC 模型的量子核方法都是仅运算一次得到的输出比特串，而并没有输出演化后量子态的信息。
    因此，为了获得输出比特串的概率分布，我们需要对 MBQC 模型的整个过程重复执行多次，
    统计这些次数中各个比特串出现的频率，从而用频率信息估算概率信息。

    Note:
        由于采样的次数有限，统计出的输出比特串频率分布会稍有偏差。

    Args:
        qubit_number (int): 比特数
        shots (int, optional): 采样次数
    """
    # Initialize
    depth = 9
    CRED = '\033[91m'
    CEND = '\033[0m'
    alpha = random.randn(qubit_number, depth)
    alpha_tensor = to_tensor(alpha, dtype="float64")

    mbqc_result, mbqc_time = cir_mbqc(alpha_tensor, shots)
    uansatz_result, uansatz_time = cir_uansatz(alpha_tensor, shots)

    print(CRED + "MBQC sampling results:" + CEND, mbqc_result)
    print(CRED + "UAnsatz sampling results:" + CEND, uansatz_result)

    result_comparison = [mbqc_result, uansatz_result]
    bar_labels = ["MBQC", "UAnsatz"]
    title = "Kernel Example: Comparison between MBQC and UAnsatz"
    xlabel = "Measurement outcomes"
    ylabel = "Distribution"
    plot_results(result_comparison, bar_labels, title, xlabel, ylabel)
