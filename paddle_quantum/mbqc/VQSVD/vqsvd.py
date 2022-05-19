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
Use MBQC to simulate VQSVD circuit and compare with the circuit model
"""

from paddle import to_tensor
from paddle_quantum.mbqc.utils import print_progress, write_running_data, read_running_data, plot_results
from time import perf_counter
from paddle_quantum.mbqc.simulator import simulate_by_mbqc, sample_by_mbqc
from numpy import random
import paddle_quantum
from paddle_quantum.mbqc.qobject import Circuit

__all__ = [
    "vqsvd_circuit",
    "cir_uansatz",
    "cir_mbqc",
    "vqsvd",
    "compare_time",
    "compare_result"
]


def vqsvd_circuit(cir, alpha):
    r"""输入 VQSVD 算法的电路。

    Args:
        cir (Circuit or UAnsatz): Circuit 为 MBQC 电路模块中的类，UAnsatz 为量桨平台 UAnsatz 电路模型中的类，
                                 二者都具有相同的量子门输入方式
        alpha (Tensor): 旋转门的角度

    Returns:
        Circuit / UAnsatz: 输入了量子门信息的电路
    """
    width = alpha.shape[0]
    depth = alpha.shape[1]

    if isinstance(cir, Circuit):
        for layer_num in range(depth):
            for which_qubit in range(width):
                cir.ry(alpha[which_qubit, layer_num], which_qubit)
            for which_qubit in range(width - 1):
                cir.cnot([which_qubit, which_qubit + 1])
        cir.measure()
    else:
        for which_qubit in range(width):
            cir.h(which_qubit)
        for layer_num in range(depth):
            for which_qubit in range(width):
                cir.ry(which_qubit, param=alpha[which_qubit, layer_num])
            for which_qubit in range(width - 1):
                cir.cnot([which_qubit, which_qubit + 1])

    return cir


def cir_uansatz(alpha, shots=1):
    r"""定义 VQSVD 算法的 UAnsatz 电路模型。

    Args:
        alpha (Tensor): 旋转门的角度
        shots (int, optional): 重复次数

    Returns:
        list: 列表中的第一项为测量结果或运算后得到的量子态，第二项为电路模拟方式运行 VQSVD 算法需要的时间
    """
    qubit_number = alpha.shape[0]

    # If bit_num >= 25, the UAnsatz could not support it
    if qubit_number >= 25:
        raise ValueError("the UAnsatz model could not support qubit number larger than 25 on a laptop.")

    else:
        # Input information of circuits
        cir = paddle_quantum.ansatz.Circuit(qubit_number)
        cir = vqsvd_circuit(cir, alpha)

        # Start running
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
    r"""定义 VQSVD 算法的 MBQC 翻译模拟方式。

    Args:
        alpha (Tensor): 旋转门的角度

    Returns:
        list: 列表中的第一项为测量结果或运算后得到的量子态，第二项为依据行序优先算法优化翻译 MBQC 模型模拟 VQSVD 电路需要的时间
    """
    qubit_number = alpha.shape[0]

    # Run MBQC
    cir = Circuit(qubit_number)
    cir = vqsvd_circuit(cir, alpha)
    start_time = perf_counter()
    # No sampling
    if shots == 1:
        simulate_by_mbqc(cir)
        sample_outcomes = {}
    # Sampling
    else:
        sample_outcomes, all_output = sample_by_mbqc(cir, shots=shots, print_or_not=True)
    end_time = perf_counter()
    result_list = [sample_outcomes, end_time - start_time]
    return result_list


def vqsvd(start_width, end_width):
    r"""定义 VQSVD 算法的主函数。

    在主函数中，我们比较了在不同宽度下 UAnsatz 电路模型模拟 VQSVD 和基于行序优先原则算法翻译为 MBQC 模型模拟 VQSVD 的时间。

    Args:
        start_width (int): 电路的起始宽度
        end_width (int): 电路的终止宽度
    """
    # Initialize
    depth = 2  # Set the depth of circuit to 2
    time_text = open("record_time.txt", 'w')
    all_width = list(range(start_width, end_width))

    # Start running VQSVD under different qubit numbers
    counter = 0
    for width in all_width:
        eg = "VQSVD with " + str(width) + " X " + str(depth) + " size."
        print_progress((counter + 1) / len(all_width), "Current Plot Progress")
        counter += 1

        alpha = random.randn(width, depth)
        alpha_tensor = to_tensor(alpha, dtype='float64')
        mbqc_result, mbqc_time = cir_mbqc(alpha_tensor)
        uansatz_result, uansatz_time = cir_uansatz(alpha_tensor)

        write_running_data(time_text, eg, width, mbqc_time, uansatz_time)

    time_text.close()


def compare_time(start_width, end_width):
    r"""定义 VQSVD 的画图函数。

    此函数用于将 UAnsatz 电路模型模拟运行 VQSVD 的时间成本和基于行序优先原则算法翻译为 MBQC 模型模拟的时间画出来。
    """
    time_comparison = read_running_data("record_time.txt")
    bar_labels = ['MBQC', 'UAnsatz']
    xticklabels = list(range(start_width, end_width))
    title = 'VQSVD Example: Time comparison between MBQC and UAnsatz'
    xlabel = 'Circuit width'
    ylabel = 'Running time (s)'
    plot_results(time_comparison, bar_labels, title, xlabel, ylabel, xticklabels)


def compare_result(qubit_number, shots=1024):
    r"""定义 VQSVD 获得最终比特串的函数。

    此函数的调用是为了获得 VQSVD 输出的最终比特串对应的字符。
    对于量子电路模型，只需要获得演化后的量子态，并重复对该量子态进行测量多次，就可以获得输出比特串的概率分布。
    MBQC 模型则与之不同，前文中所有 MBQC 模型的 VQSVD 都是仅运算一次得到的输出比特串，而并没有输出演化后量子态的信息。
    因此，为了获得输出比特串的概率分布，我们需要对 MBQC 模型的整个过程重复执行多次，
    统计这些次数中各个比特串出现的频率，从而用频率信息估算概率信息。

    Note:
        由于采样的次数有限，统计出的输出比特串频率分布会稍有偏差。

    Args:
        qubit_number (int): 比特数
        shots (int, optional): 采样次数
    """
    # Initialize
    depth = 2  # Set the depth of circuit to 2
    CRED = '\033[91m'
    CEND = '\033[0m'
    alpha = random.randn(qubit_number, depth)
    alpha_tensor = to_tensor(alpha, dtype="float64")

    mbqc_result, mbqc_time = cir_mbqc(alpha_tensor, shots)
    uansatz_result, uansatz_time = cir_uansatz(alpha_tensor, shots)

    print(CRED + "MBQC sampling results:" + CEND, mbqc_result, "s")
    print(CRED + "UAnsatz sampling results:" + CEND, uansatz_result, "s")

    result_comparison = [mbqc_result, uansatz_result]
    bar_labels = ['MBQC', 'UAnsatz']
    title = "VQSVD Example: Comparison btween MBQC and UAnsatz"
    xlabel = "Measurement outcomes"
    ylabel = "Distribution"
    plot_results(result_comparison, bar_labels, title, xlabel, ylabel)
