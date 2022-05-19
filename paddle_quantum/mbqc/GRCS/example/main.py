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
main
"""

from paddle_quantum.mbqc.GRCS.supremacy import grcs, compare_time


def main():
    r"""量子霸权电路主函数。
    """
    grcs()  # Run GRCS circuits
    compare_time()  # Plot the running time of MBQC and Qiskit


if __name__ == '__main__':
    main()
