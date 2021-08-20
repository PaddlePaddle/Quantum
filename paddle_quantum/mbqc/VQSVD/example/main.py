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

from paddle_quantum.mbqc.VQSVD.vqsvd import vqsvd, compare_time, compare_result


def main():
    r"""VQSVD 主函数。
    """
    # From 3 to 24, plot time
    start_qubit = 3
    end_qubit = 24
    vqsvd(start_qubit, end_qubit)
    compare_time(start_qubit, end_qubit)

    # Compare the probability distribution of outcome bit strings by multiple samplings
    compare_result(3, 1024)


if __name__ == '__main__':
    main()
