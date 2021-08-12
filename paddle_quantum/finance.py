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
Functions and data simulator class of quantum finance
"""

import fastdtw
import numpy as np
from paddle_quantum.utils import Hamiltonian

__all__ = [
    "DataSimulator",
    "portfolio_optimization_hamiltonian",
    "portfolio_diversification_hamiltonian",
    "arbitrage_opportunities_hamiltonian"
]


class DataSimulator:
    r"""用于生成和计算投资组合优化和投资分散化问题要用的数据和相关参数

    """
    def __init__(self, stocks, start=None, end=None):
        r"""构造函数，用于实例化一个 ``DataSimulator`` 对象。

        Args:
            stocks (list): 表示所有可投资股票的名字
            start (datetime): 默认为 ``None``，表示随机生成股票数据时交易日的起始日期
            end (datetime): 默认为 ``None``，表示随机生成股票数据时交易日的结束日期
        """
        self._n = len(stocks)
        self._stocks = stocks

        if start and end:
            self._start = start
            self._end = end

        self._data = []

        self.asset_return_mean = None
        self.asset_return_cov = None

    def set_data(self, data):
        r"""决定实验使用的数据是随机生成的还是用户本地输入的

        Args:
            data (list): 用户输入的股票数据
        """
        if len(data) == self._n:
            self._data = data
        else:
            print("invalid data, data is still empty.")
            self._data = []

    def randomly_generate(self):
        r"""根据开始日期和结束日期随机生成用于实验的股票数据

        Note:
            若要随机生成股票数据，需要以 ``datetime`` 包中的格式指定开始日期和结束日期，如 ``start = datetime.datetime(2016, 1, 1)``
        """

        if self._start and self._end:
            num_days = (self._end - self._start).days
            for _ in self._stocks:
                fluctuation = np.random.standard_normal(num_days)
                fluctuation = np.cumsum(fluctuation)
                data_i = np.random.randint(1, 101, size=1) + fluctuation
                trimmed_data_i = [max(data_i[j], 0) for j in range(num_days)]
                if 0 in trimmed_data_i:
                    zero_ind = trimmed_data_i.index(0)
                    trimmed_data_i = trimmed_data_i[:zero_ind] + [0 for _ in range(num_days - zero_ind)]

                self._data.append(trimmed_data_i)
        else:
            print("Please provide the start time and the end time you want to generate stock data.")

    def get_asset_return_mean_vector(self):
        r"""用于计算所有可投资股票的平均投资回报率

        Returns:
            list: 所有可投资的股票的平均投资回报率
        """
        returns = []
        for i in range(self._n):
            return_i = [self._data[i][j + 1] / self._data[i][j] - 1
                        if self._data[i][j] != 0
                        else np.nan for j in range(len(self._data[i]) - 1)]
            returns.append(return_i)
        self.asset_return_mean = np.mean(returns, axis=1)

        return self.asset_return_mean

    def get_asset_return_covariance_matrix(self):
        r"""用于计算所有可投资股票回报率之间的协方差矩阵

        Returns:
            list: 所有可投资股票回报率之间的协方差矩阵
        """
        returns = []
        for i in range(self._n):
            return_i = [self._data[i][j + 1] / self._data[i][j] - 1
                        if self._data[i][j] != 0
                        else np.nan for j in range(len(self._data[i]) - 1)]
            returns.append(return_i)
        self.asset_return_cov = np.cov(returns)

        return self.asset_return_cov

    def get_similarity_matrix(self):
        r"""计算各股票之间的相似矩阵

        通过动态时间规整算法（Dynamic Time Warping, DTW）计算两股票之间的相似性

        Returns:
            list: 各股票间的相似矩阵
        """
        self.rho = np.zeros((self._n, self._n))
        for i in range(0, self._n):
            self.rho[i, i] = 1
            for j in range(i + 1, self._n):
                curr_rho, _ = fastdtw.fastdtw(self._data[i], self._data[j])
                curr_rho = 1 / curr_rho
                self.rho[i, j] = curr_rho
                self.rho[j, i] = curr_rho

        return self.rho


def portfolio_optimization_hamiltonian(penalty, mu, sigma, q, budget):
    r"""构建投资组合优化问题的哈密顿量

    Args:
        penalty (int): 惩罚参数
        mu (list): 各股票的预期回报率
        sigma (list): 各股票回报率间的协方差矩阵
        q (float): 投资股票的风险
        budget (int): 投资预算, 即要投资的股票数量

    .. math::

        C(x) = q \sum_i \sum_j S_{ji}x_ix_j  - \sum_{i}x_i \mu_i + A \left(B - \sum_i x_i\right)^2


    Hint:
        将布尔变量 :math:`x_i` 映射到哈密顿矩阵上，:math:`x_i \mapsto \frac{I-Z_i}{2}`

    Returns:
        Hamiltonian: 投资组合优化问题的哈密顿量
    """
    n = len(mu)

    H_C_list1 = []
    for i in range(n):
        for j in range(n):
            sigma_ij = sigma[i][j]
            H_C_list1.append([sigma_ij / 4, 'I'])
            if i != j:
                H_C_list1.append([sigma_ij / 4, 'Z' + str(i) + ',Z' + str(j)])
            else:
                H_C_list1.append([sigma_ij / 4, 'I'])
            H_C_list1.append([- sigma_ij / 4, 'Z' + str(i)])
            H_C_list1.append([- sigma_ij / 4, 'Z' + str((j))])
    H_C_list1 = [[q * c, s] for (c, s) in H_C_list1]

    H_C_list2 = []
    for i in range(n):
        H_C_list2.append([- mu[i] / 2, 'I'])
        H_C_list2.append([mu[i] / 2, 'Z' + str(i)])

    H_C_list3 = [[budget ** 2, 'I']]
    for i in range(n):
        H_C_list3.append([- 2 * budget / 2, 'I'])
        H_C_list3.append([2 * budget / 2, 'Z' + str(i)])
        H_C_list3.append([2 / 4, 'I'])
        H_C_list3.append([- 2 / 4, 'Z' + str(i)])
        for ii in range(i):
            H_C_list3.append([2 / 4, 'I'])
            H_C_list3.append([2 / 4, 'Z' + str(i) + ',Z' + str(ii)])
            H_C_list3.append([- 2 / 4, 'Z' + str(i)])
            H_C_list3.append([- 2 / 4, 'Z' + str(ii)])
    H_C_list3 = [[penalty * c, s] for (c, s) in H_C_list3]

    H_C_list = H_C_list1 + H_C_list2 + H_C_list3
    po_hamiltonian = Hamiltonian(H_C_list)

    return po_hamiltonian


def portfolio_diversification_hamiltonian(penalty, rho, q):
    r"""构建投资组合分散化问题的哈密顿量

    Args:
        penalty (int): 惩罚参数
        rho (list): 各股票间的相似矩阵
        q (int): 股票聚类的类别数

    .. math::

        \begin{aligned}
        C_x &= -\sum_{i=1}^{n}\sum_{j=1}^{n}\rho_{ij}x_{ij} + A\left(K- \sum_{j=1}^n y_j \right)^2 + \sum_{i=1}^n A\left(\sum_{j=1}^n 1- x_{ij} \right)^2 \\
            &\quad + \sum_{j=1}^n A\left(x_{jj} - y_j\right)^2 + \sum_{i=1}^n \sum_{j=1}^n A\left(x_{ij}(1 - y_j)\right).\\
        \end{aligned}

    Hint:
        将布尔变量 :math:`x_{ij}` 映射到哈密顿矩阵上，:math:`x_{ij} \mapsto \frac{I-Z_{ij}}{2}`

    Returns:
        Hamiltonian: 投资组合分散化问题的哈密顿量
    """
    n = len(rho)

    H_C_list1 = []
    for i in range(n):
        for j in range(n):
            rho_ij = - rho[i][j]
            H_C_list1.append([rho_ij / 2, 'I'])
            H_C_list1.append([- rho_ij / 2, 'Z' + str(i * n + j)])

    H_C_list2 = [[q ** 2, 'I']]
    for j in range(n):
        H_C_list2.append([- q, 'I'])
        H_C_list2.append([q, 'Z' + str(n ** 2 + j)])
        H_C_list2.append([1 / 2, 'I'])
        H_C_list2.append([- 1 / 2, 'Z' + str(n ** 2 + j)])
        for jj in range(j):
            H_C_list2.append([1 / 2, 'I'])
            H_C_list2.append([1 / 2, 'Z' + str(n ** 2 + j) + ',Z' + str(n ** 2 + jj)])
            H_C_list2.append([- 1 / 2, 'Z' + str(n ** 2 + j)])
            H_C_list2.append([- 1 / 2, 'Z' + str(n ** 2 + jj)])
    H_C_list2 = [[penalty * c, s] for (c, s) in H_C_list2]

    H_C_list3 = []
    for i in range(n):
        H_C_list3.append([1, 'I'])
        for j in range(n):
            H_C_list3.append([- 1, 'I'])
            H_C_list3.append([1, 'Z' + str(i * n + j)])
            H_C_list3.append([1 / 2, 'I'])
            H_C_list3.append([- 1 / 2, 'Z' + str(i * n + j)])
            for jj in range(j):
                H_C_list3.append([1 / 2, 'I'])
                H_C_list3.append([- 1 / 2, 'Z' + str(i * n + j)])
                H_C_list3.append([1 / 2, 'Z' + str(i * n + j) + ',Z' + str(i * n + jj)])
                H_C_list3.append([- 1 / 2, 'Z' + str(i * n + jj)])
    H_C_list3 = [[penalty * c, s] for (c, s) in H_C_list3]

    H_C_list4 = []
    for j in range(n):
        H_C_list4.append([1 / 2, 'I'])
        H_C_list4.append([- 1 / 2, 'Z' + str(j * n + j) + ',Z' + str(n ** 2 + j)])
    H_C_list4 = [[penalty * c, s] for (c, s) in H_C_list4]

    H_C_list5 = []
    for i in range(n):
        for j in range(n):
            H_C_list5.append([1 / 4, 'I'])
            H_C_list5.append([- 1 / 4, 'Z' + str(i * n + j)])
            H_C_list5.append([1 / 4, 'Z' + str(n ** 2 + j)])
            H_C_list5.append([- 1 / 4, 'Z' + str(i * n + j) + ',Z' + str(n ** 2 + j)])
    H_C_list5 = [[penalty * c, s] for (c, s) in H_C_list5]

    H_C_list = H_C_list1 + H_C_list2 + H_C_list3 + H_C_list4 + H_C_list5
    pd_hamiltonian = Hamiltonian(H_C_list)

    return pd_hamiltonian


def arbitrage_opportunities_hamiltonian(g, penalty, n, K):
    r"""构建最佳套利机会问题的哈密顿量

    Args:
        g (networkx.DiGraph): 不同货币市场间转换的图形化表示
        A (int): 惩罚参数
        n (int): 货币种类的数量，即图 g 中的顶点数量
        K (int): 套利回路中包含的顶点数

    .. math::

        C(x) = - P(x) + A\sum_{k=0}^{K-1} \left(1 - \sum_{i=0}^{n-1} x_{i,k}\right)^2 + A\sum_{k=0}^{K-1}\sum_{(i,j)\notin E}x_{i,k}x_{j,k+1}

    Hint:
        将布尔变量 :math:`x_{i,k}` 映射到哈密顿矩阵上，:math:`x_{i,k} \mapsto \frac{I-Z_{i,k}}{2}`

    Returns:
       Hamiltonian: 最佳套利机会问题的哈密顿量
    """
    nodes = list(g.nodes)

    H_C_list1 = []
    for (i, c) in enumerate(nodes):
        for (j, cc) in enumerate(nodes):
            if i != j:
                c_ij = np.log2(g[c][cc]['weight'])
                for t in range(K):
                    H_C_list1.append([c_ij / 4, 'I'])
                    H_C_list1.append([c_ij / 4, 'Z' + str(i * n + t) + ',Z' + str((j * n + (t + 1) % K))])
                    H_C_list1.append([- c_ij / 4, 'Z' + str(i * n + t)])
                    H_C_list1.append([- c_ij / 4, 'Z' + str((j * n + (t + 1) % K))])
    H_C_list1 = [[-c, s] for (c, s) in H_C_list1]

    H_C_list2 = []
    for t in range(K):
        H_C_list2.append([1, 'I'])
        for i in range(n):
            H_C_list2.append([- 2 * 1 / 2, 'I'])
            H_C_list2.append([2 * 1 / 2, 'Z' + str(i * n + t)])
            H_C_list2.append([2 / 4, 'I'])
            H_C_list2.append([- 2 / 4, 'Z' + str(i * n + t)])
            for ii in range(i):
                H_C_list2.append([2 / 4, 'I'])
                H_C_list2.append([2 / 4, 'Z' + str(i * n + t) + ',Z' + str(ii * n + t)])
                H_C_list2.append([- 2 / 4, 'Z' + str(i * n + t)])
                H_C_list2.append([- 2 / 4, 'Z' + str(ii * n + t)])
    H_C_list2 = [[penalty * c, s] for (c, s) in H_C_list2]

    H_C_list3 = []
    for t in range(K):
        for (i, c) in enumerate(nodes):
            for (j, cc) in enumerate(nodes):
                if (c, cc) not in g.edges and c != cc:
                    H_C_list3.append([1 / 4, "I"])
                    H_C_list3.append([- 1 / 4, 'Z' + str(i * n + t)])
                    H_C_list3.append([- 1 / 4, 'Z' + str((j * n + (t + 1) % K))])
                    H_C_list3.append([- 1 / 4, 'Z' + str(i * n + t) + ',Z' + str((j * n + (t + 1) % K))])
    H_C_list3 = [[penalty * c, s] for (c, s) in H_C_list3]

    H_C_list4 = []
    for i in range(n):
        H_C_list4.append([1, 'I'])
        for t in range(K):
            H_C_list4.append([- 2 * 1 / 2, 'I'])
            H_C_list4.append([2 * 1 / 2, 'Z' + str(i * n + t)])
            H_C_list4.append([2 / 4, 'I'])
            H_C_list4.append([- 2 / 4, 'Z' + str(i * n + t)])
            for tt in range(t):
                H_C_list4.append([2 / 4, 'I'])
                H_C_list4.append([2 / 4, 'Z' + str(i * n + t) + ',Z' + str(i * n + tt)])
                H_C_list4.append([- 2 / 4, 'Z' + str(i * n + t)])
                H_C_list4.append([- 2 / 4, 'Z' + str(i * n + tt)])
    H_C_list4 = [[penalty * c, s] for (c, s) in H_C_list4]

    H_C_list = H_C_list1 + H_C_list2 + H_C_list3 + H_C_list4
    ao_hamiltonian = Hamiltonian(H_C_list)

    return ao_hamiltonian
