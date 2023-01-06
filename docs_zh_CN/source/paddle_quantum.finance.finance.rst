paddle\_quantum.finance.finance
===============================

量子金融的相关函数和模拟器类。

.. py:class:: DataSimulator(stocks, start=None, end=None)

   基类: :py:class:`object`

   用于生成和计算投资组合优化和投资分散化问题要用的数据和相关参数。

   :param stocks: 表示所有可投资股票的名字。
   :type stocks: list
   :param start: 表示随机生成股票数据时交易日的起始日期, 默认为 ``None``。
   :type start: datetime, optional
   :param end: 表示随机生成股票数据时交易日的结束日期, 默认为 ``None``。
   :type end: datetime, optional

   .. py:method:: set_data(data)

      决定实验使用的数据是随机生成的还是用户本地输入。

      :param data: 用户输入的股票数据。
      :type data: list

   .. py:method:: randomly_generate()

      根据开始日期和结束日期随机生成用于实验的股票数据。
      
      .. Note::

         若要随机生成股票数据，需要以 ``datetime`` 包中的格式指定开始日期和结束日期，如 ``start = datetime.datetime(2016, 1, 1)``。

   .. py:method:: get_asset_return_mean_vector()
        
      用于计算所有可投资股票的平均投资回报率。

      :return: 所有可投资的股票的平均投资回报率。
      :rtype: list
   
   .. py:method:: get_asset_return_covariance_matrix()

      用于计算所有可投资股票回报率之间的协方差矩阵。

      :return: 所有可投资股票回报率之间的协方差矩阵。
      :rtype: list

   .. py:method:: get_similarity_matrix()

      计算各股票之间的相似矩阵。

      通过动态时间规整算法（Dynamic Time Warping, DTW）计算两股票之间的相似性。

      :return: 各股票间的相似矩阵。
      :rtype: list

.. py:function:: portfolio_optimization_hamiltonian(penalty, mu, sigma, q, budget)

   构建投资组合优化问题的哈密顿量。

   :param penalty: 惩罚参数。
   :type penalty: int
   :param mu: 各股票的预期回报率。
   :type mu: list
   :param sigma: 各股票回报率间的协方差矩阵。
   :type sigma: list
   :param q: 投资股票的风险。
   :type q: float
   :param budget: 投资预算, 即要投资的股票数量。
   :type budget: int

   .. math::

      C(x) = q \sum_i \sum_j S_{ji}x_ix_j  - \sum_{i}x_i \mu_i + A \left(B - \sum_i x_i\right)^2
   
   .. Hint::

      将布尔变量 :math:`x_i` 映射到哈密顿矩阵上，:math:`x_i \mapsto \frac{I-Z_i}{2}`。
   
   :return: 投资组合优化问题的哈密顿量。
   :rtype: paddle_quantum.Hamiltonian
   
.. py:function::  portfolio_diversification_hamiltonian(penalty, rho, q)

   构建投资组合分散化问题的哈密顿量。

   :param penalty: 惩罚参数。
   :type penalty: int
   :param rho: 各股票间的相似矩阵。
   :type rho: list
   :param q: 股票聚类的类别数。
   :type q: int

   .. math::

      \begin{aligned}
      C_x &= -\sum_{i=1}^{n}\sum_{j=1}^{n}\rho_{ij}x_{ij} + A\left(q- \sum_{j=1}^n y_j \right)^2 + \sum_{i=1}^n A\left(\sum_{j=1}^n 1- x_{ij} \right)^2 \\
          &\quad + \sum_{j=1}^n A\left(x_{jj} - y_j\right)^2 + \sum_{i=1}^n \sum_{j=1}^n A\left(x_{ij}(1 - y_j)\right).\\
      \end{aligned}
   
   .. Hint::

      将布尔变量 :math:`x_{ij}` 映射到哈密顿矩阵上，:math:`x_{ij} \mapsto \frac{I-Z_{ij}}{2}`。

   :return: 投资组合分散化问题的哈密顿量。
   :rtype: paddle_quantum.Hamiltonian

.. py:function::  arbitrage_opportunities_hamiltonian(g, penalty, n, k)

   构建最佳套利机会问题的哈密顿量。

   :param g: 不同货币市场间转换的图形化表示。
   :type g: networkx.DiGraph
   :param penalty: 惩罚参数。
   :type penalty: int
   :param n: 货币种类的数量，即图 g 中的顶点数量。
   :type n: int
   :param k: 套利回路中包含的顶点数。
   :type k: int

   .. math::

      C(x) = - P(x) + A\sum_{k=0}^{K-1} \left(1 - \sum_{i=0}^{n-1} x_{i,k}\right)^2 + A\sum_{k=0}^{K-1}\sum_{(i,j)\notin E}x_{i,k}x_{j,k+1}

   .. Hint::

      将布尔变量 :math:`x_{i,k}` 映射到哈密顿矩阵上，:math:`x_{i,k} \mapsto \frac{I-Z_{i,k}}{2}`。

   :return: 最佳套利机会问题的哈密顿量。
   :rtype: paddle_quantum.Hamiltonian
