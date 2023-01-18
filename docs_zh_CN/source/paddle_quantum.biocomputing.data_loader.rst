paddle\_quantum.biocomputing.data_loader
==============================================

数据加载模块。

.. py:function:: load_energy_matrix_file()

   返回从Miyazawa-Jernigan potential文档中生成的能量矩阵。

   :return: 能量矩阵，氨基酸序列。 
   :rtype:	tuple[np.ndarray,List[str]]

.. py:function:: _parse_energy_matrix(matrix)

   对加载的能量矩阵进行解析。

   :param matrix: 能量矩阵
   :type matrix: np.ndarray

   :return: 解析后的能量矩阵。 
   :rtype: np.ndarray