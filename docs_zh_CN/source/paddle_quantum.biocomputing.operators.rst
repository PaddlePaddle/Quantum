paddle\_quantum.biocomputing.operator
==============================================

蛋白质哈密顿量中使用的算符集合。

.. py:function:: edge_direction_indicator(edge, affected_qubits, direction)

   用于指示边的方向的算符。

   :param edge: 蛋白质中氨基酸-氨基酸之间的边的编号。
   :type edge: Tuple[int]
   :param affected_qubits: 算符涉及到的量子比特序号。
   :type affected_qubits: Optional[List[int]]
   :param direction: 边的方向。
   :type direction: Optional[int]

   :return: 算符对应的符号，边方向指示算符。
   :rtype: Tuple[float, Dict]

.. py:function:: contact_indicator(qindex)

   蛋白质中指示两个氨基酸是否有相互作用的算符。

   :param qindex: 算符影响的量子比特序号。
   :type qindex: int

   :return: 相互作用算符。
   :rtype: openfermion.QubitOperator

.. py:function:: backwalk_indicator(e0_attrs, e1_attrs)

   反映边重叠情况的算符。

   :param e0_attrs: 给定边上的特征。
   :type e0_attrs: Dict
   :param e1_attrs: 相邻边上的特征。
   :type e1_attrs: Dict

   :return: 边重叠算符。
   :rtype: openfermion.QubitOperator