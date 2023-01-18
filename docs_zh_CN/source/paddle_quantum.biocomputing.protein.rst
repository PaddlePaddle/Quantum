paddle\_quantum.biocomputing.protein
==============================================

量桨中蛋白质结构构建工具。

.. py:function:: _check_valid_aa_seq(aa_seq)

   判断一个给定的氨基酸序列标识是否合理。

   :param aa_seq: 氨基酸序列。
   :type aa_seq: str

   :return: 返回一个氨基酸序列格式是否正确。
   :rtype: bool

.. py:function:: _generate_valid_contact_pairs(num_aa)

   根据输入的氨基酸数目给出可能发生相互作用的氨基酸对。

   :param num_aa: 蛋白质中氨基酸数量。
   :type num_aa: int

   :return: 在给定情况下可能发生相互作用的氨基酸对的序号。
   :rtype: List[Tuple[int]]

.. py:class:: Protein(aa_seqs, fixed_bond_directions, contact_pairs)

   基类：:py:class:`networkx.Graph`

   量桨中用于构建蛋白质结构的类型。

   :param aa_seqs: 蛋白质中氨基酸序列。
   :type aa_seqs: str
   :param fixed_bond_directions: 氨基酸之间确定方向的连接的序号。
   :type fixed_bond_directions: Optional[Dict[Tuple,int]]
   :param contact_pairs: 发生相互作用的氨基酸对的序号。
   :type contact_pairs: Optional[List[Tuple[int]]]

   .. py:property:: num_config_qubits()

      蛋白质中用来表示结构的量子比特数。

   .. py:property:: num_contact_qubits()

      蛋白质中用来表示氨基酸之间相互作用的量子比特数。

   .. py:property:: num_qubits()
   
      蛋白质中的总量子比特数。

   .. py:method:: distance_operator(p, q)

      蛋白质中第p个和第q个氨基酸之间的距离算符。

      :param p: 氨基酸序号。
      :type p: int
      :param q: 氨基酸序号。
      :type q: int

      :return: 氨基酸距离算符。
      :rtype: openfermion.QubitOperator

   .. py:method:: get_protein_hamiltonian(lambda0, lambda1, energy_multiplier)

      蛋白质哈密顿量。

      :param lambda0: 哈密顿量正则化因子。
      :type lambda0: Optional[float]
      :param lambda1: 哈密顿量正则化因子。
      :type lambda1: Optional[float]
      :param energy_multiplier: 能量尺度因子。
      :type energy_multiplier: Optional[float]

      :return: 蛋白质哈密顿量。
      :rtype: paddle_quantum.Hamiltonian