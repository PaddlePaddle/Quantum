paddle\_quantum.biocomputing.visualize
==============================================

蛋白质可视化工具

.. py:function:: visualize_protein_structure(aa_seq, bond_directions, view_angles)

   蛋白质结构可视化函数。

   :param aa_seq: 氨基酸序列。
   :type aa_seq: List[str]
   :param bond_directions: 氨基酸之间成键的空间取向。
   :type bond_directions: List[int]   
   :param view_angles: 输出图像的角度。
   :type view_angles: Optional[List[float]]