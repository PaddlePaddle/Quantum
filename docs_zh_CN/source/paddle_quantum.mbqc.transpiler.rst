paddle\_quantum.mbqc.transpiler
===============================

此模块包含电路模型和 MBQC 测量模式的转义工具。

.. py:function:: transpile(circuit, track)

   将输入的量子电路翻译为等价的测量模式。

   该函数通过将量子电路转化为等价的 MBQC 模型并运行，从而获得等价于原始量子电路的输出结果。

   :param circuit: 量子电路，包含可能的测量部分
   :type circuit: Circuit
   :param track: 是否显示翻译进度条的开关
   :type track: bool
   :return: 等价的测量模式
   :rtype: Pattern
