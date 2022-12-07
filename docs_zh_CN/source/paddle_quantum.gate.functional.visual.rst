paddle\_quantum.gate.functional.visual
=======================================

可视化 ``paddle_quantum.ansatz.Circuit`` 类中量子门的函数

.. py:function:: scale_circuit_plot_param(scale)

   根据 ``scale`` 修改 ``__CIRCUIT_PLOT_PARAM`` 的字典参数。

   :param scale: 画图参数的缩放标量。
   :type scale: float

.. py:function:: set_circuit_plot_param(**kwargs)

   自定义画图参数字典 ``__CIRCUIT_PLOT_PARAM``。

   :param kwargs: 需要更新的字典 ``__CIRCUIT_PLOT_PARAM`` 参数。
   :type scale: Any

.. py:function:: get_circuit_plot_param()

   输出画图参数字典 ``__CIRCUIT_PLOT_PARAM``。

   :return: ``__CIRCUIT_PLOT_PARAM`` 字典的拷贝。
   :rtype: dict

.. py:function:: reset_circuit_plot_param()
   
   重置画图参数字典 ``__CIRCUIT_PLOT_PARAM``。

