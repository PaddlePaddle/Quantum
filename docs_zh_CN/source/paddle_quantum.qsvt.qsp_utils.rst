paddle\_quantum.qsvt.qsp\_utils
===============================

量子信号处理相关工具函数包

.. py:function:: random_odd_poly_generation(degree, odd)

   生成一个随机的满足量子信号处理要求的多项式

   :param degree: 多项式的度
   :type degree: int
   :param odd: 多项式的奇偶性，输入 `True` 则为奇函数， `False` 则为偶函数
   :type odd: bool

   :return: 一个随机生成的多项式
   :rtype: Polynomial

.. py:function:: clean_small_error(array)
   
   清除相对较小的项，以提升计算精度

   :param array: 目标数组
   :type array: ndarray

   :return: 经过清除后的数组
   :rtype: ndarray

.. py:function:: poly_norm(poly, p=1)

   计算一个多项式的 p 范数

   :param poly: 目标多项式
   :type poly: Polynomial
   :param p: p 范数，默认为 `1`，输入 `0` 则是无穷范数
   :type p: Optional[int]

   :return: 目标多项式的 p 范数
   :rtype: float

.. py:function:: poly_real(poly)

   取一个多项式的实部

   :param poly: 目标多项式
   :type poly: Polynomial

   :return: 目标多项式的实部
   :rtype: Polynomial

.. py:function:: poly_imag(poly)

   取一个多项式的实部

   :param poly: 目标多项式
   :type poly: Polynomial

   :return: 目标多项式的虚部
   :rtype: Polynomial

.. py:function:: poly_matrix(poly, matrix_A)

   计算一个矩阵的多项式 poly(matrix_A)

   :param poly: 输入多项式
   :type poly: Polynomial
   :param matrix_A: 输入矩阵
   :type matrix_A: paddle.Tensor

   :return: 矩阵的多项式结果 poly(matrix_A)
   :rtype: paddle.Tensor

.. py:function:: exp_matrix(t, matrix_A)

   计算矩阵指数 :math:`e^{itA}`

   :param t: 演化时间
   :type t: float
   :param matrix_A: 目标矩阵 A
   :type matrix_A: paddle.Tensor

   :return: 矩阵指数 :math:`e^{itA}`
   :rtype: paddle.Tensor