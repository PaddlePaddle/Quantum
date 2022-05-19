paddle\_quantum.visual
=============================

量桨中的可视化的功能实现。

.. py:function:: plot_state_in_bloch_sphere(state, show_arrow: bool=False, save_gif=False, filename=None, view_angle=None, view_dist=None, set_color=None)

   将输入的量子态展示在 Bloch 球面上。

   :param state: 输入的量子态列表，可以支持态矢量和密度矩阵。
   :type state: List[paddle_quantum.State]
   :param show_arrow: 是否展示向量的箭头，默认为 ``False``。
   :type show_arrow: bool, optional
   :param save_gif: 是否存储 gif 动图，默认为 ``False``。
   :type save_gif: bool, optional
   :param filename: 存储的 gif 动图的名字。
   :type filename: str, optional
   :param view_angle: 视图的角度，第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360], 默认为 ``(30, 45)``。
   :type view_angle: Union[tuple, list], optional
   :param view_dist: 视图的距离，默认为 ``7``。
   :type view_dist: int, optional
   :param set_color: 若要设置指定的颜色，请查阅 ``cmap`` 表。默认为 ``"red to black gradient"``。
   :type set_color: str, optional

.. py:function:: plot_multi_qubits_state_in_bloch_sphere(state, which_qubits=None, show_arrow=False, save_gif=False, save_pic=True, filename=None, view_angle=None, view_dist=None, set_color='#0000FF')

   将输入的多量子比特的量子态展示在 Bloch 球面上。

   :param state: 输入的量子态，可以支持态矢量和密度矩阵。
   :type state: paddle_quantum.State
   :param which_qubits: 要展示的量子比特，默认为全展示。
   :type which_qubits: list, optional
   :param show_arrow: 是否展示向量的箭头，默认为 ``False``。
   :type show_arrow: bool, optional
   :param save_gif: 是否存储 gif 动图，默认为 ``False``。
   :type save_gif: bool, optional
   :param save_pic: 是否存储静态图片，默认为 ``True``。
   :type save_pic: bool, optional
   :param filename: 存储的图片的名字。
   :type filename: str, optional
   :param view_angle: 视图的角度，第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360], 默认为 ``(30, 45)``。
   :type view_angle: Union[tuple, list], optional
   :param view_dist: 视图的距离，默认为 ``7``。
   :type view_dist: int, optional
   :param set_color: 若要设置指定的颜色，请查阅 ``cmap`` 表。默认为 ``"blue"``。
   :type set_color: str, optional

.. py:function:: plot_rotation_in_bloch_sphere(init_state, rotating_angle, show_arrow=False, save_gif=False, filename=None, view_angle=None, view_dist=None, color_scheme=None)

   在 Bloch 球面上刻画从初始量子态开始的旋转轨迹。

   :param init_state: 输入的初始量子态，可以支持态矢量和密度矩阵。
   :type init_state: paddle_quantum.State
   :param rotating_angle: 旋转角度 ``[theta, phi, lam]``。
   :type rotating_angle: List[paddle.Tensor]
   :param show_arrow: 是否展示向量的箭头，默认为 ``False``。
   :type show_arrow: bool, optional
   :param save_gif: 是否存储 gif 动图，默认为 ``False``。
   :type save_gif: bool, optional
   :param filename: 存储的 gif 动图的名字。
   :type filename: str, optional
   :param view_angle: 视图的角度，第一个元素为关于 xy 平面的夹角 [0-360]，第二个元素为关于 xz 平面的夹角 [0-360], 默认为 ``(30, 45)``。
   :type view_angle: Union[list, tuple], optional
   :param view_dist: 视图的距离，默认为 ``7``。
   :type view_dist: int, optional
   :param color_scheme: 分别是初始颜色，轨迹颜色，结束颜色。若要设置指定的颜色，请查阅 ``cmap`` 表。默认为 ``"red"``。
   :type color_scheme: List[str], optional

.. py:function:: plot_density_matrix_graph(density_matrix, size=0.3)
   
   密度矩阵可视化工具。

   :param density_matrix: 多量子比特的量子态的状态向量或者密度矩阵,要求量子数大于 1。
   :type density_matrix: paddle_quantum.State
   :param size: 条宽度，在 0 到 1 之间，默认为 ``0.3``。
   :type size: float, optional