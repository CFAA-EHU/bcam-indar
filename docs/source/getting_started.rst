Getting Started
===============

The purpose of the `indar` package is to collect those tools developed during our research, so
it does not have a specific structure, and it is not intended to be a general-purpose package.
For now, the package only offers methods for Experimental Modal Analysis (EMA), and
for solving eigenvalue problems of Delay Differential Equations (DDE) with a single delay.

In the following example, we show how to use the package to perform a simple EMA analysis.

A two-DoF system
----------------

We simulate a hammer hit test of the two-DoF system in the figure below, and
we use the `indar` package to estimate the modal parameters of the system.

.. image:: _images/TMD.png
   :align: center
   :width: 300px

The system is excited at the first mass, and the response is measured at both masses.
The equation of motion of the masses is given by

.. math::
    M\ddot{x} + C\dot{x} + Kx = f(t)

.. math::
    M = \begin{bmatrix} m_1 & 0 \\ 0 & m_2 \end{bmatrix},\quad
    C = \begin{bmatrix} c_1 + c_2 & -c_2 \\ -c_2 & c_2 \end{bmatrix},\quad
    K = \begin{bmatrix} k_1 + k_2 & -k_2 \\ -k_2 & k_2 \end{bmatrix},

where the parameters are in the table below.

+----------------------+----------------------+----------------------+
| Mass                 | Damping              | Stiffness            |
+======================+======================+======================+
| :math:`m_1 = 1.0`    | :math:`c_1 = 0.5`    | :math:`k_1 = 50.0`   |
+----------------------+----------------------+----------------------+
| :math:`m_2 = 0.2`    | :math:`c_2 = 0.08`   | :math:`k_2 = 7.0`    |
+----------------------+----------------------+----------------------+

Suppose that the test is repeated four times with a sampling interval of
:math:`\Delta t = 0.04` s and the number of time samples is 256.
To follow the example, download the data files `2-dof-system_rep*.csv` from `this link <https://github.com/CFAA-EHU/bcam-indar/tree/main/docs/source/_downloads>`_.
