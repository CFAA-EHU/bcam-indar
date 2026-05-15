Getting Started
===============

The purpose of the `indar` package is to collect those tools developed during our research, so
it does not have a specific structure, and it is not intended to be a general-purpose package.
For now, the package only offers methods for Experimental Modal Analysis (EMA), and
for solving eigenvalue problems of Delay Differential Equations (DDE) with a single delay.

In the following example, we show how to use the package to perform a simple EMA analysis.

A two-DoF system
----------------

The system is a pair of masses connected by springs and dampers as shown in the figure below.

.. image:: _images/TMD.png
   :align: center
   :width: 300px

The system is excited at the first mass, and the response is measured at both masses.
The equation of the system is given by

.. math::
    M\ddot{x} + C\dot{x} + Kx = f(t)

.. math::
    M = \begin{bmatrix} m_1 & 0 \\ 0 & m_2 \end{bmatrix},\quad
    C = \begin{bmatrix} c_1 + c_2 & -c_2 \\ -c_2 & c_2 \end{bmatrix},\quad
    K = \begin{bmatrix} k_1 + k_2 & -k_2 \\ -k_2 & k_2 \end{bmatrix}

We use the following parameters for the system:

+----------------------+----------------------+----------------------+
| Mass                 | Damping              | Stiffness            |
+======================+======================+======================+
| :math:`m_1 = 1.0`    | :math:`c_1 = 0.5`    | :math:`k_1 = 50.0`   |
+----------------------+----------------------+----------------------+
| :math:`m_2 = 0.2`    | :math:`c_2 = 0.08`   | :math:`k_2 = 7.0`    |
+----------------------+----------------------+----------------------+
