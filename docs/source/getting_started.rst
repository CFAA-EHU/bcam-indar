===============
Getting Started
===============

The purpose of the `indar` package is to collect those tools developed during our research, so
it does not have a specific structure, and it is not intended to be a general-purpose package.
For now, the package only offers methods for Experimental Modal Analysis (EMA), and
for solving eigenvalue problems of Delay Differential Equations (DDE) with a single delay.

In the following example, we show how to use the package to perform a simple EMA analysis.

Tuned Mass Damper
=================

We simulate a hammer test of the Tuned Mass Damper (TMD) in the figure below,
which is a 2-DoF system.

.. image:: _images/TMD.png
   :align: center
   :width: 300px

The system is excited at the first mass, and the response is measured at both masses.
The equation of motion of the masses is

.. math::
    \begin{gathered}
    M\ddot{x} + C\dot{x} + Kx = f \\
    M = \begin{bmatrix} m_1 & 0 \\ 0 & m_2 \end{bmatrix},\quad
    C = \begin{bmatrix} c_1 + c_2 & -c_2 \\ -c_2 & c_2 \end{bmatrix},\quad
    K = \begin{bmatrix} k_1 + k_2 & -k_2 \\ -k_2 & k_2 \end{bmatrix},
    \end{gathered}

with parameters in the table below.

+----------------------+----------------------+----------------------+
| Mass                 | Damping              | Stiffness            |
+======================+======================+======================+
| :math:`m_1 = 1.0`    | :math:`c_1 = 0.5`    | :math:`k_1 = 50.0`   |
+----------------------+----------------------+----------------------+
| :math:`m_2 = 0.2`    | :math:`c_2 = 0.08`   | :math:`k_2 = 7.0`    |
+----------------------+----------------------+----------------------+

The test is repeated four times with a sampling interval of
:math:`\Delta t = 0.02`, and the number of time samples is 256.

To follow the example along,
download the data files `2-dof-system_rep*.csv` from `this link <https://github.com/CFAA-EHU/bcam-indar/tree/main/docs/source/_downloads>`_ and
save them in a `data` folder.
Alternatively, execute the following command in your bash terminal:

.. code-block:: bash

    wget -nv -P data https://raw.githubusercontent.com/CFAA-EHU/bcam-indar/main/docs/source/_downloads/2-dof-system_rep{0..3}.csv

.. The next block of code creates a symbolic link to the data folder.
.. plot::
    :context: reset
    :nofigs:

    from pathlib import Path
    import shutil

    source_dir = Path('_documents')
    data_dir = Path('data')
    if data_dir.is_symlink() and not data_dir.exists():
        data_dir.unlink()

    if not data_dir.exists():
        try:
            data_dir.symlink_to(source_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(source_dir, data_dir)

To load and inspect the data, execute the following code in your Python environment:

.. plot::
    :context: close-figs
    :format: python
    :include-source: True

    # Import libraries
    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load data
    impact = []
    response = []
    n_rep = 4 # number of repetitions
    for rep in range(n_rep):
        file = os.path.join('data', f'2-dof-system_rep{rep}.csv')
        df = pd.read_csv(file, sep=',')
        impact.append(df['impact'].values)
        response.append([df['response_0'].values, df['response_1'].values])
    impact = np.array(impact) # shape: (n_rep, 256)
    response = np.array(response)  # shape: (n_rep, 2, 256)
    
    dt = 0.02 # sampling interval
    ns = impact.shape[-1] # number of time samples
    n_in, n_out = 1, 2

    # Plot data for the first trial
    fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(12, 5))
    tt = np.arange(ns)*dt
    axs[0].set_title('Impact')
    axs[0].plot(tt, impact[0, :])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Force')

    axs[1].set_title('Response')
    axs[1].plot(tt, response[0, 0, :], label='mass 1')
    axs[1].plot(tt, response[0, 1, :], label='mass 2')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Acceleration')
    axs[1].legend()

    plt.show()


Impulse Response Function
-------------------------

We will not use the H1 estimator but the class :py:class:`bcam.indar.ema.LTIKernel`.
The estimator used by this class is based on a time-domain model,
which means that the estimation is not affected by leakage,
so it is robust against any type of excitation and time-windowing.
For example, in case of random excitations,
it is not necessary to ensure periodicity.
However, its downside is that the estimation takes considerably more time than the H1 estimator.

The class :class:`LTIKernel` has two parameters, :math:`\alpha` and :math:`\beta`, which control the regularization of the estimation.
To choose these parameters we may manually inspect the estimated IRF and choose the one that looks best.
However, :class:`LTIKernel` can be coupled with :class:`sklearn.model_selection.GridSearchCV`
to automatically select the best parameters based on a scoring function.

.. plot::
    :context: close-figs
    :format: python
    :include-source: True

    # Import libraries
    from bcam.indar import ema
    from sklearn.model_selection import GridSearchCV

    # Create model to estimate the IRF
    lti_model = ema.LTIKernel(dt=dt, mode='a')

    # Define parameters to search in cross-validation
    pmts = {
        'alpha': np.array([1e-2, 1.]),
        'beta': 10**np.linspace(0, 2, 8)
    }
    clf = GridSearchCV(
        lti_model, pmts, cv=n_rep,
        scoring='neg_mean_squared_error',)
    r, e = 1, 0 # response (r), input (e)
    clf.fit(impact[:, :], response[:, r, :])
    # Choose the best model
    lti_model = clf.best_estimator_

    # Use the best model to estimate the IRF
    irf_ = np.zeros((n_out, n_in, ns)) # Shape: (2, 1, 256)
    for i in range(n_out):
        # The more repetitions, the better the estimation.
        lti_model.fit(impact, response[:, i, :])
        irf_[i, 0, :] = lti_model.kernel_

The array `irf_` contains the estimated IRF with the best paramaters found by cross-validation.
If we inspect `clf`, we will see that the best parameters are :math:`\alpha = 0.01` and :math:`\beta = 7.20`.
We plot the estimated FRF of mass 2 together with the input spectrum to
check in which frequency range the estimation is expected to worsen due to the low input energy.

.. plot::
    :context: close-figs
    :format: python
    :include-source: True

    fig, ax = plt.subplots()
    axt = ax.twinx() # Twin axis to plot the input spectrum

    frf_ = np.fft.rfft(irf_, axis=-1)*dt # Compute FRF
    ax.set_title('mass 2')
    freqs = np.fft.rfftfreq(ns, dt)
    ax.plot(
        freqs, np.abs(frf_[r, e]), label='estimated')
    axt.plot(
        freqs, np.abs(np.fft.rfft(impact[0, :]))*dt, label='input',
        color='k', alpha=0.4)

    ax.set_xlabel('Frequency')
    ax.set_yscale('log')
    axt.set_yscale('log')
    # Combine legends of both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axt.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.show()


Spectral estimation
-------------------

.. plot::
    :context: close-figs
    :format: python
    :include-source: True
    :nofigs:

    # Use AAA as algorithm for rational approximation
    sr = ema.SuperResolution(
        fs=1/dt, rational_fitter={'method': 'AAA'})
    # Construct stabilization diagram
    sp = ema.StablePoles(sr, max_order=30)
    sp.fit(irf_[:, 0, :].T)

    # Summarize clusters and show them as a text table in the docs
    clusters = sp.clusters_stats()
    print(clusters.to_markdown(index=False))
