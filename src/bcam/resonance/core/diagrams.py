#!/usr/bin/env python
"""Search for the null set of a function in a region."""

# ==== IMPORTS ====
from numbers import Integral

import numpy as np
from scipy.optimize import root_scalar


# ==== USER INTERFACE ====

class SearchNullSet:

    def __init__(self,
            fun, region, stop=2e-2, n_neighs=1, seed=None):
        """Initialize the search for the null set of a function.
        
        Parameters
        ----------
        fun : callable
            The function to evaluate.
        region : array_like
            The region where the function is defined as [[x0, x1], [y0, y1]].
        stop : float, optional
            The stopping criterion for the search. The search stops when the
            proportion of values in a given interval is less than stop.
            The default is 2e-2.
        seed : int, optional
            The seed for the random number generator. The default is None.
        """
        self.fun = fun
        self.region = np.asarray(region)
        self.stop = abs(stop)
        self.n_neighs = n_neighs
        rng = np.random.default_rng(seed)
        self.rng = rng

        self.bounds = None
        self._prop = [-1]
        self._aprop = None
        self._X = None
        self._y = None

    @property
    def prop(self):
        return np.asarray(self._prop)[1:]

    @property
    def X(self):
        return self._map_unit2region(self._X)

    @property
    def y(self):
        return self._y

    @property
    def aprop(self):
        return self._aprop

    def _map_unit2region(self, X):
        region = self.region
        tX = X.copy()
        tX[:, 0] = region[0, 0] + (region[0, 1] - region[0, 0]) * tX[:, 0]
        tX[:, 1] = region[1, 0] + (region[1, 1] - region[1, 0]) * tX[:, 1]
        return tX

    def _map_region2unit(self, X):
        region = self.region
        tX = X.copy()
        tX[:, 0] = (tX[:, 0] - region[0, 0]) / (region[0, 1] - region[0, 0])
        tX[:, 1] = (tX[:, 1] - region[1, 0]) / (region[1, 1] - region[1, 0])
        return tX

    def _evaluate(self, Xn):
        """Evaluate the function at the points in Xn."""
        y = [self.fun(*x) for x in self._map_unit2region(Xn)]
        # update the proportion of values below y_scale.
        y = np.array(y)
        M = len(y)
        N = len(self._y) if self._y is not None else 0
        tag = np.empty_like(y, dtype=bool)
        tag.fill(False)
        # select y values between the bounds.
        tag[(y > self.bounds[0]) & (y < self.bounds[1])] = True
        s0 = N * self._prop[-1]
        s1 = sum(tag)
        prop = (s0 + s1) / (N + M)
        self._prop.append(prop)
        # update X and y.
        self._y = np.concatenate((self._y, y)) if self._y is not None else y
        self._X = np.concatenate((self._X, Xn), axis=0) if self._X is not None else Xn

    def _new_points(self):
        """Generate new points around the current points."""
        X = self._X
        n_points = X.shape[0]
        radius = np.sqrt(self._aprop / (np.pi * n_points))
        shift = self.rng.normal(0, radius, size=(self.n_neighs * n_points, 2))
        Xn = [X + shift[i::self.n_neighs, :] for i in range(self.n_neighs)]
        Xn = np.concatenate(Xn, axis=0)
        # remove rows where at least one entry is outside [0, 1]
        Xn = Xn[np.all(Xn >= 0, axis=1) & np.all(Xn <= 1, axis=1)]
        return Xn

    def get_set(self, bounds, X0=None, aprop=1):
        """Get the set of points where the value of the function is between the bounds.
        
        Parameters
        ----------
        bounds : a pair of floats
        X0 : int or array_like, optional.
            The initial set of points. If an integer, it generates the given number of points uniformly.
            The default is 25 points uniformly distributed.
        aprop : float, optional
            The proportion of area covered by the set. The default is 1, that is,
            a set uniformly distributed across the whole region.
        
        Returns
        -------
        array_like
            The set of points where the function is between the bounds.
        """
        self.bounds = bounds
        # generate initial set of points.
        if abs(aprop - 0.5) > 0.5:
            raise ValueError('aprop must be in [0, 1].')
        if X0 is None:
            X0 = 25
        if isinstance(X0, Integral):
            Xn = self.rng.uniform(0, 1, size=(X0, 2))
            aprop = 1
        else:
            X0 = np.asarray(X0)
            Xn = self._map_region2unit(X0)
            Xn = Xn[np.all(Xn >= 0, axis=1) & np.all(Xn <= 1, axis=1)]
        self._aprop = aprop
        self._evaluate(Xn)
        while abs(self._prop[-1] - self._prop[-2]) > self.stop:
            Xn = self._new_points()
            self._evaluate(Xn)
        # extract points where y is between the bounds.
        null = self.X[(self._y > bounds[0]) & (self._y < bounds[1])]
        return null
    
    def clear(self):
        """Clear the data."""
        self.y_scale = None
        self._prop = [-1]
        self._aprop = None
        self._X = None
        self._y = None

# ==== Hayes equation stability diagram ====

def _fun(q, tau, x):
    if q != 0:
        return (q / np.tan(tau * q)) - x
    else:
        return np.cos(tau * q) / tau - x

def _dfun(q, tau, x):
    if q != 0:
        a = tau * q
        return a * (np.sin(2 * a) / (2 * a) - 1) / np.sin(a)**2
    else:
        return 0

def _ddfun(q, tau, x):
    if q != 0:
        a = tau * q
        r = 2 * tau / np.sin(a)**2
        r = r * ((a / np.sin(a)) - 1)
        return r
    else:
        return 1 / 3

def hayes(b0, b1, tau=1):
    r"""
    Check if the Hayes equation is stable.
    
    The Hayes equation is  :math:`y'(t) = b_0 y(t) + b_1 y(t - \tau)`,
    where :math:`b_0` and :math:`b_1` are scalars.
    The stability is evaluated by using the stability boundaries given by the D-subdivision.
    
    Parameters
    ----------
    b0 : float
        The coefficient :math:`b_0`.
    b1 : float
        The coefficient :math:`b_1`.
    tau : float, optional
        The delay :math:`\tau`. The default is 1.
    
    Returns
    -------
    bool
        True if the Hayes equation is stable, False otherwise.
    """
    if b0 >= 1 / tau:
        return True
    sol = root_scalar(
        _fun, fprime=_dfun, fprime2=_ddfun,
        args=(tau, b0),
        bracket = [0, np.pi / tau - 1e-7],
        x0 = None
    )
    sol = sol.root
    bdr = -sol / np.sin(tau * sol)
    if (b1 >= -b0) or (b1 <= bdr):
        return True
    return False
