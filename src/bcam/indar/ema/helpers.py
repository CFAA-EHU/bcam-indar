import attrs
import numpy as np


def _pos_imag(x):
    x = np.asarray(x, dtype=complex)
    x[np.imag(x) < 0] = np.conj(x[np.imag(x) < 0])
    return x

@attrs.define
class Poles:
    real = attrs.field(converter=lambda x: np.asarray(x, dtype=float), default=np.array([]))
    cx = attrs.field(converter=_pos_imag, default=np.array([]))

    @classmethod
    def from_raw(cls, poles):
        poles = np.asarray(poles, dtype=complex)

        # Separate real and complex conjugated poles.
        poles_u = list(poles[np.imag(poles) >= 0])
        poles_l = list(poles[np.imag(poles) < 0])
        poles_r, poles_c = [], []
        while (len(poles_u) > 0) and (len(poles_l) > 0):
            distances = np.abs([p - np.conj(poles_u[-1]) for p in poles_l])
            idx = np.argmin(distances)
            if distances[idx] < 1e-8:
                poles_c.append(poles_u.pop())
                poles_l.pop(idx)
            else:
                # If a pole does not have a conjugate, we consider it as a real pole.
                poles_r.append(poles_u.pop())

        poles_r.extend(poles_u)
        poles_r.extend(poles_l)
        poles_r = np.real(poles_r)
        poles_c = np.array(poles_c, dtype=complex)

        return cls(real=poles_r, cx=poles_c)

    def full(self):
        return np.concatenate((self.real, self.cx, np.conj(self.cx)))

    def count(self):
        return len(self.real) + 2*len(self.cx)


@attrs.define
class HCoeffs:
    real = attrs.field(converter=lambda x: np.asarray(x, dtype=float), default=np.array([]))
    cx = attrs.field(converter=lambda x: np.asarray(x, dtype=complex), default=np.array([]))

    def full(self):
        return np.concatenate((self.real, self.cx, np.conj(self.cx)))
