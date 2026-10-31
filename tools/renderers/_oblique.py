"""Perspective projection from the original physical-learning video renderer."""
import numpy as np


class Oblique:
    """Oblique perspective view of a periodic planar network with one vertical displacement z per node, as in
    make_probe3d.py: elevation `elev` (degrees), azimuth az (radians, per frame), viewer distance `dist`.
    Edges across the periodic boundary are drawn as two pieces that stop at the edge of the floor."""

    def __init__(self, pos, edges, box, elev=30.0, dist=None):
        self.pos, self.b = pos, float(box[0])
        self.elev, self.D = np.radians(elev), dist or 3.2 * float(box[0])
        I, J = edges[:, 0], edges[:, 1]
        dv = pos[J] - pos[I]
        dv -= self.b * np.round(dv / self.b)
        pieces = []
        for e in range(len(edges)):
            for a, sign in ((I[e], 1), (J[e], -1)):
                end = pos[a] + sign * dv[e]
                if a == J[e] and np.all((end >= 0) & (end <= self.b)):
                    continue                                   # an interior edge is drawn once, from I
                t = 1.0
                for c in range(2):
                    step = sign * dv[e, c]
                    if end[c] > self.b:
                        t = min(t, (self.b - pos[a, c]) / step)
                    elif end[c] < 0:
                        t = min(t, -pos[a, c] / step)
                pieces.append((e, a, sign, t))
        pe = np.array([q[0] for q in pieces])
        self.node = np.array([q[1] for q in pieces])
        self.other = np.where(self.node == I[pe], J[pe], I[pe])
        self.dx = np.array([q[2] for q in pieces])[:, None] * np.array([q[3] for q in pieces])[:, None] * dv[pe]
        self.t = np.array([q[3] for q in pieces])

    def project(self, x, y, z, az):
        xc, yc = x - self.b / 2, y - self.b / 2
        a = xc * np.cos(az) - yc * np.sin(az)
        d = xc * np.sin(az) + yc * np.cos(az)                  # depth: larger is farther
        s = self.D / (self.D + d * np.cos(self.elev))
        return a * s, (d * np.sin(self.elev) + z * np.cos(self.elev)) * s, s

    def floor(self, az):
        X, Y, _ = self.project(np.array([0, self.b, self.b, 0.0]), np.array([0, 0, self.b, self.b]), np.zeros(4), az)
        return np.stack([X, Y], 1)

    def segments(self, z, az):
        x0, y0 = self.pos[self.node, 0], self.pos[self.node, 1]
        z0 = z[self.node]
        z1 = z0 + self.t * (z[self.other] - z0)
        X0, Y0, _ = self.project(x0, y0, z0, az)
        X1, Y1, _ = self.project(x0 + self.dx[:, 0], y0 + self.dx[:, 1], z1, az)
        return np.stack([np.stack([X0, Y0], 1), np.stack([X1, Y1], 1)], 1)

    def back_to_front(self, az):
        xc, yc = self.pos[:, 0] - self.b / 2, self.pos[:, 1] - self.b / 2
        return np.argsort(-(xc * np.sin(az) + yc * np.cos(az)))
