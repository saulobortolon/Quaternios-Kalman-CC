import numpy as np
from scipy.linalg import expm

# ---------- SO(3) auxiliares ----------
def hat(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

def Jl(phi):
    """Jacobiano à esquerda de SO(3): Exp(phi) = I + ..., e Exp_SE3(phi, rho) = (Exp(phi), Jl(phi) rho)."""
    th = np.linalg.norm(phi); A = hat(phi)
    if th < 1e-8:
        return np.eye(3) + 0.5 * A
    return np.eye(3) + (1 - np.cos(th)) / th**2 * A + (th - np.sin(th)) / th**3 * A @ A

def Jl_inv(phi):
    th = np.linalg.norm(phi); A = hat(phi)
    if th < 1e-8:
        return np.eye(3) - 0.5 * A
    return np.eye(3) - 0.5 * A + (1 / th**2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * A @ A

def Jr(phi):
    return Jl(-phi)

# ---------- quaternions (Hamilton, [w, x, y, z]) ----------
def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qpure(v):
    return np.array([0.0, v[0], v[1], v[2]])

def qExp(phi):
    """rotação de ângulo |phi| em torno de phi/|phi|: q = (cos(|phi|/2), sen(|phi|/2) phi/|phi|)."""
    th = np.linalg.norm(phi)
    if th < 1e-12:
        q = np.array([1.0, *(0.5 * np.asarray(phi))]); return q / np.linalg.norm(q)
    return np.array([np.cos(th / 2), *(np.sin(th / 2) * np.asarray(phi) / th)])

def qLog(q):
    q = q if q[0] >= 0 else -q            # q e -q são a mesma rotação: escolhe o hemisfério w >= 0
    nv = np.linalg.norm(q[1:])
    if nv < 1e-12:
        return 2 * q[1:]
    return 2 * np.arctan2(nv, q[0]) * q[1:] / nv

def qrot(q, v):
    return qmul(qmul(q, qpure(v)), qconj(q))[1:]

def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])

def qfrommat(R):
    return qExp(Log3(R))

def Exp3(phi):
    return qmat(qExp(phi))

def Log3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    if th < 1e-10:
        return v
    if np.pi - th < 1e-6:                  # perto de 180°: usa a diagonal
        A = (R + np.eye(3)) / 2; k = np.argmax(np.diag(A))
        ax = A[:, k] / np.sqrt(A[k, k]); return th * ax * (1 if ax @ v >= 0 else -1)
    return th / np.sin(th) * v

# ---------- dual quaternions: sigma = r + eps d, com d = 1/2 p r ----------
def dq(q, p):
    return (q.copy(), 0.5 * qmul(qpure(p), q))

def dq_parts(s):
    r, d = s
    return r, 2 * qmul(d, qconj(r))[1:]

def dqmul(s1, s2):
    r1, d1 = s1; r2, d2 = s2
    return (qmul(r1, r2), qmul(r1, d2) + qmul(d1, r2))

def dqconj(s):                              # conjugado quaterniônico = inverso de um dual quaternion unitário
    return (qconj(s[0]), qconj(s[1]))

def dqExp(xi):
    """xi = (phi, rho) em R^6 -> dual quaternion unitário (parafuso)."""
    phi, rho = xi[:3], xi[3:]
    return dq(qExp(phi), Jl(phi) @ rho)

def dqLog(s):
    r, p = dq_parts(s); phi = qLog(r)
    return np.concatenate([phi, Jl_inv(phi) @ p])

# ---------- "quaternions tridentes": tau = r + eps1 a + eps2 b, a = 1/2 v r, b = 1/2 p r ----------
def tq(q, v, p):
    return (q.copy(), 0.5 * qmul(qpure(v), q), 0.5 * qmul(qpure(p), q))

def tq_parts(t):
    r, a, b = t
    return r, 2 * qmul(a, qconj(r))[1:], 2 * qmul(b, qconj(r))[1:]

def tqmul(t1, t2):
    r1, a1, b1 = t1; r2, a2, b2 = t2
    return (qmul(r1, r2), qmul(r1, a2) + qmul(a1, r2), qmul(r1, b2) + qmul(b1, r2))

def tqconj(t):
    return (qconj(t[0]), qconj(t[1]), qconj(t[2]))

def tqExp(xi):
    """xi = (theta, nu, rho) em R^9 -> elemento de SE_2(3) na forma tridente."""
    th, nu, rho = xi[:3], xi[3:6], xi[6:]
    J = Jl(th); return tq(qExp(th), J @ nu, J @ rho)

def tqLog(t):
    r, v, p = tq_parts(t); th = qLog(r); Ji = Jl_inv(th)
    return np.concatenate([th, Ji @ v, Ji @ p])

# ---------- matrizes equivalentes (só para conferir) ----------
def se3_mat(R, p):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M

def se23_mat(R, v, p):
    M = np.eye(5); M[:3, :3] = R; M[:3, 3] = v; M[:3, 4] = p; return M

def se3_hat(xi):
    M = np.zeros((4, 4)); M[:3, :3] = hat(xi[:3]); M[:3, 3] = xi[3:]; return M

def se23_hat(xi):
    M = np.zeros((5, 5)); M[:3, :3] = hat(xi[:3]); M[:3, 3] = xi[3:6]; M[:3, 4] = xi[6:]; return M
