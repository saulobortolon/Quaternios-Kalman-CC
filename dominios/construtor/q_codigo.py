"""Código comum do domínio VI, inserido nas células dos notebooks (cada notebook roda sozinho no Colab).

Convenções (as mesmas em todo o domínio):
- q = (w, x, y, z) = w + x i + y j + z k, com o produto de Hamilton (i² = j² = k² = ijk = −1);
- rotação ativa: v_mundo = q v_corpo q*, isto é, R(q) leva coordenadas do corpo para o mundo;
- cinemática: q' = ½ q ⊗ (0, ω), com ω a velocidade angular medida no corpo (o que um giroscópio mede);
- vetor de rotação φ = θ n; Exp(φ) = (cos(θ/2), sen(θ/2) n).
"""

CLASSE_Q = r'''
class Q:
    """Quaternion w + x i + y j + z k. As componentes podem ser floats, Fractions, inteiros ou números duais."""
    def __init__(self, w, x=0, y=0, z=0): self.w, self.x, self.y, self.z = w, x, y, z
    def __mul__(self, o):
        if not isinstance(o, Q):                           # quaternion vezes escalar
            return Q(self.w * o, self.x * o, self.y * o, self.z * o)
        a, b = self, o
        return Q(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                 a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                 a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                 a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w)
    def __rmul__(self, s): return Q(s * self.w, s * self.x, s * self.y, s * self.z)
    def __add__(self, o): return Q(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)
    def __sub__(self, o): return Q(self.w - o.w, self.x - o.x, self.y - o.y, self.z - o.z)
    def __neg__(self): return Q(-self.w, -self.x, -self.y, -self.z)
    def conj(self): return Q(self.w, -self.x, -self.y, -self.z)
    def norma2(self): return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    def vet(self): return np.array([self.w, self.x, self.y, self.z])
    def __eq__(self, o): return all(p == r for p, r in zip(self.vet(), o.vet()))
    def __repr__(self): return f"({self.w}) + ({self.x})i + ({self.y})j + ({self.z})k"
'''

FUNCOES_Q = r'''
def sinc(x):
    """sen(x)/x, sem perder dígitos perto de 0 (np.sinc usa sen(πx)/(πx))."""
    return np.sinc(np.asarray(x, float) / np.pi)

def qmul(a, b):
    """Produto de Hamilton de quaternions guardados como arrays (w, x, y, z); aceita pilhas (..., 4)."""
    a, b = np.asarray(a), np.asarray(b)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)

def qconj(q):
    q = np.asarray(q); return q * np.array([1, -1, -1, -1])

def qnorm(q):
    q = np.asarray(q, float); return q / np.linalg.norm(q, axis=-1, keepdims=True)

def qrot(q, v):
    """O vetor v (3,) levado pela rotação q: a parte vetorial de q (0, v) q*."""
    return qmul(qmul(q, np.concatenate([[0.0], v])), qconj(q))[1:]

def qmat(q):
    """A matriz de rotação R(q), com R v = q v q* (q unitário)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def mat_q(R):
    """De uma matriz de rotação para um quaternion (método de Shepperd: usa o maior pivô, sem perder dígitos)."""
    t = np.trace(R); d = np.diag(R)
    k = np.argmax([t, *d])
    if k == 0:
        w = np.sqrt(1 + t) / 2; q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    elif k == 1:
        x = np.sqrt(1 + 2 * d[0] - t) / 2; q = [(R[2, 1] - R[1, 2]) / (4 * x), x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x)]
    elif k == 2:
        y = np.sqrt(1 + 2 * d[1] - t) / 2; q = [(R[0, 2] - R[2, 0]) / (4 * y), (R[0, 1] + R[1, 0]) / (4 * y), y, (R[1, 2] + R[2, 1]) / (4 * y)]
    else:
        z = np.sqrt(1 + 2 * d[2] - t) / 2; q = [(R[1, 0] - R[0, 1]) / (4 * z), (R[0, 2] + R[2, 0]) / (4 * z), (R[1, 2] + R[2, 1]) / (4 * z), z]
    q = np.array(q); return q if q[0] >= 0 else -q

def qexp(phi):
    """Exp do vetor de rotação φ = θ n: (cos(θ/2), sen(θ/2) n), com o sinc estável perto de θ = 0."""
    phi = np.asarray(phi, float); th = np.linalg.norm(phi)
    return np.concatenate([[np.cos(th / 2)], 0.5 * sinc(th / 2) * phi])

def qlog(q):
    """Log: o vetor de rotação φ de q (escolhendo o representante com w ≥ 0, isto é, |θ| ≤ π)."""
    q = np.asarray(q, float)
    if q[0] < 0: q = -q
    s = np.linalg.norm(q[1:]); th = 2 * np.arctan2(s, q[0])     # atan2: estável para ângulos pequenos e grandes
    return 2 * q[1:] / sinc(th / 2)

def slerp(q0, q1, s):
    """Interpolação esférica: q0 Exp(s Log(q0* q1)), pelo arco mais curto."""
    return qmul(q0, qexp(s * qlog(qmul(qconj(q0), q1))))

def chapeu(v):
    """A matriz antissimétrica [v]×, com [v]× u = v × u."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
'''
