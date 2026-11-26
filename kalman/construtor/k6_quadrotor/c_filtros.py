G_VEC = np.array([0.0, 0.0, -9.81])

# ---------------- simulação do quadrotor ----------------
def trajetoria(t, ampl=(4.0, 3.0, 1.0), freq=(0.12, 0.17, 0.09)):
    """Lissajous 3D (como em Böhm et al.) e suas derivadas; guinada que gira devagar."""
    A = np.array(ampl); w = np.array(freq) * 2 * np.pi; fase = np.array([0.0, 0.7, 1.3])
    p = A * np.sin(w * t + fase) + np.array([0, 0, 3.0])
    v = A * w * np.cos(w * t + fase)
    a = -A * w**2 * np.sin(w * t + fase)
    psi = 0.4 * t + 0.8 * np.sin(0.3 * t)
    return p, v, a, psi

def atitude_quadrotor(a, psi):
    """Um quadrotor só empurra ao longo do seu eixo z: z_corpo // (a − g). A guinada psi é livre."""
    f = a - G_VEC; zb = f / np.linalg.norm(f)
    xc = np.array([np.cos(psi), np.sin(psi), 0.0])
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb); xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])

def propaga_fisica(R, v, p, w_m, a_m, dt):
    """Integração de um passo com a IMU (giro w_m e força específica a_m, no corpo)."""
    acc = R @ a_m + G_VEC
    return R @ Exp3(w_m * dt), v + acc * dt, p + v * dt + 0.5 * acc * dt**2

def simula_voo(T=30.0, dt=0.01, sig_g=0.05, sig_a=0.1, rng=None):
    n = int(round(T / dt)); ts = np.arange(n + 1) * dt
    Rs = [atitude_quadrotor(trajetoria(t)[2], trajetoria(t)[3]) for t in ts]
    w_true = np.array([Log3(Rs[k].T @ Rs[k + 1]) / dt for k in range(n)])
    a_true = np.array([Rs[k].T @ (trajetoria(ts[k])[2] - G_VEC) for k in range(n)])
    p0, v0 = trajetoria(0.0)[:2]
    R, v, p = Rs[0], v0.copy(), p0.copy(); X = [(R, v, p)]
    for k in range(n):                       # a "verdade" é a integração exata do mesmo modelo discreto
        R, v, p = propaga_fisica(R, v, p, w_true[k], a_true[k], dt); X.append((R, v, p))
    w_m = w_true + sig_g * rng.standard_normal(w_true.shape)
    a_m = a_true + sig_a * rng.standard_normal(a_true.shape)
    return dict(ts=ts, X=X, w_m=w_m, a_m=a_m, dt=dt, w_true=w_true, a_true=a_true)

# ---------------- os três grupos ----------------
class GrupoSO3xR6:
    """(a) quaternion + velocidade + posição: erro de atitude multiplicativo, v e p aditivos (no mundo)."""
    nome = "SO(3)×R⁶ (quaternion + vetores)"
    def de_Rvp(self, R, v, p): return (qfrommat(R), v.copy(), p.copy())
    def para_Rvp(self, X): return qmat(X[0]), X[1], X[2]
    def boxplus(self, X, d): return (qmul(X[0], qExp(d[:3])), X[1] + d[3:6], X[2] + d[6:])
    def boxminus(self, X1, X2): return np.concatenate([qLog(qmul(qconj(X2[0]), X1[0])), X1[1] - X2[1], X1[2] - X2[2]])
    def F_G(self, X, w_m, a_m, dt):
        R = qmat(X[0]); Gm = Exp3(w_m * dt); A = hat(a_m); I = np.eye(3); Z = np.zeros((3, 3))
        F = np.block([[Gm.T, Z, Z], [-R @ A * dt, I, Z], [-0.5 * R @ A * dt**2, I * dt, I]])
        Gn = np.block([[-Jr(w_m * dt) * dt, Z], [Z, -R * dt], [Z, -0.5 * R * dt**2]])
        return F, Gn
    def H_pos(self, X): return np.hstack([np.zeros((3, 6)), np.eye(3)])
    def ad(self, d):
        M = np.zeros((9, 9)); M[:3, :3] = hat(d[:3]); return M

class GrupoSE3xR3:
    """(b) dual quaternion (pose) + velocidade: erro de pose no grupo SE(3), velocidade aditiva no mundo."""
    nome = "SE(3)×R³ (dual quaternion + velocidade)"
    def de_Rvp(self, R, v, p): return (dq(qfrommat(R), p), v.copy())
    def para_Rvp(self, X): r, p = dq_parts(X[0]); return qmat(r), X[1], p
    def boxplus(self, X, d):   # ordem do erro: (theta, v, rho)
        return (dqmul(X[0], dqExp(np.concatenate([d[:3], d[6:]]))), X[1] + d[3:6])
    def boxminus(self, X1, X2):
        xi = dqLog(dqmul(dqconj(X2[0]), X1[0]))
        return np.concatenate([xi[:3], X1[1] - X2[1], xi[3:]])
    def F_G(self, X, w_m, a_m, dt):
        R = self.para_Rvp(X)[0]; Gm = Exp3(w_m * dt); A = hat(a_m); I = np.eye(3); Z = np.zeros((3, 3))
        F = np.block([[Gm.T, Z, Z], [-R @ A * dt, I, Z], [-0.5 * Gm.T @ A * dt**2, Gm.T @ R.T * dt, Gm.T]])
        Gn = np.block([[-Jr(w_m * dt) * dt, Z], [Z, -R * dt], [Z, -0.5 * Gm.T * dt**2]])
        return F, Gn
    def H_pos(self, X): return np.hstack([np.zeros((3, 6)), self.para_Rvp(X)[0]])
    def ad(self, d):
        M = np.zeros((9, 9)); th, rho = d[:3], d[6:]
        M[:3, :3] = hat(th); M[6:, :3] = hat(rho); M[6:, 6:] = hat(th); return M

class GrupoSE23:
    """(c) quaternion tridente, erro à esquerda (no corpo): X = X̂ ⊗ Exp(δ)  (LQEKF de Ouyang e Wu)."""
    nome = "SE₂(3) tridente, erro à esquerda (LQEKF)"
    def de_Rvp(self, R, v, p): return tq(qfrommat(R), v, p)
    def para_Rvp(self, X): r, v, p = tq_parts(X); return qmat(r), v, p
    def boxplus(self, X, d): return tqmul(X, tqExp(d))
    def boxminus(self, X1, X2): return tqLog(tqmul(tqconj(X2), X1))
    def F_G(self, X, w_m, a_m, dt):
        # repare: nenhum termo depende do estado estimado X — só da IMU
        Gm = Exp3(w_m * dt); A = hat(a_m); Z = np.zeros((3, 3))
        F = np.block([[Gm.T, Z, Z], [-Gm.T @ A * dt, Gm.T, Z], [-0.5 * Gm.T @ A * dt**2, Gm.T * dt, Gm.T]])
        Gn = np.block([[-Jr(w_m * dt) * dt, Z], [Z, -Gm.T * dt], [Z, -0.5 * Gm.T * dt**2]])
        return F, Gn
    def H_pos(self, X): return np.hstack([np.zeros((3, 6)), self.para_Rvp(X)[0]])
    def ad(self, d):
        M = np.zeros((9, 9)); th, nu, rho = d[:3], d[3:6], d[6:]
        M[:3, :3] = hat(th); M[3:6, :3] = hat(nu); M[3:6, 3:6] = hat(th); M[6:, :3] = hat(rho); M[6:, 6:] = hat(th)
        return M

def propaga_grupo(g, X, w_m, a_m, dt):
    return g.de_Rvp(*propaga_fisica(*g.para_Rvp(X), w_m, a_m, dt))

def propaga_tridente(tau, w_m, a_m, dt):
    """A mesma propagação escrita só com produtos de tridentes: tau+ = Gama ⊗ Phi(tau) ⊗ Upsilon."""
    gama = tq(np.array([1.0, 0, 0, 0]), G_VEC * dt, 0.5 * G_VEC * dt**2)       # gravidade (à esquerda)
    ups = tq(qExp(w_m * dt), a_m * dt, 0.5 * a_m * dt**2)                         # incremento da IMU (à direita)
    r, a, b = tau
    phi = (r, a, b + a * dt)                                                      # Phi: p <- p + v dt
    return tqmul(tqmul(gama, phi), ups)

# ---------------- o filtro (o mesmo código para os três grupos) ----------------
def H_e_residuo(g, X, z, tipo):
    """tipo 'pose': z = (R_med, p_med); tipo 'pos': z = p_med. Resíduo no espaço da medida."""
    R, v, p = g.para_Rvp(X)
    if tipo == "pose":
        Ha = g.H_att(X) if hasattr(g, "H_att") else np.hstack([np.eye(3), np.zeros((3, 6))])
        H = np.vstack([Ha, g.H_pos(X)])
        r = np.concatenate([Log3(R.T @ z[0]), z[1] - p])
    else:
        H = g.H_pos(X); r = z - p
    return H, r

def eskf(g, sim, med, x0, P0, sig_g, sig_a, Rmed, tipo, usa_reset=True):
    """med: dicionário {índice do passo: medida}. x0 = (R, v, p) inicial estimado."""
    X = g.de_Rvp(*x0); P = P0.copy(); Qn = np.diag([sig_g**2] * 3 + [sig_a**2] * 3)
    est, covs = [X], [P.copy()]
    for k in range(len(sim["w_m"])):
        F, Gn = g.F_G(X, sim["w_m"][k], sim["a_m"][k], sim["dt"])
        X = propaga_grupo(g, X, sim["w_m"][k], sim["a_m"][k], sim["dt"])
        P = F @ P @ F.T + Gn @ Qn @ Gn.T
        if k + 1 in med:
            H, r = H_e_residuo(g, X, med[k + 1], tipo)
            S = H @ P @ H.T + Rmed
            K = np.linalg.solve(S, H @ P).T
            d = K @ r
            X = g.boxplus(X, d)
            P = (np.eye(9) - K @ H) @ P
            if usa_reset:
                Gr = np.eye(9) - 0.5 * g.ad(d); P = Gr @ P @ Gr.T
            P = 0.5 * (P + P.T)
        est.append(X); covs.append(P.copy())
    return est, np.array(covs)

def Ad23(R, v, p):
    Z = np.zeros((3, 3))
    return np.block([[R, Z, Z], [hat(v) @ R, R, Z], [hat(p) @ R, Z, R]])

class GrupoSE23Dir(GrupoSE23):
    """(d) tridente com erro à DIREITA (no referencial do mundo): X = Exp(δ) ⊗ X̂  (RQEKF de Ouyang e Wu)."""
    nome = "SE₂(3) tridente, erro à direita (RQEKF)"
    def boxplus(self, X, d): return tqmul(tqExp(d), X)
    def boxminus(self, X1, X2): return tqLog(tqmul(X1, tqconj(X2)))
    def F_G(self, X, w_m, a_m, dt):
        # F não depende nem do estado nem da IMU: só da gravidade e de dt
        I = np.eye(3); Z = np.zeros((3, 3))
        MPhi = np.block([[I, Z, Z], [Z, I, Z], [Z, I * dt, I]])
        F = Ad23(I, G_VEC * dt, 0.5 * G_VEC * dt**2) @ MPhi
        _, Gl = GrupoSE23.F_G(self, X, w_m, a_m, dt)
        Xn = self.para_Rvp(propaga_grupo(self, X, w_m, a_m, dt))
        return F, Ad23(*Xn) @ Gl                    # o ruído entra "girado" pelo estado
    def H_att(self, X): return np.hstack([self.para_Rvp(X)[0].T, np.zeros((3, 6))])
    def H_pos(self, X):
        p = self.para_Rvp(X)[2]; return np.hstack([-hat(p), np.zeros((3, 3)), np.eye(3)])
    def ad(self, d): return -GrupoSE23.ad(self, d)  # o reset à direita tem o sinal trocado: G ≈ I + ½ ad
