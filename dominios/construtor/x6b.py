from comum import *
from q_codigo import FUNCOES_Q

c = Caderno(arquivo("6", "b"))
cabecalho(c, "Domínio VI (b): sistemas de equações com quaternions",
          "A orientação de um satélite como um autovetor, nuvens de pontos 3D e a média de rotações",
          dom="6", etapa="b", anterior=arquivo("6", "a"), proximo=arquivo("6", "c"))

c.md(r"""
**Objetivo.** No domínio V, alinhar duas nuvens de pontos do plano (o problema de Procrustes) coube numa linha de números complexos. No espaço, o problema correspondente tem uma solução tão elegante quanto, e mais surpreendente: a rotação ótima é o **autovetor** de uma matriz simétrica $4\times4$. Um problema não linear, com a restrição $R^\top R=I$, vira um problema de autovalores, que a álgebra linear resolve de forma confiável. A razão é que $R(q)$ é **quadrática** em $q$: qualquer soma de produtos escalares $\mathbf r\cdot R(q)\,\mathbf b$ é uma forma quadrática $q^\top Kq$, e maximizar uma forma quadrática na esfera unitária é achar o maior autovalor.

**Roteiro**
1. O problema de Wahba: a atitude de um satélite a partir de direções medidas
2. Três soluções: TRIAD, o método q de Davenport, e o SVD
3. Quão bem: a covariância da atitude, conferida por Monte Carlo
4. Nuvens de pontos 3D: a orientação absoluta de Horn e o ICP
5. A média de rotações: por que somar quaternions não funciona
6. Exercícios
""")

c.code(FERRAMENTAS + r'''
from scipy.spatial import cKDTree
rng = np.random.default_rng(62)
''' + FUNCOES_Q + r'''
def unit(v): return np.asarray(v, float) / np.linalg.norm(v, axis=-1, keepdims=True)
def angulo_entre(a, b): return np.linalg.norm(qlog(qmul(qconj(a), b)))
''')

# ------------------------------------------------------------- 1
c.md(r"""
---
## 1. O problema de Wahba

Um satélite conhece, no referencial do mundo (inercial), as direções $\mathbf r_k$ de alguns objetos: o Sol, o campo magnético da Terra no ponto da órbita, algumas estrelas. Os sensores a bordo medem as mesmas direções no referencial do satélite: $\mathbf b_k$. Se $R$ é a orientação do satélite (corpo → mundo), sem ruído valeria $\mathbf b_k=R^\top\mathbf r_k$. Com ruído, Grace Wahba (1965) propôs a rotação que minimiza

$$L(R)=\tfrac12\sum_k a_k\,\lvert\mathbf b_k-R^\top\mathbf r_k\rvert^2=\text{const}-\sum_k a_k\,\mathbf r_k\cdot R\,\mathbf b_k,$$

com pesos $a_k=1/\sigma_k^2$. Minimizar $L$ é maximizar $g(R)=\sum_k a_k\,\mathbf r_k\cdot R\,\mathbf b_k$. O cenário: um sensor de Sol (0,5°), um magnetômetro (2°, o campo da Terra é mal modelado) e um rastreador de estrelas vendo três estrelas (10 segundos de arco, ou seja, 0,003°).
""")

c.code(r'''
q_ver = qnorm(rng.normal(size=4))
R_ver = qmat(q_ver)
SIG = np.radians(np.array([0.5, 2.0, 10 / 3600, 10 / 3600, 10 / 3600]))
nomes = ["Sol", "campo magnético", "estrela 1", "estrela 2", "estrela 3"]
r = unit(rng.normal(size=(5, 3)))                         # direções conhecidas no mundo
r[3:] = unit(r[2] + 0.08 * rng.normal(size=(2, 3)))         # as três estrelas estão no mesmo campo de visão (uns 5°)

def mede(r, R, sig, rng):
    """b = Rᵀ r, perturbado por uma pequena rotação aleatória de desvio sig (em cada eixo transversal)."""
    b = r @ R                                               # cada linha: Rᵀ r_k
    out = []
    for bk, s in zip(b, sig):
        out.append(qrot(qexp(s * rng.normal(size=3)), bk))
    return unit(np.array(out))

b = mede(r, R_ver, SIG, rng)
a = 1 / SIG**2
sep = lambda u, v: np.degrees(np.arccos(np.clip(u @ v, -1, 1)))
print(f"separações: Sol–campo {sep(r[0], r[1]):.1f}°;  estrela 1–2 {sep(r[2], r[3]):.1f}°, 1–3 {sep(r[2], r[4]):.1f}°;  Sol–estrela 1 {sep(r[0], r[2]):.1f}°")
for k_ in range(5):
    print(f"{nomes[k_]:16s}: σ = {np.degrees(SIG[k_]) * 3600:8.1f}″;  erro desta medida {np.degrees(np.arccos(np.clip(b[k_] @ (R_ver.T @ r[k_]), -1, 1))) * 3600:8.1f}″")
''')

# ------------------------------------------------------------- 2
c.md(r"""
---
## 2. Três soluções

**TRIAD** (anos 1960, ainda usado): usa só **duas** direções. Monta-se um triedro ortonormal com $\mathbf r_1$, $\mathbf r_1\times\mathbf r_2$ e o terceiro eixo, e o mesmo com $\mathbf b_1,\mathbf b_2$; a rotação leva um triedro no outro. A primeira direção é obedecida exatamente, a segunda só no que sobra. É rápido e não usa os pesos.

**O método q de Davenport** (1968): como $R(q)$ é quadrática em $q$, $g=q^\top Kq$ com

$$B=\sum_k a_k\,\mathbf b_k\mathbf r_k^\top,\quad \sigma=\mathrm{tr}\,B,\quad S=B+B^\top,\quad \mathbf z=\begin{bmatrix}B_{23}-B_{32}\\B_{31}-B_{13}\\B_{12}-B_{21}\end{bmatrix},\quad K=\begin{bmatrix}S-\sigma I&\mathbf z\\\mathbf z^\top&\sigma\end{bmatrix},$$

escrita para $q=(\mathbf e,q_4)$, com a parte vetorial primeiro, como na literatura de atitude. O máximo de $q^\top Kq$ com $|q|=1$ é o autovetor do **maior autovalor**, e é a atitude ótima. Não há chute inicial, não há iteração, não há mínimo local.

**O SVD** (Markley, 1988): com $B=U\Sigma V^\top$, a atitude (mundo → corpo) é $U\,\mathrm{diag}(1,1,\det U\det V)\,V^\top$. É o mesmo Procrustes do domínio V, em 3D.

O método q e o SVD resolvem o **mesmo** problema exatamente; têm de dar a mesma rotação.
""")

c.code(r'''
def triad(r1, r2, b1, b2):
    def triedro(u, v):
        t1 = unit(u); t2 = unit(np.cross(u, v)); return np.column_stack([t1, t2, np.cross(t1, t2)])
    return triedro(r1, r2) @ triedro(b1, b2).T            # R (corpo → mundo): leva o triedro de b no de r

def davenport(r, b, a):
    B = sum(ak * np.outer(bk, rk) for ak, rk, bk in zip(a, r, b))
    sig = np.trace(B); S = B + B.T
    z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    K = np.zeros((4, 4)); K[:3, :3] = S - sig * np.eye(3); K[:3, 3] = z; K[3, :3] = z; K[3, 3] = sig
    lam, V = np.linalg.eigh(K)
    e, q4 = V[:3, -1], V[3, -1]
    q = np.concatenate([[q4], e])                          # para a ordem (w, x, y, z) deste domínio
    return (q if q[0] >= 0 else -q), lam

def wahba_svd(r, b, a):
    B = sum(ak * np.outer(bk, rk) for ak, rk, bk in zip(a, r, b))
    U, s, Vt = np.linalg.svd(B)
    A = U @ np.diag([1, 1, np.linalg.det(U) * np.linalg.det(Vt)]) @ Vt     # atitude mundo → corpo
    return A.T

q_dav, lam = davenport(r, b, a)
R_svd = wahba_svd(r, b, a)
R_tri = triad(r[0], r[1], b[0], b[1])
R_tri_estrelas = triad(r[2], r[3], b[2], b[3])
confere("o método q e o SVD dão a mesma rotação", qmat(q_dav), R_svd, tol=1e-9)
confere("o autovalor máximo é o valor máximo de g", lam[-1], sum(ak * rk @ (R_svd @ bk) for ak, rk, bk in zip(a, r, b)), tol=1e-6, relativo=True)
erro = lambda R_: np.degrees(angulo_entre(mat_q(R_), q_ver)) * 3600
print(f"erro de atitude:  Davenport {erro(qmat(q_dav)):7.1f}″;  SVD {erro(R_svd):7.1f}″;  "
      f"TRIAD (Sol, campo) {erro(R_tri):8.1f}″;  TRIAD (duas estrelas) {erro(R_tri_estrelas):7.1f}″")
''')

c.md(r"""
**Para pensar.** O TRIAD com o Sol e o campo magnético erra quase 2000″, porque as duas direções estão a só 15° uma da outra: o giro em torno do Sol é decidido pela componente do campo perpendicular ao Sol, que é pequena, e o erro de 2° do magnetômetro é ampliado por $1/\mathrm{sen}\,15°\approx4$. Com duas estrelas, o TRIAD fica perto do ótimo, porque elas são medidas com uma precisão centenas de vezes maior. O método q, com as cinco medidas e os pesos certos, é melhor que os dois. A seção 3 mede quanto, com muitas repetições.
""")

# ------------------------------------------------------------- 3
c.md(r"""
---
## 3. Quão bem: a covariância da atitude

Escrevendo o erro como uma pequena rotação no corpo, $R_{\text{est}}=R\,\mathrm{Exp}(\boldsymbol\delta)$, a covariância de $\boldsymbol\delta$ para o estimador de Wahba é, em primeira ordem (Shuster, 1989),

$$P_{\delta\delta}=\Big(\sum_k\frac1{\sigma_k^2}\big(I-\mathbf b_k\mathbf b_k^\top\big)\Big)^{-1}.$$

Cada medida informa só sobre os **dois** eixos perpendiculares a ela ($I-\mathbf b\mathbf b^\top$ projeta no plano transversal): girar em torno da própria direção medida não muda a medida. É a mesma estrutura $\mathbf J^\top\mathbf R^{-1}\mathbf J$ de Gauss–Newton (notebook II(b)) e do filtro de Kalman. Monte Carlo com 4000 repetições confere a fórmula e compara os três métodos.
""")

c.code(r'''
P_teo = np.linalg.inv(sum((1 / s**2) * (np.eye(3) - np.outer(bk, bk)) for s, bk in zip(SIG, r @ R_ver)))
erros = {"Davenport": [], "TRIAD (Sol, campo)": [], "TRIAD (2 estrelas)": []}
for _ in range(4000):
    bb = mede(r, R_ver, SIG, rng)
    qd, _ = davenport(r, bb, a)
    erros["Davenport"].append(qlog(qmul(qconj(q_ver), qd)))
    erros["TRIAD (Sol, campo)"].append(qlog(qmul(qconj(q_ver), mat_q(triad(r[0], r[1], bb[0], bb[1])))))
    erros["TRIAD (2 estrelas)"].append(qlog(qmul(qconj(q_ver), mat_q(triad(r[2], r[3], bb[2], bb[3])))))
D = np.array(erros["Davenport"])
P_mc = np.cov(D.T)
seg = lambda x: np.degrees(x) * 3600
print("desvios-padrão por eixo do corpo (segundos de arco):")
print(f"   fórmula de Shuster:  {seg(np.sqrt(np.diag(P_teo)))}")
print(f"   Monte Carlo:         {seg(np.sqrt(np.diag(P_mc)))}")
confere("a covariância de Monte Carlo bate com a fórmula (a menos de 10%)", np.diag(P_mc) / np.diag(P_teo), np.ones(3), tol=0.1)
for nome, E in erros.items():
    print(f"{nome:20s}: erro RMS {seg(np.sqrt(np.mean(np.sum(np.array(E)**2, axis=1)))):9.1f}″")
w_, V_ = np.linalg.eigh(P_teo)
print(f"o eixo pior determinado faz {np.degrees(np.arccos(abs(V_[:, -1] @ (R_ver.T @ unit(r[2] + r[3]))))):.1f}° com a direção média das estrelas, "
      f"e tem desvio {seg(np.sqrt(w_[-1])):.1f}″, contra {seg(np.sqrt(w_[0])):.1f}″ do melhor")
''')

c.md(r"""
**Leitura.** A fórmula de Shuster prevê desvios de 26″, 34″ e 27″ nos três eixos do corpo, e o Monte Carlo dá 26″, 34″ e 28″: concordam a menos de 3%. Em 4000 repetições, o método q erra 51″ (RMS), o TRIAD com duas estrelas 58″, e o TRIAD com o Sol e o campo magnético 28 mil segundos de arco (quase 8°, o efeito da geometria ruim). Repare que a vantagem do método q sobre o TRIAD com estrelas é de só 13%: quase toda a informação vem das estrelas, e o Sol e o campo, com erros centenas de vezes maiores, pesam pouco.

A covariância não é redonda. O eixo pior determinado faz 2° com a direção média das estrelas, e tem desvio de 50″, contra 6″ do melhor: girar o satélite em torno da linha de visada do rastreador quase não move as estrelas, que estão todas perto dela. É por isso que satélites de precisão levam **dois** rastreadores de estrelas, apontados em direções diferentes. O $I-\mathbf b\mathbf b^\top$ da fórmula diz exatamente isso, e é a mesma conta que o filtro de Kalman da etapa (d) faz a cada medida.
""")

# ------------------------------------------------------------- 4
c.md(r"""
---
## 4. Nuvens de pontos 3D: Horn e o ICP

Um LiDAR, uma câmera de profundidade ou um scanner de laboratório produzem **nuvens de pontos** 3D. Alinhar duas nuvens (uma peça escaneada contra o seu modelo CAD, dois *scans* consecutivos de um robô) é achar a rotação $R$ e a translação $\mathbf t$ que minimizam $\sum_k|\mathbf y_k-(R\,\mathbf x_k+\mathbf t)|^2$ para pares $(\mathbf x_k,\mathbf y_k)$ correspondentes. Horn (1987) mostrou, com quaternions: a translação ótima leva o centroide de $\mathbf x$ no de $\mathbf y$, e a rotação é a de Wahba para os pontos centrados (direções de comprimentos diferentes, com peso 1). É a **orientação absoluta**.

Sem saber quem corresponde a quem, o **ICP** (*Iterative Closest Point*) alterna: casar cada ponto com o vizinho mais próximo na outra nuvem, resolver Horn, repetir. Como no plano (domínio V), o ICP converge para um mínimo local, e a pergunta é: a partir de que rotação inicial ele ainda acerta?

O objeto: pontos sobre a superfície de um elipsoide de semieixos 3, 2 e 1, com uma saliência de um lado (para que não haja simetrias). O **modelo** (como um arquivo CAD) tem 20 000 pontos; a **medida** (o *scan*) tem 1 000 pontos, amostrados de novo (não os mesmos), com ruído de 1 cm, girada e transladada. O ICP leva a medida ao modelo.
""")

c.code(r'''
def objeto(n, rng):
    """Pontos na superfície de um elipsoide (3, 2, 1) com uma saliência perto de +x+y+z."""
    d = unit(rng.normal(size=(n, 3)))
    bump = 1 + 0.6 * np.exp(-np.sum((d - unit([1, 1, 1]))**2, axis=1) / 0.15)
    return d * np.array([3.0, 2.0, 1.0]) * bump[:, None]

def horn(X, Y):
    """Rotação e translação que levam X em Y (pares correspondentes), pelo autovetor de Davenport."""
    cx, cy = X.mean(0), Y.mean(0)
    q, _ = davenport(Y - cy, X - cx, np.ones(len(X)))        # b_k = Rᵀ r_k  ⇔  r = y, b = x
    return q, cy - qmat(q) @ cx

X = objeto(500, rng)
q_t, t_t = qnorm(rng.normal(size=4)), np.array([0.5, -1.0, 2.0])
Y = X @ qmat(q_t).T + t_t + 0.01 * rng.normal(size=X.shape)
qh, th_ = horn(X, Y)
confere("Horn recupera a rotação (pares conhecidos, ruído de 1 cm)", np.degrees(angulo_entre(qh, q_t)) < 0.1, True)
confere("e a translação (a menos de 2 mm)", np.linalg.norm(th_ - t_t) < 2e-3, True)

def icp(X, Y, q0, t0, it_max=200):
    """Alinha a nuvem X à nuvem Y (densa). Devolve (q, t) com Y ≈ R(q) X + t."""
    arvore = cKDTree(Y); q_, t_ = q0, t0
    for it in range(it_max):
        Xt = X @ qmat(q_).T + t_
        d, idx = arvore.query(Xt)
        dq, dt_ = horn(Xt, Y[idx])
        q_, t_ = qmul(dq, q_), qmat(dq) @ t_ + dt_
        if angulo_entre(dq, [1, 0, 0, 0]) < 1e-9: break
    return q_, t_, np.sqrt(np.mean(d**2)), it + 1

# o modelo (CAD): 20 000 pontos; a medida (scan): 1 000 pontos, girados e transladados, com ruído de 1 cm.
# o ICP leva a medida ao modelo; a orientação do objeto é a inversa do resultado.
modelo = objeto(20000, rng)
medida = objeto(1000, rng) @ qmat(q_t).T + t_t + 0.01 * rng.normal(size=(1000, 3))

def alinha(chute):
    q0 = qconj(chute); t0 = modelo.mean(0) - qmat(q0) @ medida.mean(0)
    q_, t_, rms_, n_ = icp(medida, modelo, q0, t0)
    return qconj(q_), rms_, n_

q_icp, rms, n_it = alinha(qmul(q_t, qexp([0.3, -0.2, 0.25])))
print(f"ICP a partir de um chute a {np.degrees(np.linalg.norm([0.3, -0.2, 0.25])):.0f}°: {n_it} iterações, erro de rotação "
      f"{np.degrees(angulo_entre(q_icp, q_t)):.2f}°, distância RMS final {rms * 100:.2f} cm")
confere("o ICP acerta a partir de um chute razoável (erro < 3°)", np.degrees(angulo_entre(q_icp, q_t)) < 3, True)
''')

c.code(r'''
angs = [15, 30, 45, 60, 90, 120, 150, 180]
taxa, erros_ok = [], []
for graus in angs:
    acertos = 0
    for _ in range(20):
        q_, _, _ = alinha(qmul(q_t, qexp(np.radians(graus) * unit(rng.normal(size=3)))))
        e_ = np.degrees(angulo_entre(q_, q_t))
        if e_ < 3: acertos += 1; erros_ok.append(e_)
    taxa.append(acertos / 20)
for g_, t_ in zip(angs, taxa):
    print(f"chute a {g_:3d}° da verdade: o ICP acerta em {100 * t_:3.0f}% das 20 tentativas")
print(f"quando acerta, o erro de rotação fica entre {min(erros_ok):.2f}° e {max(erros_ok):.2f}°")
confere("o ICP sempre acerta com chutes de até 45°", min(taxa[:3]), 1.0)
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(angs, 100 * np.array(taxa), "o-"); ax.set_xlabel("ângulo do chute inicial (graus)"); ax.set_ylabel("acertos (%)")
ax.set_title("ICP 3D com Horn: a bacia de convergência", fontsize=10); plt.tight_layout(); plt.show()
''')

c.md(r"""
**Leitura.** Com os pares conhecidos, Horn recupera rotação e translação em uma conta. Sem os pares, o ICP acerta sempre com chutes de até 60°, em 90% dos casos a 90°, e quase nunca a partir de 120°: a bacia de convergência é bem maior que a do ICP no plano (domínio V), porque o objeto não tem simetrias e a saliência guia o alinhamento. Quando acerta, o erro de rotação fica entre 0,4° e 0,9°, bem acima do que o ruído de 1 cm explicaria. O erro que sobra vem da **amostragem**: o ICP ponto a ponto casa cada ponto medido com um ponto do modelo, e os pontos do modelo ficam a uns 3 cm uns dos outros (é a distância RMS final). Com o modelo mais denso, o erro cai; a variante ponto a plano, que casa cada ponto com o plano tangente do modelo, elimina quase todo esse viés (exercício 6).
""")

# ------------------------------------------------------------- 5
c.md(r"""
---
## 5. A média de rotações

Vários sensores, ou várias medidas no tempo, dão estimativas $q_1,\dots,q_N$ da mesma orientação. Qual é a média? Três candidatas:
- **ingênua:** somar as componentes e normalizar. Funciona quando todos os $q_i$ estão do mesmo lado da esfera, mas $q$ e $-q$ são a mesma rotação, e um algoritmo que devolve "o representante com $w\ge0$" (como `mat_q` ou um sensor) pode entregar sinais misturados;
- **de Markley** (2007): o autovetor do maior autovalor de $M=\sum_iq_iq_i^\top$. Como $q_iq_i^\top=(-q_i)(-q_i)^\top$, o sinal não importa;
- **geodésica** (média de Karcher): o ponto $\bar q$ em que os $\mathrm{Log}(\bar q^*q_i)$ somam zero, achado por Gauss–Newton no grupo: $\bar q\leftarrow\bar q\,\mathrm{Exp}\big(\tfrac1N\sum_i\mathrm{Log}(\bar q^*q_i)\big)$.

O teste: 50 medidas espalhadas (10° de desvio) em torno de uma orientação de 178°, perto do ponto em que $w$ troca de sinal.
""")

c.code(r'''
q_media = qexp(np.radians(178) * unit([1, -2, 0.5]))
amostras = np.array([qmul(q_media, qexp(np.radians(10) * rng.normal(size=3))) for _ in range(50)])
amostras = np.array([q_ if q_[0] >= 0 else -q_ for q_ in amostras])          # como um sensor entregaria
print(f"{np.sum(np.sign(amostras[:, 1]) != np.sign(amostras[0, 1]))} das 50 amostras têm a parte vetorial do lado oposto da primeira")

ingenua = qnorm(amostras.mean(0))
M = amostras.T @ amostras
markley = np.linalg.eigh(M)[1][:, -1]
karcher = amostras[0].copy()
for it in range(50):
    passo = np.mean([qlog(qmul(qconj(karcher), q_)) for q_ in amostras], axis=0)
    karcher = qmul(karcher, qexp(passo))
    if np.linalg.norm(passo) < 1e-14: break
for nome, qq in [("ingênua", ingenua), ("Markley", markley), ("Karcher", karcher)]:
    print(f"{nome:8s}: a {np.degrees(angulo_entre(qq, q_media)):7.3f}° da orientação verdadeira")
print(f"Markley e Karcher diferem de {np.degrees(angulo_entre(markley, karcher)):.3f}°")
confere("Markley e Karcher concordam a menos de 0,2° (são estimadores diferentes, que coincidem quando o espalhamento é pequeno)", np.degrees(angulo_entre(markley, karcher)) < 0.2, True)
confere("e estão perto da verdade (o erro esperado é da ordem de 10°/√50 ≈ 1,4° por eixo)", np.degrees(angulo_entre(karcher, q_media)) < 5, True)
confere("a média ingênua erra muito", np.degrees(angulo_entre(ingenua, q_media)) > 20, True)
print(f"Karcher convergiu em {it + 1} iterações de Gauss–Newton no grupo")
''')

c.md(r"""
**Leitura.** Com a orientação média a 178°, quase metade das amostras chega com o sinal "trocado", e a média ingênua das componentes, que soma quaternions quase opostos, erra 36°. Markley e Karcher ignoram o sinal (um pelo produto $qq^\top$, o outro pelo $\mathrm{Log}$ que escolhe o representante) e erram 2,5°, dentro do esperado para 50 amostras com 10° de desvio. Os dois diferem entre si de só 0,05°: o de Markley minimiza a distância **cordal** (em $\mathbb R^4$), o de Karcher a **geodésica** (sobre a esfera), e as duas quase coincidem quando as amostras estão concentradas. O de Markley custa um autovetor; o de Karcher, aqui, 8 iterações.

---
## 6. Exercícios

1. **Alinhamento de proteínas.** O RMSD entre duas conformações de uma proteína é a distância RMS depois do melhor alinhamento rígido dos átomos (o algoritmo de Kabsch, que é o SVD, ou o de Horn, com quaternions). Baixe duas estruturas da mesma proteína do Protein Data Bank (o mesmo número de átomos de carbono alfa), alinhe-as com `horn` e compare o RMSD com o de `wahba_svd`.
2. **Um pulso esférico.** Um braço robótico com três juntas de rotação cujos eixos se cruzam num ponto (um punho esférico, ZYZ) orienta a ferramenta. Resolva a cinemática inversa da orientação: dado um quaternion, ache os três ângulos. Onde está a singularidade? Relacione com o *gimbal lock* do notebook VI(a).
3. **Identificar as estrelas.** Um rastreador de estrelas não sabe **quais** estrelas está vendo. Monte um catálogo sintético de 500 estrelas, gire-o, meça as 6 mais brilhantes num campo de 10°, e identifique-as comparando os ângulos entre pares (eles não dependem da atitude). Depois, aplique Davenport.
4. **Fractais de quaternions.** Repita o fractal de Newton do notebook IV(b) para $q^2+1=0$ em quaternions: quantas raízes há? (Dica: todo quaternion puro unitário $\mathbf u$ satisfaz $\mathbf u^2=-1$.)
5. **TRIAD na outra ordem.** Troque a ordem do Sol e do campo magnético no TRIAD. O erro muda? Qual direção deve ir primeiro, e por quê?
6. **ICP ponto a plano.** Troque o resíduo $\lvert\mathbf y-(R\mathbf x+\mathbf t)\rvert^2$ por $\big(\mathbf n_y\cdot(\mathbf y-R\mathbf x-\mathbf t)\big)^2$, com $\mathbf n_y$ a normal do modelo no ponto casado (estime-a pela direção de menor variância dos vizinhos, por PCA). Linearize a rotação ($R\approx I+[\boldsymbol\delta]_\times$) e resolva o sistema $6\times6$ a cada iteração. O erro final cai? E o número de iterações?
""")

c.salva()
