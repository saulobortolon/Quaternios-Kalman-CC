import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Pepé Le Gambá e o Filtro de Kalman: gerador do vídeo do IMM (Parte 2)

Este notebook gera um vídeo que compara as maneiras de Pepé perseguir Penélope no **mundo de três modos** (escondida, fugindo, procurando):

- **modelo único** (o Pepé da Parte 1) × **IMM de 2 modos** × **IMM de 3 modos**;
- mirando onde ela **está** × onde ela **estará**.

Ele é **independente**: traz todo o código da simulação, do Kalman iterado e do IMM, e não precisa dos outros notebooks.

**Em cada quadro do vídeo**

- **Arena (à esquerda):**
  - Penélope é uma estrela com a cor do modo **verdadeiro**: cinza escondida, amarela fugindo, azul procurando;
  - Pepé é o triângulo roxo;
  - cada filtro do IMM aparece como um ponto na cor do seu modo, com **tamanho proporcional à probabilidade** $\mu_j$ que Pepé lhe atribui;
  - a estimativa combinada aparece com a elipse de 2σ, e o **destino** escolhido por Pepé com um ×;
  - o círculo laranja pontilhado é o alcance do som em volta de Penélope.
- **Painel de cima, à direita:** as probabilidades de modo empilhadas ao longo do tempo, com o modo verdadeiro numa faixa fina acima.
- **Painel de baixo, à direita:** o erro da estimativa contra a incerteza declarada; em cinza, os momentos em que Pepé **não ouve** o coração.

**Como usar**

1. Rode a verificação (seção 0).
2. Mude parâmetros (seção 1) e configurações (seção 2) se quiser.
3. Rode a prévia (seção 5).
4. Rode a renderização (seção 7).

Com os valores padrão, cada trecho leva de 1 a 2 minutos no Colab. `MODO_RAPIDO = True` reduz o tempo a cerca de um terço.
""")

md(r"""
---
# 0. Verificação do ambiente

O vídeo é montado pelo `ffmpeg`, que já vem instalado no Google Colab. Num computador pessoal: `sudo apt install ffmpeg` (Ubuntu), `brew install ffmpeg` (macOS) ou `conda install -c conda-forge ffmpeg` (Anaconda).
""")

code(r"""
import os, shutil, subprocess, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from IPython.display import Video, display

if shutil.which("ffmpeg") is None:
    print("ffmpeg NÃO encontrado. Instale-o antes de renderizar.")
else:
    print("ffmpeg encontrado:", subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0])
""")

# ---------------------------------------------------------------------
md(r"""
---
# 1. Parâmetros que você pode mudar

Sugestões de experimentos:

- **Penélope mais previsível**: durações de fuga e de procura mais curtas e regulares;
- **Pepé com outra cadeia de Markov**: mude `DURACOES_IMAGINADAS` e veja como mudam o atraso para perceber as trocas de modo e o acerto;
- **vento**: aumente ou diminua `SIG_W` e `SIG_I`. Com pouco vento, o modelo importa menos;
- **mundo de 2 modos**: `MODOS_DO_MUNDO = 2` faz Penélope se esconder logo após fugir, sem procurar.
""")

code(r"""
# ---------------- o mundo ----------------
SEED          = 4        # semente: posição inicial de Penélope e todos os sorteios
MODOS_DO_MUNDO = 3       # 3: escondida → fugindo → procurando → escondida;  2: sem a procura
DT, T_TOTAL   = 0.2, 90.0
V_PEPE        = 1.0
DIST_INICIAL  = 15.0
BORDA         = 40.0

GATILHO      = (2.0, 4.5)   # distância de Pepé que dispara a fuga
VEL_FUGA     = (2.5, 3.5)
DUR_FUGA     = (3.0, 5.0)   # s
DESVIO_FUGA  = 35           # graus, em torno de "para longe de Pepé"
DUR_PROCURA  = (3.0, 6.0)   # s
RAIO_PROCURA = 2.5
VEL_PROCURA  = 1.5

# ---------------- sensores (Fase 3, dia de vento) ----------------
A_C, SIG_C = 1.0, 6.0                 # cheiro
B_S, EPS_S, R_SOM = 1.0, 1.0, 6.0     # coração
SIG_W, SIG_I = 0.02, 0.05             # vento: gradiente e intensidade do cheiro
SIG_SG, SIG_SI = 0.002, 0.003         # ruído do som: gradiente e intensidade

# ---------------- Pepé ----------------
Q_UNICO = 1.0                          # "humildade" do modelo único (Parte 1)
# modelos de cada modo
ESCONDIDA_QP, ESCONDIDA_VAR_V = 1e-3, 4.0   # parada; velocidade de arranque incerta
FUGINDO_Q = 0.5                             # velocidade constante
PROCURANDO_A, PROCURANDO_VAR_V = 0.6, 1.5   # velocidade que se renova (Singer)
# cadeia de Markov imaginada: durações médias (s) de escondida, fugindo, procurando
DURACOES_IMAGINADAS = [6.0, 4.0, 4.5]
SAIDA_DA_PROCURA = {0: 0.8, 1: 0.2}         # ao sair de "procurando": 80% esconde-se, 20% foge
N_ITER  = 5                                  # iterações do Kalman iterado em cada filtro
TAU_MAX = 4.0                                # horizonte máximo de "onde ela estará" (s)

# ---------------- o vídeo ----------------
VELOCIDADE_VIDEO = 3.0
MODO_RAPIDO      = False
DURACAO_CARTELA  = 3.0
N_MC             = 24      # perseguições para a tabela-resumo (0 = sem tabela)
N_MC_PREVISAO    = 8       # perseguições para medir o erro de previsão a 4 s (0 = não medir)
PASTA            = "video_pepe_imm"
ARQUIVO_FINAL    = "pepe_imm_configuracoes.mp4"
""")

# ---------------------------------------------------------------------
md(r"""
---
# 2. As configurações (um trecho do vídeo para cada uma)

Cada entrada de `configs` vira um trecho. O valor é um `dict` com:

| chave | valores | significado |
|---|---|---|
| `pepe` | `"unico"`, `"imm2"`, `"imm3"` | modelo único da Parte 1, IMM de 2 modos ou IMM de 3 modos |
| `alvo` | `"esta"` ou `"estara"` | Pepé anda para onde ela está ou para onde ela estará quando ele chegar |

O dicionário `descricoes` dá duas linhas de texto para a cartela de cada trecho.
""")

code(r"""
configs = {
    "Modelo único, mira onde ela está":   dict(pepe="unico", alvo="esta"),
    "Modelo único, mira onde ela estará": dict(pepe="unico", alvo="estara"),
    "IMM de 2 modos, mira onde ela está": dict(pepe="imm2", alvo="esta"),
    "IMM de 3 modos, mira onde ela está": dict(pepe="imm3", alvo="esta"),
    "IMM de 3 modos, mira onde ela estará": dict(pepe="imm3", alvo="estara"),
}
descricoes = {
    "Modelo único, mira onde ela está":   ["O Pepé da Parte 1: um só Kalman iterado, velocidade quase constante.",
                                           "Para prever, ele só sabe extrapolar a linha reta."],
    "Modelo único, mira onde ela estará": ["Mesmo filtro, mas mira onde ela estará quando ele chegar.",
                                           "Durante a fuga, a previsão segue a reta para além do fim da fuga."],
    "IMM de 2 modos, mira onde ela está": ["Dois filtros: escondida e fugindo, ligados por uma cadeia de Markov.",
                                           "Não tem um modelo para a procura de esconderijo."],
    "IMM de 3 modos, mira onde ela está": ["Três filtros: escondida, fugindo e procurando.",
                                           "Os pontos coloridos são os filtros; o tamanho é a probabilidade do modo."],
    "IMM de 3 modos, mira onde ela estará": ["O Pepé completo: imagina o futuro com a cadeia de Markov.",
                                             "O × marca para onde ele anda: onde ela deverá estar quando ele chegar."],
}
print(len(configs), "configurações")
""")

# ---------------------------------------------------------------------
md(r"""
---
# 3. O mundo, os sensores, o Kalman iterado e o IMM

É o mesmo código da Parte 2, lendo os parâmetros da seção 1. As explicações linha a linha estão lá. **Se você mudar parâmetros, rode esta célula de novo.**
""")

code(r"""
N_PASSOS = int(round(T_TOTAL / DT))
ESC, FUG, PRO = 0, 1, 2
NOMES_MODOS = ["escondida", "fugindo", "procurando"]
CORES_MODOS = ["0.3", "gold", "deepskyblue"]

# ---------- Penélope ----------
def inicia_fuga(pen, xP, rng):
    longe = np.arctan2(*(pen["p"] - xP)[::-1]); ang = longe + rng.normal(0, np.deg2rad(DESVIO_FUGA))
    pen.update(modo=FUG, t_rest=rng.uniform(*DUR_FUGA), v=rng.uniform(*VEL_FUGA) * np.array([np.cos(ang), np.sin(ang)]))

def novo_ponto_de_procura(pen, rng):
    r, a = RAIO_PROCURA * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
    pen["alvo"] = pen["centro"] + r * np.array([np.cos(a), np.sin(a)])

def passo_penelope(pen, xP, rng):
    perto = np.linalg.norm(xP - pen["p"]) < pen["gatilho"]
    if pen["modo"] in (ESC, PRO) and perto:
        inicia_fuga(pen, xP, rng)
    elif pen["modo"] == FUG:
        pen["t_rest"] -= DT
        if pen["t_rest"] <= 0:
            if MODOS_DO_MUNDO == 3:
                pen.update(modo=PRO, t_rest=rng.uniform(*DUR_PROCURA), centro=pen["p"].copy()); novo_ponto_de_procura(pen, rng)
            else:
                pen.update(modo=ESC, v=np.zeros(2), gatilho=rng.uniform(*GATILHO))
    elif pen["modo"] == PRO:
        pen["t_rest"] -= DT
        if pen["t_rest"] <= 0:
            pen.update(modo=ESC, v=np.zeros(2), gatilho=rng.uniform(*GATILHO))
        else:
            d = pen["alvo"] - pen["p"]
            if np.linalg.norm(d) < VEL_PROCURA * DT: novo_ponto_de_procura(pen, rng); d = pen["alvo"] - pen["p"]
            pen["v"] = VEL_PROCURA * d / np.linalg.norm(d)
    pen["p"] = pen["p"] + pen["v"] * DT
    for k in range(2):
        if abs(pen["p"][k]) > BORDA: pen["p"][k] = np.sign(pen["p"][k]) * BORDA; pen["v"][k] *= -1
    return pen

# ---------- sensores ----------
def cheiro(x, p): d = x - p; return A_C * np.exp(-(d @ d) / (2 * SIG_C**2))
def grad_cheiro(x, p): return -cheiro(x, p) / SIG_C**2 * (x - p)
def som(x, p): d = x - p; return B_S / (d @ d + EPS_S**2)
def grad_som(x, p): d = x - p; return -2 * B_S * d / (d @ d + EPS_S**2)**2

def mede(xP, p, rng):
    z = [grad_cheiro(xP, p) + SIG_W * rng.standard_normal(2), [cheiro(xP, p) + SIG_I * rng.standard_normal()]]
    var = [SIG_W**2, SIG_W**2, SIG_I**2]
    ouviu = np.linalg.norm(xP - p) < R_SOM
    if ouviu:
        z += [grad_som(xP, p) + SIG_SG * rng.standard_normal(2), [som(xP, p) + SIG_SI * rng.standard_normal()]]
        var += [SIG_SG**2, SIG_SG**2, SIG_SI**2]
    return np.concatenate(z), np.diag(var), ouviu

def h_modelo(s, xP, ouviu):
    p = s[:2]; partes = [grad_cheiro(xP, p), [cheiro(xP, p)]]
    if ouviu: partes += [grad_som(xP, p), [som(xP, p)]]
    return np.concatenate(partes)

def J_modelo(s, xP, ouviu):
    p = s[:2]; d = xP - p; c = cheiro(xP, p)
    L = [c / SIG_C**2 * (np.eye(2) - np.outer(d, d) / SIG_C**2), (c * d / SIG_C**2)[None, :]]
    if ouviu:
        D = d @ d + EPS_S**2
        L += [2 * B_S / D**2 * (np.eye(2) - 4 * np.outer(d, d) / D), (2 * B_S * d / D**2)[None, :]]
    Jp = np.vstack(L); return np.hstack([Jp, np.zeros_like(Jp)])

# ---------- modelos de movimento e cadeia de Markov ----------
I2, Z2 = np.eye(2), np.zeros((2, 2))
def modelo_escondida():
    return np.block([[I2, Z2], [Z2, Z2]]), np.block([[ESCONDIDA_QP * DT * I2, Z2], [Z2, ESCONDIDA_VAR_V * I2]])
def modelo_cv(q):
    return (np.block([[I2, DT * I2], [Z2, I2]]),
            q * np.block([[DT**3 / 3 * I2, DT**2 / 2 * I2], [DT**2 / 2 * I2, DT * I2]]))
def modelo_procurando():
    return np.block([[I2, DT * I2], [Z2, PROCURANDO_A * I2]]), np.block([[1e-3 * I2, Z2], [Z2, PROCURANDO_VAR_V * I2]])

def matriz_de_transicao(duracoes, destinos):
    M = len(duracoes); Pi = np.zeros((M, M))
    for i, (d, dest) in enumerate(zip(duracoes, destinos)):
        Pi[i, i] = 1 - DT / d
        for j, w in dest.items(): Pi[i, j] += (DT / d) * w
    return Pi

def constroi_pepe(tipo):
    if tipo == "unico":
        return dict(modelos=[modelo_cv(Q_UNICO)], Pi=np.ones((1, 1)), mu0=[1.0])
    if tipo == "imm2":
        return dict(modelos=[modelo_escondida(), modelo_cv(FUGINDO_Q)],
                    Pi=matriz_de_transicao(DURACOES_IMAGINADAS[:2], [{1: 1.0}, {0: 1.0}]), mu0=[0.9, 0.1])
    if tipo == "imm3":
        return dict(modelos=[modelo_escondida(), modelo_cv(FUGINDO_Q), modelo_procurando()],
                    Pi=matriz_de_transicao(DURACOES_IMAGINADAS, [{1: 1.0}, {2: 1.0}, SAIDA_DA_PROCURA]), mu0=[0.8, 0.1, 0.1])
    raise ValueError(tipo)

# ---------- Kalman iterado e IMM ----------
def atualizacao_iterada(m, P, z, R, h, J, n_iter=N_ITER, tol=1e-9):
    s = m.copy()
    for _ in range(n_iter):
        H = J(s); S = H @ P @ H.T + R; K = np.linalg.solve(S, H @ P).T
        s_novo = m + K @ (z - h(s) - H @ (m - s))
        if np.linalg.norm(s_novo - s) < tol: s = s_novo; break
        s = s_novo
    H = J(s); S = H @ P @ H.T + R; K = np.linalg.solve(S, H @ P).T
    v = z - h(s) - H @ (m - s)
    ll = -0.5 * (v @ np.linalg.solve(S, v) + np.linalg.slogdet(2 * np.pi * S)[1])
    P_novo = P - K @ S @ K.T
    return s, (P_novo + P_novo.T) / 2, ll

def mistura(mu, ms, Ps, Pi):
    c = mu @ Pi; W = (Pi * mu[:, None]) / c[None, :]
    m0, P0 = [], []
    for j in range(len(mu)):
        mj = sum(W[i, j] * ms[i] for i in range(len(mu)))
        m0.append(mj); P0.append(sum(W[i, j] * (Ps[i] + np.outer(ms[i] - mj, ms[i] - mj)) for i in range(len(mu))))
    return c, m0, P0

def preve_por_modo(modelos, m0, P0):
    return [F @ m for (F, Q), m in zip(modelos, m0)], [F @ P @ F.T + Q for (F, Q), P in zip(modelos, P0)]

def combina(mu, ms, Ps):
    m = sum(w * x for w, x in zip(mu, ms))
    return m, sum(w * (Pj + np.outer(x - m, x - m)) for w, x, Pj in zip(mu, ms, Ps))

def preve_adiante(mu, ms, Ps, modelos, Pi, n):
    for _ in range(n):
        c, m0, P0 = mistura(mu, ms, Ps, Pi); ms, Ps = preve_por_modo(modelos, m0, P0); mu = c
    return combina(mu, ms, Ps)[0][:2]

def simula(pepe_tipo, alvo="esta", seed=0, guarda_filtros=True):
    pepe = constroi_pepe(pepe_tipo); modelos, Pi = pepe["modelos"], pepe["Pi"]
    mu = np.array(pepe["mu0"], float); M = len(modelos)
    rng = np.random.default_rng(seed); ang0 = rng.uniform(0, 2 * np.pi)
    pen = dict(p=DIST_INICIAL * np.array([np.cos(ang0), np.sin(ang0)]), v=np.zeros(2), modo=ESC, t_rest=0.0,
               gatilho=rng.uniform(*GATILHO))
    xP = np.zeros(2); ms = [np.zeros(4) for _ in range(M)]; Ps = [np.diag([64.0, 64.0, 1.0, 1.0]) for _ in range(M)]
    reg = []
    for k in range(N_PASSOS):
        pen = passo_penelope(pen, xP, rng)
        z, R, ouviu = mede(xP, pen["p"], rng)
        if k > 0:
            c, m0, P0 = mistura(mu, ms, Ps, Pi); ms, Ps = preve_por_modo(modelos, m0, P0)
        else:
            c = mu.copy()
        lls = np.zeros(M)
        for j in range(M):
            ms[j], Ps[j], lls[j] = atualizacao_iterada(ms[j], Ps[j], z, R, lambda s: h_modelo(s, xP, ouviu),
                                                       lambda s: J_modelo(s, xP, ouviu))
        w = c * np.exp(lls - lls.max()); mu = w / w.sum(); mu = np.maximum(mu, 1e-12); mu /= mu.sum()
        m, P = combina(mu, ms, Ps)
        if alvo == "estara":
            n = int(round(min(np.linalg.norm(m[:2] - xP) / V_PEPE, TAU_MAX) / DT))
            destino = preve_adiante(mu, ms, Ps, modelos, Pi, n) if n > 0 else m[:2]
        else:
            destino = m[:2]
        r = dict(t=k * DT, xP=xP.copy(), p=pen["p"].copy(), modo=pen["modo"], ouviu=ouviu, mu=mu.copy(),
                 m=m.copy(), P=P.copy(), destino=destino.copy())
        if guarda_filtros: r.update(ms=[x.copy() for x in ms], Ps=[x.copy() for x in Ps])
        reg.append(r)
        d = destino - xP
        if np.linalg.norm(d) > 1e-9: xP = xP + V_PEPE * DT * d / np.linalg.norm(d)
    return reg

def series(reg):
    out = {"t": np.array([r["t"] for r in reg]), "modo": np.array([r["modo"] for r in reg]),
           "ouviu": np.array([r["ouviu"] for r in reg])}
    out["mu"] = np.array([r["mu"] for r in reg])
    out["erro"] = np.array([np.linalg.norm(r["m"][:2] - r["p"]) for r in reg])
    out["sd"] = np.array([np.sqrt(np.trace(r["P"][:2, :2])) for r in reg])
    out["dist"] = np.array([np.linalg.norm(r["xP"] - r["p"]) for r in reg])
    out["nees"] = np.array([(r["m"][:2] - r["p"]) @ np.linalg.solve(r["P"][:2, :2], r["m"][:2] - r["p"]) for r in reg])
    return out

print(f"{N_PASSOS} passos por perseguição;  matriz de transição do IMM de 3 modos:")
print(np.round(constroi_pepe("imm3")["Pi"], 4))
""")

# ---------------------------------------------------------------------
md(r"""
---
# 4. O desenho de cada quadro e a gravação

Cada quadro é desenhado com o Matplotlib e enviado diretamente ao `ffmpeg` como imagem crua, sem salvar PNGs. Cada trecho vira um `.mp4` separado, e no fim os arquivos são concatenados sem recodificar.

- **A "câmera" acompanha a ação.** A janela da arena contém sempre Pepé, Penélope e o destino. O centro e o tamanho são suavizados por uma média móvel exponencial, para a imagem não tremer: a janela cresce depressa quando é preciso e encolhe devagar.
- **O painel das probabilidades** só mostra o que já aconteceu até o instante do quadro. Para o modelo único, que não tem modos, ele mostra uma faixa única e um aviso.
""")

code(r"""
FPS, DPI = 30, 100
W, H = (854, 480) if MODO_RAPIDO else (1280, 720)
PULO = 2 if MODO_RAPIDO else 1
PASSOS_POR_SEG = VELOCIDADE_VIDEO / DT / PULO
ESCALA = W / 1280
ENC = ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(FPS)]

def janelas(reg, suavidade=0.06):
    # Uma janela por quadro, que acompanha a ação: contém Pepé, Penélope e o destino.
    # O centro e o tamanho são suavizados (média móvel exponencial) para a "câmera" não tremer.
    alvo_c, alvo_L = [], []
    for r in reg:
        pts = np.array([r["xP"], r["p"], r["destino"]])
        lo, hi = pts.min(0), pts.max(0)
        alvo_c.append((lo + hi) / 2); alvo_L.append(max(12.0, 0.5 * (hi - lo).max() + 6))
    c, L, out = alvo_c[0], alvo_L[0], []
    for ac, aL in zip(alvo_c, alvo_L):
        c = c + suavidade * (ac - c)
        L = max(L + suavidade * (aL - L), aL * 0.85)       # cresce depressa se for preciso, encolhe devagar
        out.append((c - L, c + L))
    return out

def nova_figura():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")
    axA = fig.add_axes([0.03, 0.06, 0.50, 0.84])
    axM = fig.add_axes([0.60, 0.53, 0.37, 0.32])
    axE = fig.add_axes([0.60, 0.10, 0.37, 0.30])
    return fig, axA, axM, axE

def prepara(kw):
    reg = simula(kw["pepe"], kw["alvo"], seed=SEED)
    return reg, series(reg), janelas(reg)

def desenha_quadro(fig, axA, axM, axE, nome, kw, reg, S_, k, jans):
    (x0, y0), (x1, y1) = jans[k]; r = reg[k]; fs = ESCALA; M = len(r["mu"])
    gx, gy = np.meshgrid(np.linspace(x0, x1, 120), np.linspace(y0, y1, 120))
    for a in (axA, axM, axE): a.cla()
    fig.texts.clear()
    fig.text(0.03, 0.955, nome, fontsize=16 * fs, fontweight="bold", color="#3b2a63")
    fig.text(0.60, 0.955, f"t = {r['t']:4.1f} s    Penélope: {NOMES_MODOS[r['modo']]}" + ("    Pepé OUVE" if r["ouviu"] else ""),
             fontsize=13 * fs, color={0: "#333333", 1: "#b8860b", 2: "#0277bd"}[r["modo"]])
    # ---- arena ----
    C = A_C * np.exp(-((gx - r["p"][0])**2 + (gy - r["p"][1])**2) / (2 * SIG_C**2))
    axA.imshow(C, extent=[x0, x1, y0, y1], origin="lower", cmap="Purples", alpha=.4, vmin=0, vmax=1.2)
    cam = np.array([x["p"] for x in reg[:k + 1]]); camP = np.array([x["xP"] for x in reg[:k + 1]])
    axA.plot(*cam.T, color="gray", lw=1, ls="--"); axA.plot(*camP.T, color="C4", lw=1.6)
    axA.add_patch(Circle(r["p"], R_SOM, fill=False, ls=":", color="C1", lw=1.3))
    axA.plot(*r["p"], "*", color=CORES_MODOS[r["modo"]], mec="k", ms=20 * fs, zorder=5, label="Penélope (cor = modo verdadeiro)")
    axA.plot(*r["xP"], "^", color="C4", ms=14 * fs, zorder=5, label="Pepé")
    if M > 1:
        for j, x in enumerate(r["ms"]):
            if x0 < x[0] < x1 and y0 < x[1] < y1:
                axA.plot(*x[:2], "o", color=CORES_MODOS[j], mec="k", ms=(3 + 16 * r["mu"][j]) * fs, alpha=.9, zorder=4)
    m = r["m"][:2]
    if x0 < m[0] < x1 and y0 < m[1] < y1:
        w_, V_ = np.linalg.eigh(r["P"][:2, :2]); ang = np.degrees(np.arctan2(V_[1, -1], V_[0, -1]))
        axA.add_patch(Ellipse(m, 4 * np.sqrt(w_[-1]), 4 * np.sqrt(w_[0]), angle=ang, fill=False, color="C3", lw=2))
        axA.plot(*m, "o", color="C3", ms=6 * fs, label="estimativa combinada (elipse 2σ)")
    if kw["alvo"] == "estara" and x0 < r["destino"][0] < x1 and y0 < r["destino"][1] < y1:
        axA.plot(*r["destino"], "x", color="C3", ms=13 * fs, mew=3, label="destino: onde ela estará")
        axA.annotate("", xy=r["destino"], xytext=r["xP"], arrowprops=dict(arrowstyle="->", color="C3", ls="--", lw=1.2))
    axA.set_xlim(x0, x1); axA.set_ylim(y0, y1); axA.set_aspect("equal"); axA.tick_params(labelsize=10 * fs)
    axA.legend(loc="lower left", fontsize=8.5 * fs, framealpha=.85)
    if M > 1:
        axA.text(0.99, 0.01, "bolinhas: um filtro por modo, tamanho ∝ probabilidade", transform=axA.transAxes,
                 ha="right", fontsize=8.5 * fs, color="0.3")
    # ---- probabilidades de modo ----
    tt = S_["t"][:k + 1]
    for j in range(3):
        axM.fill_between(tt, 1.03, 1.12, where=S_["modo"][:k + 1] == j, color=CORES_MODOS[j], step="mid",
                         transform=axM.get_xaxis_transform(), clip_on=False)
    if M > 1:
        axM.stackplot(tt, S_["mu"][:k + 1].T, colors=CORES_MODOS[:M], alpha=.85, step="mid", labels=NOMES_MODOS[:M])
        axM.legend(fontsize=8 * fs, loc="lower right", ncol=M)
        axM.set_title("probabilidades de modo segundo Pepé (faixa acima: modo verdadeiro)", fontsize=10 * fs, pad=22 * fs)
    else:
        axM.text(T_TOTAL / 2, 0.5, "modelo único: Pepé não pensa em modos", ha="center", va="center", fontsize=12 * fs, color="0.4")
        axM.set_title("modo verdadeiro (faixa acima)", fontsize=10 * fs, pad=22 * fs)
    axM.set_xlim(0, T_TOTAL); axM.set_ylim(0, 1); axM.tick_params(labelsize=9 * fs)
    # ---- erro ----
    axE.fill_between(tt, 1e-3, 1e3, where=~S_["ouviu"][:k + 1], color="0.88", step="mid")
    axE.semilogy(tt, np.clip(S_["erro"][:k + 1], 1e-2, 1e3), "C3"); axE.semilogy(tt, 2 * S_["sd"][:k + 1], "k--", lw=1)
    axE.set_xlim(0, T_TOTAL); axE.set_ylim(1e-2, 100); axE.tick_params(labelsize=9 * fs); axE.set_xlabel("t (s)", fontsize=10 * fs)
    axE.set_title("erro (vermelho), 2σ declarado (tracejado); cinza: Pepé não ouve", fontsize=10 * fs)

def grava_trecho(nome, kw, saida):
    reg, S_, jans = prepara(kw)
    fig, axA, axM, axE = nova_figura()
    tmp = saida.replace(".mp4", ".part.mp4")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                             "-r", f"{PASSOS_POR_SEG:.6f}", "-i", "-"] + ENC + [tmp], stdin=subprocess.PIPE)
    for k in range(0, len(reg), PULO):
        desenha_quadro(fig, axA, axM, axE, nome, kw, reg, S_, k, jans)
        fig.canvas.draw(); proc.stdin.write(bytes(fig.canvas.buffer_rgba()))
    plt.close(fig); proc.stdin.close()
    if proc.wait() != 0: raise RuntimeError(f"o ffmpeg falhou no trecho {nome!r}")
    os.replace(tmp, saida)

def png_para_mp4(png, saida, dur):
    tmp = saida.replace(".mp4", ".part.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", png, "-vf",
                    f"format=yuv420p,loop=-1:1,trim=duration={dur},fade=in:0:10,fade=out:st={dur - 0.4}:d=0.4"] + ENC + [tmp], check=True)
    os.replace(tmp, saida)

def cartela(linhas, saida, dur=DURACAO_CARTELA, tamanhos=None):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#1d1530")
    tamanhos = tamanhos or [32] + [19] * (len(linhas) - 1)
    y = 0.62
    for txt, tam in zip(linhas, tamanhos):
        fig.text(0.5, y, txt, ha="center", va="center", color="white", fontsize=tam * ESCALA, fontweight="bold" if tam > 25 else "normal")
        y -= 0.1 if tam > 25 else 0.07
    png = saida.replace(".mp4", ".png"); fig.savefig(png, dpi=DPI, facecolor=fig.get_facecolor()); plt.close(fig)
    png_para_mp4(png, saida, dur)

def cartela_legenda(saida, dur=6.0):
    # uma cartela que explica as cores e os símbolos antes dos trechos
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#f3f0fa"); fs = ESCALA
    fig.text(0.5, 0.86, "Como ler o vídeo", ha="center", color="#3b2a63", fontsize=30 * fs, fontweight="bold")
    itens = [("*", CORES_MODOS[0], "Penélope escondida"), ("*", CORES_MODOS[1], "Penélope fugindo"),
             ("*", CORES_MODOS[2], "Penélope procurando esconderijo"), ("^", "C4", "Pepé"),
             ("o", "C3", "estimativa combinada de Pepé, com a elipse de 2σ"),
             ("o", "0.7", "bolinhas coloridas: um filtro por modo; tamanho = probabilidade do modo"),
             ("x", "C3", "destino: onde Pepé acha que ela estará quando ele chegar")]
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    y = 0.72
    for mk, cor, txt in itens:
        ax.plot(0.16, y, mk, color=cor, mec="k" if mk != "x" else None, ms=20 * fs, mew=3 if mk == "x" else 1)
        ax.text(0.21, y, txt, color="#222222", fontsize=17 * fs, va="center"); y -= 0.085
    png = saida.replace(".mp4", ".png"); fig.savefig(png, dpi=DPI, facecolor=fig.get_facecolor()); plt.close(fig)
    png_para_mp4(png, saida, dur)
""")

# ---------------------------------------------------------------------
md(r"""
---
# 5. Prévia: alguns quadros, em segundos

Mude `CONFIG_PREVIA` e `INSTANTES_PREVIA` à vontade. A célula imprime também os instantes em que Penélope trocou de modo, para ajudar a escolher bons momentos.
""")

code(r"""
CONFIG_PREVIA = "IMM de 3 modos, mira onde ela estará"
INSTANTES_PREVIA = [13.0, 18.0, 30.0]

kw = configs[CONFIG_PREVIA]
reg, S_, jans = prepara(kw)
trocas = [(float(round(S_["t"][k], 1)), NOMES_MODOS[S_["modo"][k]]) for k in range(1, len(S_["t"])) if S_["modo"][k] != S_["modo"][k - 1]]
print("trocas de modo (t, novo modo):", trocas[:12], "..." if len(trocas) > 12 else "")
for t_ in INSTANTES_PREVIA:
    k = min(int(round(t_ / DT)), len(reg) - 1)
    fig, axA, axM, axE = nova_figura()
    desenha_quadro(fig, axA, axM, axE, CONFIG_PREVIA, kw, reg, S_, k, jans)
    plt.show()
""")

# ---------------------------------------------------------------------
md(r"""
---
# 6. A tabela-resumo

Cada configuração é simulada `N_MC` vezes (sementes a partir de `SEED + 100`). As colunas são:

| coluna | o que mede |
|---|---|
| **erro mediano** | quão bem Pepé sabe onde ela está |
| **elipse 2σ** | % do tempo com Penélope dentro da elipse (ideal ≈ 86%) |
| **encontros** | quantas vezes Pepé chegou perto o bastante para ela fugir |
| **acerto de modo** | % do tempo em que o modo mais provável é o verdadeiro |
| **previsão a 4 s** | erro mediano ao prever onde ela estará 4 s depois. É a principal vantagem da cadeia de Markov; medida em `N_MC_PREVISAO` perseguições |

Com os valores padrão, a célula leva de 2 a 4 minutos (a coluna de previsão é a mais cara). Use `N_MC = 0` para pular a tabela, ou `N_MC_PREVISAO = 0` para pular só a previsão.
""")

code(r"""
resumo = []
if N_MC > 0:
    t0 = time.time()
    for nome, kw in configs.items():
        E, CONS, ENCONTROS, AC, PREV = [], [], [], [], []
        for sd in range(N_MC):
            S_mc = series(simula(kw["pepe"], kw["alvo"], seed=SEED + 100 + sd, guarda_filtros=False))
            E.append(np.median(S_mc["erro"])); CONS.append(np.mean(S_mc["nees"] <= 4.0))
            ENCONTROS.append(np.sum((S_mc["modo"][1:] == FUG) & (S_mc["modo"][:-1] != FUG)))
            M = S_mc["mu"].shape[1]
            if M == 3: AC.append(np.mean(S_mc["mu"].argmax(1) == S_mc["modo"]))
            if M == 2: AC.append(np.mean(S_mc["mu"].argmax(1) == np.where(S_mc["modo"] == ESC, 0, 1)))
        H4 = int(round(4.0 / DT)); pep = constroi_pepe(kw["pepe"])
        for sd in range(N_MC_PREVISAO):
            rg = simula(kw["pepe"], kw["alvo"], seed=SEED + 300 + sd)
            for k in range(50, len(rg) - H4, 4):
                r = rg[k]
                PREV.append(np.linalg.norm(preve_adiante(r["mu"], r["ms"], r["Ps"], pep["modelos"], pep["Pi"], H4) - rg[k + H4]["p"]))
        resumo.append(dict(nome=nome, erro=np.median(E), cons=100 * np.mean(CONS), enc=np.mean(ENCONTROS),
                           acerto=100 * np.mean(AC) if AC else None, prev=np.median(PREV) if PREV else None))
        print(f"[{time.time() - t0:4.0f} s] {nome:40s} erro {resumo[-1]['erro']:.2f} | elipse {resumo[-1]['cons']:.0f}% | "
              f"encontros {resumo[-1]['enc']:.1f} | acerto {resumo[-1]['acerto'] if resumo[-1]['acerto'] is None else round(resumo[-1]['acerto'])} | "
              f"previsão 4 s {resumo[-1]['prev'] if resumo[-1]['prev'] is None else round(resumo[-1]['prev'], 2)}")

def br(x, casas=2):
    return "—" if x is None else f"{x:.{casas}f}".replace(".", ",")

def cartela_resumo(saida, dur=10.0):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#1d1530"); fs = ESCALA
    fig.text(0.5, 0.88, f"Resumo em {N_MC} perseguições", ha="center", color="white", fontsize=30 * fs, fontweight="bold")
    xs = [0.04, 0.47, 0.58, 0.69, 0.80, 0.92]
    cab = ["configuração", "erro\nmediano", "elipse\n2σ", "encontros", "acerto\nde modo", "previsão\na 4 s"]
    for x, c in zip(xs, cab):
        fig.text(x, 0.76, c, color="#c9b8ff", fontsize=14 * fs, ha="left" if x == xs[0] else "center", va="center")
    melhor_prev = min((r["prev"] for r in resumo if r["prev"] is not None), default=None)
    y = 0.64; passo_y = min(0.09, 0.55 / max(len(resumo), 1))
    for r in resumo:
        cor = "#a5d6a7" if (r["prev"] is not None and r["prev"] == melhor_prev) else "white"
        vals = [r["nome"], br(r["erro"]), br(r["cons"], 0) + "%", br(r["enc"], 1),
                "—" if r["acerto"] is None else br(r["acerto"], 0) + "%", br(r["prev"])]
        for x, v in zip(xs, vals):
            fig.text(x, y, v, color=cor, fontsize=(15 if x == xs[0] else 17) * fs, ha="left" if x == xs[0] else "center", va="center")
        y -= passo_y
    fig.text(0.5, 0.06, "verde: melhor previsão a 4 s  •  erros em unidades de distância  •  elipse 2σ: ideal ≈ 86%",
             ha="center", color="#bbbbbb", fontsize=12 * fs)
    png = saida.replace(".mp4", ".png"); fig.savefig(png, dpi=DPI, facecolor=fig.get_facecolor()); plt.close(fig)
    png_para_mp4(png, saida, dur)
""")

# ---------------------------------------------------------------------
md(r"""
---
# 7. Renderização

Esta célula apaga a pasta `PASTA` e gera, em ordem: a abertura, a cartela "Como ler o vídeo", um título e um trecho para cada configuração, e a cartela-resumo. No fim, junta tudo e mostra o vídeo. No Colab, o arquivo final fica no painel **Arquivos**.
""")

code(r"""
if os.path.isdir(PASTA): shutil.rmtree(PASTA)
os.makedirs(PASTA)
arquivos, t0 = [], time.time()
def caminho(nome): return os.path.join(PASTA, nome)

cartela(["Pepé Le Gambá e o Filtro de Kalman", "Parte 2: Pepé descobre as cadeias de Markov (IMM)",
         f"mundo de {MODOS_DO_MUNDO} modos  •  mesma perseguição inicial em todos os trechos (semente {SEED})  •  velocidade {VELOCIDADE_VIDEO:g}×"],
        caminho("00_abertura.mp4"), dur=4.5, tamanhos=[34, 22, 14])
cartela_legenda(caminho("01_legenda.mp4"))
arquivos += [caminho("00_abertura.mp4"), caminho("01_legenda.mp4")]

for i, (nome, kw) in enumerate(configs.items(), 1):
    c_, v_ = caminho(f"{i + 1:02d}a_titulo.mp4"), caminho(f"{i + 1:02d}b_trecho.mp4")
    cartela([nome] + descricoes.get(nome, [f"pepe = {kw['pepe']}, alvo = {kw['alvo']}", ""]), c_)
    grava_trecho(nome, kw, v_)
    arquivos += [c_, v_]
    print(f"[{time.time() - t0:5.0f} s]  pronto: {nome}")

if resumo:
    cartela_resumo(caminho("99_resumo.mp4")); arquivos.append(caminho("99_resumo.mp4"))

with open(caminho("lista.txt"), "w") as f:
    for a in arquivos: f.write(f"file '{os.path.basename(a)}'\n")
tmp = ARQUIVO_FINAL.replace(".mp4", ".part.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", caminho("lista.txt"), "-c", "copy", tmp], check=True)
os.replace(tmp, ARQUIVO_FINAL)
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", ARQUIVO_FINAL],
                           capture_output=True, text=True).stdout)
print(f"\nvídeo final: {ARQUIVO_FINAL}  ({dur:.0f} s, {os.path.getsize(ARQUIVO_FINAL) / 1e6:.1f} MB, renderizado em {time.time() - t0:.0f} s)")
""")

code(r"""
display(Video(ARQUIVO_FINAL, embed=True, width=960))
""")

md(r"""
---
# 8. Sugestões de experimentos

1. **Pepé impaciente.** Mude `DURACOES_IMAGINADAS` para `[1.0, 1.0, 1.0]`. As probabilidades de modo ficam nervosas? E o acerto?
2. **Pepé lento para mudar de ideia.** Use `[30.0, 20.0, 20.0]`. Quanto tempo ele leva para perceber cada fuga? (É o Pepé teimoso da Parte 1, agora em versão IMM.)
3. **Sem procura.** `MODOS_DO_MUNDO = 2`. O IMM de 2 modos passa a ser o melhor?
4. **Calmaria.** `SIG_W, SIG_I = 0.005, 0.01`. Com sensores muito bons, a diferença entre os Pepés diminui? Por quê?
5. **Previsão mais longa.** `TAU_MAX = 8.0`. Mirar onde ela estará passa a ajudar ou a atrapalhar?
6. **Outra perseguição.** Troque `SEED` e veja se as histórias dos trechos mudam de caráter.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "pepe_imm_gerador_video.ipynb")
print(len(cells), "células")
