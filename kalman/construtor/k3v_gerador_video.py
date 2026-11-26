import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Pepé Le Gambá e o Filtro de Kalman: gerador do vídeo

Este notebook gera o vídeo que compara as configurações de Pepé (Fases 1, 2 e 3; EKF e Kalman iterado; Pepé teimoso). **Ele é independente**: traz todo o código da simulação e do filtro, e não precisa do notebook da Parte 1.

**Como usar**

1. Rode a célula de **verificação** (seção 0) para conferir que o `ffmpeg` está disponível.
2. Mude o que quiser nas seções 1 e 2:
   - **seção 1**: parâmetros do mundo, de Pepé e do vídeo;
   - **seção 2**: a lista de configurações que viram trechos do vídeo.
3. Rode a **prévia** (seção 5), que desenha poucos quadros em segundos, para conferir se ficou como você quer.
4. Rode a **renderização** (seção 7). Com os valores padrão, cada configuração leva cerca de 1 minuto no Colab. O vídeo aparece no fim do notebook e fica salvo na pasta indicada.

> **Dica.** Para experimentar, use `MODO_RAPIDO = True`: o vídeo sai em resolução menor e desenha um passo a cada dois, levando cerca de um terço do tempo.
""")

md(r"""
---
# 0. Verificação do ambiente

O vídeo é montado pelo programa `ffmpeg`. No Google Colab ele já vem instalado. Num computador pessoal, instale-o pelo gerenciador de pacotes (`sudo apt install ffmpeg` no Ubuntu, `brew install ffmpeg` no macOS, `conda install -c conda-forge ffmpeg` no Anaconda).
""")

code(r"""
import os, shutil, subprocess, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from IPython.display import Video, display

if shutil.which("ffmpeg") is None:
    print("ffmpeg NÃO encontrado. Instale-o antes de renderizar (veja o texto acima).")
else:
    versao = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    print("ffmpeg encontrado:", versao)
""")

# ---------------------------------------------------------------------
md(r"""
---
# 1. Parâmetros que você pode mudar

Todos os números que definem o mundo, Pepé e o vídeo estão nesta célula. Sugestões de experimentos:

- **vento mais forte**: aumente `SIG_W` e `SIG_I`;
- **Penélope mais rápida ou mais arisca**: mude `VEL_FUGA` e `GATILHO`;
- **sensores**: mude o alcance do som, `R_SOM`, ou a largura do cheiro, `SIG_C`;
- **Pepé mais ou menos humilde**: mude `Q_PADRAO`;
- **outra perseguição**: troque `SEED`.
""")

code(r"""
# ---------------- o mundo ----------------
SEED      = 7          # semente: define a posição inicial de Penélope e todos os sorteios
DT        = 0.2        # passo de tempo (s)
T_TOTAL   = 60.0       # duração da perseguição (s)
V_PEPE    = 1.0        # velocidade de Pepé
DIST_INICIAL = 15.0    # distância inicial entre os dois
BORDA     = 25.0       # Penélope fica dentro do quadrado [-BORDA, BORDA]²

# cheiro: c(x) = A_C exp(-|x - p|² / (2 SIG_C²))
A_C, SIG_C = 1.0, 6.0
# som do coração: s(x) = B_S / (|x - p|² + EPS_S²), audível a menos de R_SOM
B_S, EPS_S, R_SOM = 1.0, 1.0, 6.0

# ruídos (desvios-padrão) das medidas
SIG_W  = 0.01          # vento no gradiente do cheiro
SIG_I  = 0.02          # vento na intensidade do cheiro
SIG_SG = 0.002         # ruído no gradiente do som
SIG_SI = 0.003         # ruído na intensidade do som

# comportamento de fuga de Penélope (intervalos de sorteio uniforme)
GATILHO  = (2.0, 4.5)  # distância que dispara a fuga
VEL_FUGA = (1.8, 3.0)  # velocidade da fuga
DUR_FUGA = (1.5, 4.0)  # duração da fuga (s)
DESVIO_FUGA_GRAUS = 35 # desvio-padrão do ângulo em torno de "para longe de Pepé"

# ---------------- Pepé (o filtro) ----------------
Q_PADRAO = 1.0                                  # "humildade": intensidade das acelerações aleatórias no modelo
P0_PADRAO = np.diag([64.0, 64.0, 1.0, 1.0])      # incerteza inicial (desvio 8 na posição, 1 na velocidade)
M0_PADRAO = np.zeros(4)                         # palpite inicial: Penélope em cima de Pepé, parada

# ---------------- o vídeo ----------------
VELOCIDADE_VIDEO = 3.0     # 3 = três vezes mais rápido que o tempo real
MODO_RAPIDO      = False   # True: resolução menor e um passo a cada dois (para experimentar)
DURACAO_CARTELA  = 3.0     # segundos de cada cartela de título
N_MC             = 40      # perseguições para a tabela-resumo final (0 = sem tabela)
PASTA            = "video_pepe"
ARQUIVO_FINAL    = "pepe_kalman_configuracoes.mp4"
""")

# ---------------------------------------------------------------------
md(r"""
---
# 2. As configurações (um trecho do vídeo para cada uma)

Cada entrada do dicionário `configs` vira um trecho. A **chave** é o título que aparece no vídeo, e o **valor** diz como simular:

| chave do `dict` | valores | significado |
|---|---|---|
| `estrategia` | `"kalman"` (padrão) ou `"ingenuo"` | Pepé anda para a posição estimada, ou segue o gradiente medido |
| `fase` | 1, 2 ou 3 | sensores: só gradiente; + intensidade; + ouvidos |
| `n_iter` | 1 = EKF; mais que 1 = Kalman iterado | iterações de Gauss–Newton por medida |
| `q` | número (padrão `Q_PADRAO`) | humildade de Pepé; $10^{-6}$ é o Pepé teimoso |
| `P0`, `m0` | matriz 4×4, vetor de 4 | crença inicial de Pepé |

O dicionário `descricoes` dá duas linhas de texto para a cartela de cada trecho. Se uma configuração não tiver descrição, a cartela mostra os parâmetros.

Para acrescentar um trecho, basta uma linha nova, por exemplo:

```python
"F3 Kalman iterado, Pepé ansioso": dict(fase=3, n_iter=10, q=50.0),
```
""")

code(r"""
configs = {
    "F1 ingênuo (filtro EKF de carona)": dict(estrategia="ingenuo", fase=1, n_iter=1),
    "F1 EKF":                            dict(fase=1, n_iter=1),
    "F1 Kalman iterado":                 dict(fase=1, n_iter=10),
    "F2 EKF":                            dict(fase=2, n_iter=1),
    "F2 Kalman iterado":                 dict(fase=2, n_iter=10),
    "F3 Kalman iterado":                 dict(fase=3, n_iter=10),
    "F3 Kalman iterado, Pepé TEIMOSO":   dict(fase=3, n_iter=10, q=1e-6),
}

descricoes = {
    "F1 ingênuo (filtro EKF de carona)": ["Fase 1: o nariz mede só o gradiente do cheiro.",
                                          "Pepé ignora o filtro e anda na direção do gradiente medido."],
    "F1 EKF":                            ["Fase 1: só o gradiente.", "Pepé anda para onde o EKF estima que Penélope está."],
    "F1 Kalman iterado":                 ["Fase 1: só o gradiente.", "Kalman iterado (Gauss–Newton) a cada medida."],
    "F2 EKF":                            ["Fase 2: gradiente + intensidade do cheiro.",
                                          "EKF: linearizado no ponto errado, fica confiante demais e se perde."],
    "F2 Kalman iterado":                 ["Fase 2: gradiente + intensidade do cheiro.", "Kalman iterado: a intensidade dá a distância."],
    "F3 Kalman iterado":                 ["Fase 3: nariz + ouvidos (o coração de Penélope).", "Integração de sensores com o Kalman iterado."],
    "F3 Kalman iterado, Pepé TEIMOSO":   ["Fase 3, mas com q = 10⁻⁶: Pepé acredita que Penélope não foge.",
                                          "O Pepé do desenho: bons sensores, crença que não se atualiza."],
}
print(len(configs), "configurações")
""")

# ---------------------------------------------------------------------
md(r"""
---
# 3. O mundo e o filtro

O código abaixo é o mesmo da Parte 1, agora lendo os parâmetros da seção 1. **Não é preciso mudar nada aqui**; as explicações linha a linha estão na Parte 1.

Se você mudar parâmetros na seção 1, **rode de novo esta célula** antes de simular (ela recalcula `N_PASSOS`, `F` etc.).
""")

code(r"""
N_PASSOS = int(round(T_TOTAL / DT))

# ----- campos físicos -----
def cheiro(x, p):
    d = x - p
    return A_C * np.exp(-(d @ d) / (2 * SIG_C**2))

def grad_cheiro(x, p):
    return -cheiro(x, p) / SIG_C**2 * (x - p)

def som(x, p):
    d = x - p
    return B_S / (d @ d + EPS_S**2)

def grad_som(x, p):
    d = x - p
    return -2 * B_S * d / (d @ d + EPS_S**2)**2

# ----- Penélope -----
def passo_penelope(pen, xP, rng):
    if not pen["fugindo"] and np.linalg.norm(xP - pen["p"]) < pen["gatilho"]:
        longe = np.arctan2(*(pen["p"] - xP)[::-1])
        ang = longe + rng.normal(0, np.deg2rad(DESVIO_FUGA_GRAUS))
        vel = rng.uniform(*VEL_FUGA)
        pen.update(fugindo=True, t_restante=rng.uniform(*DUR_FUGA), v=vel * np.array([np.cos(ang), np.sin(ang)]))
    if pen["fugindo"]:
        pen["t_restante"] -= DT
        if pen["t_restante"] <= 0:
            pen.update(fugindo=False, v=np.zeros(2), gatilho=rng.uniform(*GATILHO))
    pen["p"] = pen["p"] + pen["v"] * DT
    for k in range(2):
        if abs(pen["p"][k]) > BORDA:
            pen["p"][k] = np.sign(pen["p"][k]) * BORDA; pen["v"][k] *= -1
    return pen

# ----- sensores de Pepé -----
def mede(xP, p, fase, rng):
    z, var = [grad_cheiro(xP, p) + SIG_W * rng.standard_normal(2)], [SIG_W**2, SIG_W**2]
    if fase >= 2:
        z.append([cheiro(xP, p) + SIG_I * rng.standard_normal()]); var += [SIG_I**2]
    ouviu = fase >= 3 and np.linalg.norm(xP - p) < R_SOM
    if ouviu:
        z.append(grad_som(xP, p) + SIG_SG * rng.standard_normal(2)); var += [SIG_SG**2, SIG_SG**2]
        z.append([som(xP, p) + SIG_SI * rng.standard_normal()]);    var += [SIG_SI**2]
    return np.concatenate(z), np.diag(var), ouviu

# ----- modelo de Pepé -----
F = np.block([[np.eye(2), DT * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])

def Q_mat(q):
    return q * np.block([[DT**3 / 3 * np.eye(2), DT**2 / 2 * np.eye(2)],
                         [DT**2 / 2 * np.eye(2), DT * np.eye(2)]])

def h_modelo(s, xP, fase, ouviu):
    p = s[:2]; partes = [grad_cheiro(xP, p)]
    if fase >= 2: partes.append([cheiro(xP, p)])
    if ouviu:     partes += [grad_som(xP, p), [som(xP, p)]]
    return np.concatenate(partes)

def J_modelo(s, xP, fase, ouviu):
    p = s[:2]; d = xP - p; c = cheiro(xP, p)
    linhas = [c / SIG_C**2 * (np.eye(2) - np.outer(d, d) / SIG_C**2)]
    if fase >= 2: linhas.append((c * d / SIG_C**2)[None, :])
    if ouviu:
        D = d @ d + EPS_S**2
        linhas.append(2 * B_S / D**2 * (np.eye(2) - 4 * np.outer(d, d) / D))
        linhas.append((2 * B_S * d / D**2)[None, :])
    Jp = np.vstack(linhas)
    return np.hstack([Jp, np.zeros_like(Jp)])

# ----- EKF (n_iter = 1) ou Kalman iterado (n_iter > 1) -----
def atualizacao(m, P, z, R, h, J, n_iter=1, tol=1e-9):
    s = m.copy()
    for _ in range(n_iter):
        H = J(s); S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T
        s_novo = m + K @ (z - h(s) - H @ (m - s))
        if np.linalg.norm(s_novo - s) < tol: s = s_novo; break
        s = s_novo
    H = J(s); S = H @ P @ H.T + R; K = np.linalg.solve(S, H @ P).T
    P_novo = P - K @ S @ K.T
    return s, (P_novo + P_novo.T) / 2

# ----- a perseguição -----
def simula(estrategia="kalman", fase=1, seed=0, n_iter=1, q=None, P0=None, m0=None):
    q = Q_PADRAO if q is None else q
    P0 = P0_PADRAO if P0 is None else P0
    m0 = M0_PADRAO if m0 is None else m0
    rng = np.random.default_rng(seed)
    ang0 = rng.uniform(0, 2 * np.pi)
    pen = dict(p=DIST_INICIAL * np.array([np.cos(ang0), np.sin(ang0)]), v=np.zeros(2),
               fugindo=False, t_restante=0.0, gatilho=rng.uniform(*GATILHO))
    xP, m, P, Q = np.zeros(2), np.array(m0, float), np.array(P0, float), Q_mat(q)
    reg = []
    for k in range(N_PASSOS):
        pen = passo_penelope(pen, xP, rng)
        z, R, ouviu = mede(xP, pen["p"], fase, rng)
        if k > 0: m, P = F @ m, F @ P @ F.T + Q
        m, P = atualizacao(m, P, z, R, lambda s: h_modelo(s, xP, fase, ouviu),
                           lambda s: J_modelo(s, xP, fase, ouviu), n_iter)
        reg.append(dict(t=k * DT, xP=xP.copy(), p=pen["p"].copy(), fugindo=pen["fugindo"], ouviu=ouviu,
                        z=z, m=m.copy(), P=P.copy()))
        direcao = z[:2] if estrategia == "ingenuo" else m[:2] - xP
        if np.linalg.norm(direcao) > 1e-12:
            xP = xP + V_PEPE * DT * direcao / np.linalg.norm(direcao)
    return reg

def series(reg):
    t = np.array([r["t"] for r in reg])
    erro = np.array([np.linalg.norm(r["m"][:2] - r["p"]) for r in reg])
    sd = np.array([np.sqrt(np.trace(r["P"][:2, :2])) for r in reg])
    dist = np.array([np.linalg.norm(r["xP"] - r["p"]) for r in reg])
    fug = np.array([r["fugindo"] for r in reg]); ouv = np.array([r["ouviu"] for r in reg])
    nees = np.array([(r["m"][:2] - r["p"]) @ np.linalg.solve(r["P"][:2, :2], r["m"][:2] - r["p"]) for r in reg])
    return dict(t=t, erro=erro, sd=sd, dist=dist, fugindo=fug, ouviu=ouv, nees=nees)

print(f"{N_PASSOS} passos por perseguição")
""")

# ---------------------------------------------------------------------
md(r"""
---
# 4. O desenho de cada quadro e a gravação

Cada quadro é desenhado com o Matplotlib e enviado **diretamente** ao `ffmpeg` como uma imagem crua (*rawvideo*), sem salvar arquivos PNG intermediários. Isso é bem mais rápido.

Alguns detalhes:

- **A janela da arena é fixa** em cada trecho, calculada a partir dos caminhos completos de Pepé e de Penélope. Assim o fundo não "pula" de um quadro para outro.
- **Estimativa fora da tela.** Quando a estimativa de Pepé sai da janela (como no EKF da Fase 2), aparece uma seta vermelha apontando para onde ela foi.
- **Os gráficos da direita** só mostram o que já aconteceu até aquele instante.
- **Cada trecho vira um arquivo `.mp4` separado**, e no fim os arquivos são **concatenados sem recodificar** (`-c copy`). Para isso funcionar, todos têm de ter o mesmo tamanho, taxa de quadros e codificação.
- **Arquivo temporário.** Cada arquivo é gravado primeiro com um nome temporário e só é renomeado se o `ffmpeg` terminar sem erro. Assim, uma renderização interrompida não deixa um vídeo incompleto que pareça bom.
""")

code(r"""
FPS, DPI = 30, 100
W, H = (854, 480) if MODO_RAPIDO else (1280, 720)
PULO = 2 if MODO_RAPIDO else 1                                   # desenha um passo a cada PULO
PASSOS_POR_SEG = VELOCIDADE_VIDEO / DT / PULO                    # quadros de simulação por segundo de vídeo
ESCALA = W / 1280                                                # para as fontes acompanharem a resolução
ENC = ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", str(FPS)]

def elipse(ax, centro, Ppos, k=2.0, **kw):
    w, V = np.linalg.eigh(Ppos)
    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
    ax.add_patch(Ellipse(centro, 2 * k * np.sqrt(w[-1]), 2 * k * np.sqrt(w[0]), angle=ang, fill=False, **kw))

def janela(reg):
    pts = np.array([r["xP"] for r in reg] + [r["p"] for r in reg])
    lo, hi = pts.min(0) - 6, pts.max(0) + 6
    c, L = (lo + hi) / 2, max(max(hi - lo) / 2, 14)
    return c - L, c + L

def nova_figura():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")
    axA = fig.add_axes([0.03, 0.06, 0.50, 0.84])
    axE = fig.add_axes([0.60, 0.55, 0.37, 0.33])
    axD = fig.add_axes([0.60, 0.10, 0.37, 0.33])
    return fig, axA, axE, axD

def desenha_quadro(fig, axA, axE, axD, nome, kw, reg, S_, k, jan, grade):
    (x0, y0), (x1, y1) = jan; gx, gy = grade; r = reg[k]; fs = ESCALA
    for a in (axA, axE, axD): a.cla()
    fig.texts.clear()
    fig.text(0.03, 0.955, nome, fontsize=17 * fs, fontweight="bold", color="#3b2a63")
    estado = "FUGINDO" if r["fugindo"] else "escondida"
    fig.text(0.60, 0.955, f"t = {r['t']:4.1f} s     Penélope: {estado}" + ("     Pepé OUVE" if r["ouviu"] else ""),
             fontsize=13 * fs, color="#b8860b" if r["fugindo"] else "#333333")
    # --- arena ---
    C = A_C * np.exp(-((gx - r["p"][0])**2 + (gy - r["p"][1])**2) / (2 * SIG_C**2))
    axA.imshow(C, extent=[x0, x1, y0, y1], origin="lower", cmap="Purples", alpha=.45, vmin=0, vmax=1.2)
    cam = np.array([x["p"] for x in reg[:k + 1]]); camP = np.array([x["xP"] for x in reg[:k + 1]])
    est = np.array([x["m"][:2] for x in reg[max(0, k - 60):k + 1]])
    axA.plot(*cam.T, color="gray", lw=1, ls="--")
    axA.plot(*camP.T, color="C4", lw=1.6)
    axA.plot(*est.T, color="C3", lw=1, alpha=.7)
    if kw["fase"] >= 3:
        axA.add_patch(Circle(r["p"], R_SOM, fill=False, ls=":", color="C1", lw=1.5))
    axA.plot(*r["p"], "*", color="gold" if r["fugindo"] else "k", mec="k", ms=20 * fs, label="Penélope (amarela quando foge)")
    axA.plot(*r["xP"], "^", color="C4", ms=14 * fs, label="Pepé")
    m = r["m"][:2]
    if x0 < m[0] < x1 and y0 < m[1] < y1:
        axA.plot(*m, "o", color="C3", ms=8 * fs, label="estimativa de Pepé (elipse 2σ)")
        elipse(axA, m, r["P"][:2, :2], color="C3", lw=2)
    else:
        d = (m - r["xP"]) / np.linalg.norm(m - r["xP"])
        axA.annotate("", xy=r["xP"] + 6 * d, xytext=r["xP"], arrowprops=dict(arrowstyle="->", color="C3", lw=2))
        axA.text(*(r["xP"] + 7.5 * d), "estimativa de Pepé:\nfora da tela", color="C3", fontsize=10 * fs, ha="center", va="center")
    g = r["z"][:2]
    if np.linalg.norm(g) > 0:
        axA.annotate("", xy=r["xP"] + 3 * g / np.linalg.norm(g), xytext=r["xP"],
                     arrowprops=dict(arrowstyle="->", color="C2", lw=2.5))
    axA.set_xlim(x0, x1); axA.set_ylim(y0, y1); axA.set_aspect("equal")
    axA.tick_params(labelsize=10 * fs)
    axA.legend(loc="lower left", fontsize=9 * fs, framealpha=.85)
    axA.text(0.99, 0.01, "seta verde: gradiente medido", transform=axA.transAxes, ha="right", fontsize=9 * fs, color="C2")
    # --- séries ---
    tt = S_["t"][:k + 1]
    for ax_ in (axE, axD):
        ax_.fill_between(tt, 0, 1, where=S_["fugindo"][:k + 1], color="gold", alpha=.35,
                         transform=ax_.get_xaxis_transform(), step="mid")
        ax_.set_xlim(0, T_TOTAL); ax_.grid(alpha=.3); ax_.tick_params(labelsize=10 * fs)
    axE.plot(tt, np.clip(S_["erro"][:k + 1], 1e-2, 1e3), "C3")
    axE.plot(tt, 2 * S_["sd"][:k + 1], "k--", lw=1)
    axE.set_yscale("log"); axE.set_ylim(1e-2, 1e3)
    axE.set_title("erro da estimativa (vermelho) e incerteza declarada 2σ (tracejado)", fontsize=10.5 * fs)
    axD.plot(tt, S_["dist"][:k + 1], "C4")
    axD.set_ylim(0, max(25, S_["dist"].max() * 1.05)); axD.set_xlabel("t (s)", fontsize=10 * fs)
    axD.set_title("distância Pepé–Penélope   (amarelo: fuga)", fontsize=11 * fs)

def prepara(kw):
    reg = simula(seed=SEED, **kw); S_ = series(reg); jan = janela(reg)
    (x0, y0), (x1, y1) = jan
    grade = np.meshgrid(np.linspace(x0, x1, 140), np.linspace(y0, y1, 140))
    return reg, S_, jan, grade

def grava_trecho(nome, kw, saida):
    reg, S_, jan, grade = prepara(kw)
    fig, axA, axE, axD = nova_figura()
    tmp = saida.replace(".mp4", ".part.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
           "-r", f"{PASSOS_POR_SEG:.6f}", "-i", "-"] + ENC + [tmp]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in range(0, len(reg), PULO):
        desenha_quadro(fig, axA, axE, axD, nome, kw, reg, S_, k, jan, grade)
        fig.canvas.draw()
        proc.stdin.write(bytes(fig.canvas.buffer_rgba()))
    plt.close(fig)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"o ffmpeg falhou no trecho {nome!r}")
    os.replace(tmp, saida)

def png_para_mp4(png, saida, dur):
    tmp = saida.replace(".mp4", ".part.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", png, "-vf",
                    f"format=yuv420p,loop=-1:1,trim=duration={dur},fade=in:0:10,fade=out:st={dur - 0.4}:d=0.4"]
                   + ENC + [tmp], check=True)
    os.replace(tmp, saida)

def cartela(linhas, saida, dur=DURACAO_CARTELA, tamanhos=None):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#1d1530")
    tamanhos = tamanhos or [34] + [20] * (len(linhas) - 1)
    y = 0.62
    for txt, tam in zip(linhas, tamanhos):
        fig.text(0.5, y, txt, ha="center", va="center", color="white", fontsize=tam * ESCALA,
                 fontweight="bold" if tam > 25 else "normal")
        y -= 0.1 if tam > 25 else 0.07
    png = saida.replace(".mp4", ".png"); fig.savefig(png, dpi=DPI, facecolor=fig.get_facecolor()); plt.close(fig)
    png_para_mp4(png, saida, dur)

def descricao_de(nome, kw):
    if nome in descricoes: return descricoes[nome]
    return ["parâmetros: " + ", ".join(f"{k} = {v}" for k, v in kw.items() if k not in ("P0", "m0")), ""]
""")

# ---------------------------------------------------------------------
md(r"""
---
# 5. Prévia: alguns quadros, em segundos

Antes de renderizar o vídeo inteiro, desenhe alguns quadros de uma configuração para conferir se o resultado está como você quer. Mude `CONFIG_PREVIA` e `INSTANTES_PREVIA` (em segundos) à vontade.
""")

code(r"""
CONFIG_PREVIA = "F3 Kalman iterado"
INSTANTES_PREVIA = [5, 12, 40]

kw = configs[CONFIG_PREVIA]
reg, S_, jan, grade = prepara(kw)
for t_ in INSTANTES_PREVIA:
    k = min(int(round(t_ / DT)), len(reg) - 1)
    fig, axA, axE, axD = nova_figura()
    desenha_quadro(fig, axA, axE, axD, CONFIG_PREVIA, kw, reg, S_, k, jan, grade)
    plt.show()
""")

# ---------------------------------------------------------------------
md(r"""
---
# 6. A tabela-resumo

Para a cartela final, cada configuração é simulada `N_MC` vezes com sementes diferentes (de `SEED + 100` em diante). Para cada uma, medimos:

- **erro mediano** da estimativa de posição;
- **tempo perdido**: porcentagem do tempo com erro maior que 10;
- **dentro da elipse 2σ**: porcentagem do tempo em que Penélope está dentro da elipse de 2σ de Pepé (o ideal é cerca de 86%).

Com `N_MC = 40`, esta célula leva menos de um minuto. Use `N_MC = 0` para pular a tabela.
""")

code(r"""
resumo = []
if N_MC > 0:
    t0 = time.time()
    for nome, kw in configs.items():
        E, PERD, CONS = [], [], []
        for sd in range(N_MC):
            S_mc = series(simula(seed=SEED + 100 + sd, **kw))
            E.append(np.median(S_mc["erro"])); PERD.append(np.mean(S_mc["erro"] > 10)); CONS.append(np.mean(S_mc["nees"] <= 4.0))
        resumo.append((nome, np.median(E), 100 * np.mean(PERD), 100 * np.mean(CONS)))
        print(f"{nome:40s} erro mediano {resumo[-1][1]:8.2f} | perdido {resumo[-1][2]:5.1f}% | na elipse {resumo[-1][3]:5.1f}%")
    print(f"({time.time() - t0:.0f} s)")

def fmt_num(x):
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}").replace(".", ",")

def fmt_pct(x):
    return (f"{x:.0f}%" if x >= 1 or x == 0 else f"{x:.1f}%").replace(".", ",")

def cartela_resumo(saida, dur=9.0):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="#1d1530")
    fs = ESCALA
    fig.text(0.5, 0.88, f"Resumo em {N_MC} perseguições", ha="center", color="white", fontsize=32 * fs, fontweight="bold")
    xs = [0.06, 0.56, 0.71, 0.86]
    for x, c in zip(xs, ["configuração", "erro mediano", "tempo perdido", "dentro da elipse 2σ"]):
        fig.text(x, 0.77, c, color="#c9b8ff", fontsize=16 * fs, ha="left" if x == xs[0] else "center")
    melhor = min(r[1] for r in resumo)
    passo_y = min(0.075, 0.62 / max(len(resumo), 1))
    y = 0.69
    for nome, e, perd, cons in resumo:
        cor = "#ff8a80" if perd > 50 else ("#a5d6a7" if e == melhor else "white")
        for x, c in zip(xs, [nome, fmt_num(e), fmt_pct(perd), fmt_pct(cons)]):
            fig.text(x, y, c, color=cor, fontsize=17 * fs, ha="left" if x == xs[0] else "center")
        y -= passo_y
    fig.text(0.5, 0.06, "verde: menor erro  •  vermelho: perdido mais da metade do tempo  •  elipse 2σ: ideal ≈ 86%",
             ha="center", color="#bbbbbb", fontsize=13 * fs)
    png = saida.replace(".mp4", ".png"); fig.savefig(png, dpi=DPI, facecolor=fig.get_facecolor()); plt.close(fig)
    png_para_mp4(png, saida, dur)
""")

# ---------------------------------------------------------------------
md(r"""
---
# 7. Renderização

Esta célula apaga a pasta `PASTA` (para não misturar trechos de uma renderização anterior com parâmetros diferentes) e gera, em ordem:

1. a cartela de abertura;
2. para cada configuração, uma cartela de título e o trecho da perseguição;
3. a cartela-resumo (se `N_MC > 0`).

No fim, junta tudo num só arquivo e o mostra aqui. No Colab, o arquivo fica no painel **Arquivos**, à esquerda, de onde pode ser baixado.
""")

code(r"""
if os.path.isdir(PASTA): shutil.rmtree(PASTA)
os.makedirs(PASTA)
arquivos, t0 = [], time.time()

ab = os.path.join(PASTA, "00_abertura.mp4")
cartela(["Pepé Le Gambá e o Filtro de Kalman", f"{len(configs)} maneiras de farejar Penélope",
         f"mesma perseguição inicial em todos os trechos (semente {SEED})  •  velocidade {VELOCIDADE_VIDEO:g}×"],
        ab, dur=4.0, tamanhos=[36, 24, 16])
arquivos.append(ab)

for i, (nome, kw) in enumerate(configs.items(), 1):
    c_ = os.path.join(PASTA, f"{i:02d}a_titulo.mp4"); v_ = os.path.join(PASTA, f"{i:02d}b_trecho.mp4")
    cartela([nome] + descricao_de(nome, kw), c_)
    grava_trecho(nome, kw, v_)
    arquivos += [c_, v_]
    print(f"[{time.time() - t0:5.0f} s]  pronto: {nome}")

if resumo:
    rs = os.path.join(PASTA, "99_resumo.mp4"); cartela_resumo(rs); arquivos.append(rs)

lista = os.path.join(PASTA, "lista.txt")
with open(lista, "w") as f:
    for a in arquivos: f.write(f"file '{os.path.basename(a)}'\n")
tmp = ARQUIVO_FINAL.replace(".mp4", ".part.mp4")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lista, "-c", "copy", tmp], check=True)
os.replace(tmp, ARQUIVO_FINAL)
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", ARQUIVO_FINAL],
                           capture_output=True, text=True).stdout)
print(f"\nvídeo final: {ARQUIVO_FINAL}  ({dur:.0f} s de vídeo, {os.path.getsize(ARQUIVO_FINAL) / 1e6:.1f} MB, "
      f"renderizado em {time.time() - t0:.0f} s)")
""")

code(r"""
display(Video(ARQUIVO_FINAL, embed=True, width=960))
""")

md(r"""
---
# 8. Sugestões de experimentos para os alunos

1. **Vento forte.** Multiplique `SIG_W` e `SIG_I` por 5. Qual configuração sofre mais? A Fase 3 compensa?
2. **Penélope atleta.** Use `VEL_FUGA = (3.0, 5.0)` e `DUR_FUGA = (3.0, 6.0)`. O Kalman iterado ainda acompanha? Como deveria mudar `Q_PADRAO`?
3. **Pepé ansioso.** Acrescente a configuração `"F3 ansioso": dict(fase=3, n_iter=10, q=100.0)`. Compare com o teimoso: qual dos dois exageros é pior?
4. **Ouvido apurado.** Aumente `R_SOM` para 12. A integração de sensores passa a dominar?
5. **Palpite inicial confiante e errado.** Acrescente `dict(fase=3, n_iter=10, m0=np.array([10.0, 10.0, 0, 0]), P0=np.diag([0.01, 0.01, 0.01, 0.01]))`. É o Pepé do desenho logo no começo: quanto tempo ele leva para se recuperar, se é que se recupera?
6. **Outra história.** Troque `SEED` e compare: as conclusões da tabela-resumo mudam? E as do vídeo?
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "pepe_kalman_gerador_video.ipynb")
print(len(cells), "células")
