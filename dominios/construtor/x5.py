from comum import *

c = Caderno(arquivo("5"))
cabecalho(c, "Domínio V: os números dual-complexos",
          "A pose no plano num número só: posição e orientação, compostas por multiplicação",
          dom="5", anterior=arquivo("4"))

c.md(r"""
**O domínio.** Os dual-complexos $q=a+b\varepsilon$, com $a,b\in\mathbb C$, $\varepsilon^2=0$ e a regra $z\varepsilon=\varepsilon\bar z$, juntam o $i$ do domínio IV (que gira) e o $\varepsilon$ do domínio III (que carrega uma perturbação). A regra de troca torna o produto **não comutativo**, e é exatamente isso que os faz servir para **poses**: girar e depois transladar não é o mesmo que transladar e depois girar. A pose "girar $\theta$, transladar $t$" é $q=r+\tfrac12 t\bar r\,\varepsilon$, com $r=e^{i\theta/2}$; compor poses é multiplicar; inverter é conjugar. O grupo das poses do plano, $SE(2)$, aparece como os dual-complexos unitários, com $q$ e $-q$ representando a mesma pose.

Três consequências que atravessam o domínio:
- a **ordem importa**: o erro de um filtro pode ser definido no corpo, $q=\hat q\,\mathrm{Exp}(\xi)$, ou no mundo, $q=\mathrm{Exp}(\xi)\,\hat q$, e as duas escolhas dão filtros diferentes;
- a $\mathrm{Exp}$ de uma velocidade constante é um **arco de círculo**, a trajetória exata de um corpo com velocidades constantes no próprio referencial; a reta de Euler é só uma aproximação;
- integradores de ordem alta precisam do **colchete de Lie** $[X_1,X_2]$, que mede o quanto dois movimentos não comutam: sem ele, Magnus de ordem 4 cai para ordem 2.

**Os exemplos físicos deste domínio**
- **Um braço articulado** e o **alinhamento de dois *scans*** de um LiDAR: cinemática inversa e ICP, na etapa (b).
- **Uma nave de Asteroids** e a **banana** da odometria, na etapa (c).
- **Pepé inteiro**: a pose de Pepé e a posição de Penélope estimadas juntas, com um radar polar, e três maneiras de definir o erro, na etapa (d).
- **Sete apêndices** com aplicações que não cabiam nas quatro etapas: SLAM, quatro barras, carro, curling, futebol de robôs, estilingue gravitacional e treliça.

A animação mostra o mecanismo de quatro barras (à esquerda) e um carro que faz um oito com o modelo de bicicleta (à direita); nos dois, a pose de uma peça é um dual-complexo, e o centro instantâneo de rotação (×) é o ponto fixo da pose incremental.
""")

c.code(FERRAMENTAS + r'''
from matplotlib import animation
from IPython.display import HTML

def pose_fixo(r1, t1, r2, t2):
    """O centro instantâneo: ponto fixo do deslocamento rígido que leva (r1, t1) a (r2, t2): e^{iΔθ} p + Δt = p."""
    dr = r2 / r1; dt = t2 - dr * t1
    return dt / (1 - dr) if abs(1 - dr) > 1e-9 else None

# quatro barras (apêndice): a = 1, b = 3,5, c = 3, d = 4
def quatro_barras(t2):
    A = np.exp(1j * t2); O4 = 4.0 + 0j; D = abs(O4 - A)
    x = (3.5**2 - 3**2 + D**2) / (2 * D); h = np.sqrt(3.5**2 - x**2); u = (O4 - A) / D
    B = A + x * u + h * 1j * u
    return A, B
angs = np.linspace(0, 2 * np.pi, 61)
p_local = 1.2 + 1.6j
curva = []
for t2 in np.linspace(0, 2 * np.pi, 400):
    A, B = quatro_barras(t2); curva.append(A + (B - A) / abs(B - A) * p_local)
curva = np.array(curva)

# carro (apêndice): modelo de bicicleta, esterçamento alternado, um oito
L, v, dt = 2.7, 5.0, 0.05
th, p, carro = 0.0, 0j, []
T_volta = 2 * np.pi * (L / np.tan(np.radians(15))) / v
for k in range(int(2 * T_volta / dt)):
    delta = np.radians(15) * (1 if k * dt < T_volta else -1)
    w = v * np.tan(delta) / L
    p += v * np.exp(1j * (th + w * dt / 2)) * np.sinc(w * dt / 2 / np.pi) * dt     # o arco exato da Exp
    th += w * dt
    carro.append((p, th, w))
carro = carro[::max(1, len(carro) // 60)][:60]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6), dpi=60)
for ax in (a1, a2):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
a1.set_xlim(-2, 6); a1.set_ylim(-2.5, 4.5); a1.set_title("quatro barras", fontsize=9)
a1.plot(curva.real, curva.imag, color="tab:red", lw=1); a1.plot([0, 4], [0, 0], "k-", lw=4, alpha=0.3)
barras, = a1.plot([], [], "k-o", ms=4, lw=2); tri, = a1.fill([0], [0], color="tab:blue", alpha=0.3)
ci1, = a1.plot([], [], "x", color="tab:green", ms=10, mew=2)
a2.set_xlim(-24, 24); a2.set_ylim(-23, 23); a2.set_title("carro: modelo de bicicleta", fontsize=9)
rastro, = a2.plot([], [], lw=0.8, color="tab:purple"); corpo, = a2.plot([], [], color="tab:purple", lw=4)
ci2, = a2.plot([], [], "x", color="tab:green", ms=10, mew=2)
plt.tight_layout()

def quadro(i):
    A, B = quatro_barras(angs[i]); P = A + (B - A) / abs(B - A) * p_local
    barras.set_data([0, A.real, B.real, 4], [0, A.imag, B.imag, 0]); tri.set_xy(np.array([[A.real, A.imag], [B.real, B.imag], [P.real, P.imag]]))
    A2, B2 = quatro_barras(angs[i] + 1e-4)
    c1 = pose_fixo((B - A) / abs(B - A), A, (B2 - A2) / abs(B2 - A2), A2)
    ci1.set_data([c1.real] if c1 is not None and abs(c1) < 8 else [], [c1.imag] if c1 is not None and abs(c1) < 8 else [])
    p, th, w = carro[i]
    pts = np.array([c_[0] for c_ in carro[:i + 1]]); rastro.set_data(pts.real, pts.imag)
    frente = p + L * np.exp(1j * th); corpo.set_data([p.real, frente.real], [p.imag, frente.imag])
    c2 = p + 1j * np.exp(1j * th) * v / w                  # Ackermann: à distância R = v/ω, sobre o eixo traseiro
    ci2.set_data([c2.real], [c2.imag])
    return barras, tri, ci1, rastro, corpo, ci2

anim = animation.FuncAnimation(fig, quadro, frames=60, interval=150, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())
''')

c.md(r"""
---
## O que este domínio estabelece para a série

| ideia | domínio IV (complexos) | domínio V (dual-complexos) | o que muda depois |
|---|---|---|---|
| **o que se representa** | a rotação, $e^{i\theta}$ | a pose, $r+\tfrac12t\bar r\,\varepsilon$ | a rotação no espaço (VI), a pose no espaço (VII) |
| **compor** | $q_1q_2=q_2q_1$ | $q_1q_2\neq q_2q_1$ | não comutativo também na rotação |
| **$\boxplus$** | $q\,e^{i\delta}$ = somar ao ângulo | $\hat q\,\mathrm{Exp}(\xi)$ (corpo) ou $\mathrm{Exp}(\xi)\,\hat q$ (mundo): **diferentes** | idem, com $\mathrm{Ad}$ $6\times6$ |
| **$\mathrm{Exp}$** | $e^{i\theta}$ | um arco de círculo; o $\mathrm{sinc}$ estável perto de zero | a fórmula de Rodrigues; o parafuso de Chasles |
| **integrar** | RK4 encolhe; RKMK não | Magnus de ordem 4 precisa do colchete $[X_1,X_2]$ | colchetes de $\mathfrak{so}(3)$ e $\mathfrak{se}(3)$ |
| **incerteza** | um ângulo, uma variância | a **banana**: gaussiana nas coordenadas exponenciais | o mesmo, em dimensão maior |
| **filtro** | estado de erro $\delta\psi$ | erro no corpo, no mundo (invariante), nas coordenadas | o ESKF e o IEKF de atitude e pose |

---
## As quatro folhas deste domínio
""")

resumo = {
    "a": ("O teste da não comutatividade e a representação matricial $\\begin{bmatrix}a&b\\\\0&\\bar a\\end{bmatrix}$ pelo `sympy`; a pose "
          "$r+\\tfrac12t\\bar r\\varepsilon$, a composição e a inversa; um braço de três elos como um produto de três números; $\\mathrm{Exp}$ e "
          "$\\mathrm{Log}$ com o $\\mathrm{sinc}$ estável; a dobradiça da porta como ponto fixo; interpolar pelo arco ou pela reta; o *skinning* "
          "que não afina o cotovelo (espessura 0,6 contra 0,155 das matrizes); 200 poses exatas com inteiros gaussianos (69 dígitos) "
          "contra o `float` ($3\\times10^{-14}$); a interface de domínio em $SE(2)$."),
    "b": ("Cinemática inversa de um braço de dois elos por Newton com jacobiano dual (bacias de 49%, 49% e 2% para as duas soluções e a "
          "divergência); o número de condição cresce como $1/\\sqrt{\\text{folga}}$ perto da singularidade; Levenberg–Marquardt para um alvo "
          "fora do alcance; a pose completa com três elos e a redundância; Procrustes em uma linha complexa, igual ao SVD; o ICP, que "
          "acerta sempre com chutes de até 30° e nunca entre 90° e 150°."),
    "c": ("O colchete de Lie conferido pelo comutador de grupo; Euler, ponto médio, Magnus com e sem colchete, RK4 nas coordenadas e "
          "nas componentes (ordens 1, 2, 2, 4, 4 e 4; só o último sai do grupo, $3\\times10^{-7}$); a nave de Asteroids, que conserva a "
          "velocidade no mundo; a banana da odometria, com cobertura melhor nas coordenadas exponenciais (0,93 contra 0,87 a 0,04 rad/√s, "
          "0,49 contra 0,23 a 0,15) e abaixo de 95% em primeira ordem; a matriz $\\mathrm{Ad}$."),
    "d": ("Pepé estima a própria pose e a posição de Penélope com odometria e um radar polar. Com faróis, os três filtros com a "
          "predição pelo arco empatam (4,2 cm, NEES 3,3), e o EKF com o passo de Euler fica otimista (5,2 cm, NEES 4,5). Com GPS e a "
          "direção inicial desconhecida, o erro no mundo é inconsistente no primeiro segundo (NEES 22 contra 3,5): o erro no mundo é "
          "invariante para os faróis, o erro no corpo para o GPS."),
}
linhas = []
for e, slug, titulo in ETAPAS:
    arq = arquivo("5", e)
    linhas.append(f"### ({e}) {titulo[0].upper() + titulo[1:]}\n{link(arq)}\n\n{resumo[e]}\n")
c.md("\n".join(linhas))

apendices = [
    ("slam", "SLAM por grafo de poses",
     "Um robô dá a volta num quadrado de 10 m e reconhece o ponto de partida. Gauss–Newton no grupo com 41 poses: a última pose "
     "passa de 84 cm para 7 mm da verdade, mas o meio do caminho fica com 31 cm, porque um único fechamento de laço não diz por onde o robô passou."),
    ("quatrobarras", "o mecanismo de quatro barras",
     "Posição por interseção de círculos, velocidades pela equação do laço derivada, a curva do acoplador, e o centro instantâneo como "
     "ponto fixo da pose incremental, conferido pelo teorema de Kennedy; a condição de Grashof conferida por varredura."),
    ("carro", "odometria de um carro",
     "O modelo de bicicleta: o ponto fixo da $\\mathrm{Exp}$ é o centro de Ackermann; Euler tem ordem 1, a $\\mathrm{Exp}$ é exata. Calibrar "
     "raios e bitola com dois circuitos fechados falha (um circuito fechado não tem escala); com uma reta de 20 m, os três parâmetros saem certos."),
    ("curling", "a pedra de curling (modelo didático)",
     "Uma pedra que desliza e gira, integrada no mundo e no grupo (o mesmo esquema, em dois referenciais); o giro decide o lado da curva; "
     "Newton acha o arremesso que para no centro da casa. O mecanismo físico real da curva não é consenso, e o notebook diz isso."),
    ("robocup", "futebol de robôs",
     "Quatro adversários vistos por uma câmera que não diz quem é quem: filtros de Kalman, distância de Mahalanobis e o algoritmo húngaro "
     "contra o guloso. Em 40 simulações, o húngaro tem menos da metade das trocas de identidade; a maior parte das trocas vem de rastros que se perdem."),
    ("estilingue", "o estilingue gravitacional",
     "A hipérbole de Kepler com RK4 e o tempo de Sundman; o ganho de velocidade passando por trás do planeta; o tiro de Newton com a "
     "derivada por duais de partes complexas, a álgebra comutativa que não serve para poses mas serve para derivar."),
    ("trelica", "a treliça de uma ponte",
     "Forças pelo equilíbrio dos nós, conferidas pelo corte de Ritter; a derivada dual de um sistema linear com a mesma fatoração LU; a "
     "altura de menor volume (um quarto do vão, num mínimo raso); a linha de influência de uma diagonal que troca de sinal."),
]
linhas = ["---\n## Apêndices\n\nAplicações que não cabiam nas quatro etapas, cada uma num notebook independente. Todas usam as ferramentas do domínio (poses, $\\mathrm{Exp}$, Gauss–Newton no grupo, derivadas duais) e conferem os próprios números.\n"]
for slug, titulo, texto in apendices:
    linhas.append(f"- **{titulo[0].upper() + titulo[1:]}.** {link(apendice('5', slug))}. {texto}")
c.md("\n".join(linhas))

K5 = kalman("kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", "notebook 5")
K6 = kalman("kalman_quadrotor_quaternions.ipynb", "notebook 6")
c.md(f"""
---
## Para quem vem da série Kalman

Pontes com a série Kalman (pasta `kalman/` do mesmo repositório):
- os três filtros da etapa (d), com o erro nas coordenadas, no corpo e no mundo, são o EKF, o ESKF e o IEKF do {K5}, em $SE(2)$;
- o resultado de Barrau e Bonnabel visto na etapa (d) (o erro no mundo é invariante para medidas de pontos conhecidos, o erro no corpo para medidas de posição) é o que o {K5} discute em geral;
- a $\\mathrm{{Exp}}$, o $\\boxplus$ multiplicativo e o colchete preparam a pose do quadrirrotor do {K6}, onde a rotação já não comuta.

**Domínio anterior:** {link(arquivo("4"), "os números complexos")}. **Próximo domínio:** {link(arquivo("6"), "os quaternions")}, a rotação no espaço, onde a própria rotação deixa de comutar.
""")

c.salva()
