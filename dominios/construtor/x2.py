from comum import *

c = Caderno(arquivo("2"))
cabecalho(c, "Domínio II: os vetores do plano",
          "Pontos no plano: duas coordenadas, medidas que as misturam",
          dom="2", anterior=arquivo("1"))

c.md(r"""
**O domínio.** O estado de um objeto que anda num plano, sem girar, é um **ponto** $\mathbf p=(x,y)\in\mathbb R^2$; para a dinâmica, junta-se a velocidade $\mathbf v=(v_x,v_y)$. Mover-se é somar um **vetor**: o grupo de movimentos é o das translações do plano, $(\mathbb R^2,+)$, que ainda é comutativo. Para o movimento, o plano é "duas retas lado a lado".

O que muda são as **perguntas**. A distância até um ponto, $\sqrt{x^2+y^2}$, e a direção, $\mathrm{atan2}(y,x)$, misturam as duas coordenadas e são não lineares. O arrasto do ar depende do **módulo** da velocidade. Cada etapa deste domínio é sobre um desses momentos em que as duas retas deixam de ser independentes.

**Os dois exemplos físicos deste domínio**
- **Uma bola de tênis de mesa lançada no ar**, com arrasto: o exemplo das etapas (b) e (c). Leve e grande para a massa, ela tem um arrasto forte, que muda tudo no voo.
- **Pepé e Penélope numa arena quadrada de 40 m:** ela foge, ele persegue, guiado por sensores de cheiro. É o exemplo da etapa (d).

A animação abaixo mostra os dois. Ela é só para dar o tom: os detalhes estão nas folhas.
""")

c.code(FERRAMENTAS + r'''
from matplotlib import animation
from IPython.display import HTML

# a bola de tênis de mesa (notebook c): no vácuo e com arrasto quadrático, lançada a 10 m/s e 45°
G, K_ARR = 9.81, 0.1396

def voo(arrasto, n=60, T=1.45):
    s, h, traj = np.array([0.0, 0.0, 10 * np.cos(np.pi / 4), 10 * np.sin(np.pi / 4)]), T / (n * 20), []
    for i in range(n * 20):
        v = np.hypot(s[2], s[3]) if arrasto else 0.0
        s = s + h * np.array([s[2], s[3], -K_ARR * v * s[2], -G - K_ARR * v * s[3]])
        if i % 20 == 19:
            traj.append(s[:2].copy() if s[1] >= 0 else traj[-1])
    return np.array(traj)

bola_vacuo, bola_ar = voo(False), voo(True)

# Pepé e Penélope: um trecho ilustrativo (o mundo completo está no notebook d)
rng = np.random.default_rng(7)
pa, va, pp, f_al, PA, PP = np.array([28.0, 20.0]), np.zeros(2), np.array([12.0, 20.0]), np.zeros(2), [], []
for k in range(60):
    for _ in range(6):
        d = pa - pp; r = np.hypot(*d)
        f_al += -f_al / 1.5 * 0.05 + 2.0 * np.sqrt(2 * 0.05 / 1.5) * rng.standard_normal(2)
        f = f_al + 3.0 * d / r * max(0, 1 - r / 8) - 50 * np.minimum(pa, 0) - 50 * np.maximum(pa - 40, 0)
        va += (f - va) * 0.05; pa = pa + va * 0.05
        vp = 1.5 * (pa - pp); n_ = np.hypot(*vp); vp = vp * min(1, 2.5 / n_)
        pp = pp + vp * 0.05
    PA.append(pa.copy()); PP.append(pp.copy())
PA, PP = np.array(PA), np.array(PP)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.4), dpi=60, gridspec_kw={"width_ratios": [1.6, 1]})
a1.set_xlim(-0.3, 10.5); a1.set_ylim(-0.2, 3.0); a1.set_aspect("equal"); a1.set_title("bola no vácuo (pontilhada) e no ar", fontsize=9)
a2.set_xlim(-2, 42); a2.set_ylim(-2, 42); a2.set_aspect("equal"); a2.set_title("Pepé (roxo) persegue Penélope (preta)", fontsize=9)
a2.add_patch(plt.Rectangle((0, 0), 40, 40, fill=False, lw=2))
for a in (a1, a2):
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
rastro_v, = a1.plot([], [], ":", color="gray"); rastro_a, = a1.plot([], [], color="tab:blue")
b_v, = a1.plot([], [], "o", color="gray", ms=6); b_a, = a1.plot([], [], "o", color="tab:blue", ms=6)
r_pa, = a2.plot([], [], color="k", lw=.8, alpha=.5); r_pp, = a2.plot([], [], color="tab:purple", lw=.8, alpha=.5)
m_pa, = a2.plot([], [], "o", color="k", ms=8); m_pp, = a2.plot([], [], "o", color="tab:purple", ms=8)
plt.tight_layout()

def quadro(i):
    rastro_v.set_data(*bola_vacuo[:i + 1].T); rastro_a.set_data(*bola_ar[:i + 1].T)
    b_v.set_data([bola_vacuo[i, 0]], [bola_vacuo[i, 1]]); b_a.set_data([bola_ar[i, 0]], [bola_ar[i, 1]])
    j = max(0, i - 15)
    r_pa.set_data(*PA[j:i + 1].T); r_pp.set_data(*PP[j:i + 1].T)
    m_pa.set_data([PA[i, 0]], [PA[i, 1]]); m_pp.set_data([PP[i, 0]], [PP[i, 1]])
    return rastro_v, rastro_a, b_v, b_a, r_pa, r_pp, m_pa, m_pp

anim = animation.FuncAnimation(fig, quadro, frames=60, interval=120, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())
''')

c.md(r"""
---
## O que este domínio estabelece para a série

| ideia | no domínio I (ℝ) | no domínio II (ℝ²) | o que muda depois |
|---|---|---|---|
| **compor movimentos** | somar números | somar vetores, coordenada a coordenada | com rotações (domínio V em diante), a ordem passa a importar |
| **produto de elementos** | o produto de ℝ | não há um produto natural; as escolhas $\mathbf e^2=0$ e $\mathbf e^2=-1$ dão os duais e os complexos | os domínios III e IV fazem essas escolhas |
| **estado e incremento** | ambos números | o estado é um **ponto**, o incremento é um **vetor** | o estado vira uma pose; o incremento continua vetor: $x\boxplus\delta$ |
| **integrar uma EDO** | RK4 | RK4 com estado vetorial | RKMK |
| **filtro** | KF, EKF, filtro em grade | KF = dois KFs da reta; EKF; **filtro de partículas** | ESKF |

E duas ideias que valem em **todos** os domínios seguintes:
- **a incerteza propagada por uma dinâmica não linear deixa de ser gaussiana** (a "banana" da etapa c), e a escolha das coordenadas decide se ela volta a ser;
- **um sensor que mede só parte do estado** (só a distância, só a direção) gera ambiguidades que nenhum filtro resolve: é preciso outro sensor, ou movimento.

---
## As quatro folhas deste domínio
""")

resumo = {
    "a": ("Os axiomas de espaço vetorial; pontos e vetores (o espaço afim e por que não se somam dois pontos); "
          "os produtos de ℝ²: escalar, determinante, componente a componente, e a família $\\mathbf e^2=\\alpha$ que dá "
          "os complexos, os duais e os hiperbólicos; `hypot` e `atan2`; o teste de orientação, que em `float` erra o lado "
          "em 1% de uma grade de pontos quase alinhados; translações e transformações lineares; a interface de domínio em ℝ²."),
    "b": ("Triangulação: duas retas que se cruzam, com condicionamento $\\cot(\\gamma/2)$; trilateração: dois círculos e a "
          "ambiguidade do reflexo, o truque que torna o sistema linear, Gauss–Newton na cota de Cramér–Rao, a GDOP e o que ela "
          "não vê; a mira de uma bola com arrasto: Newton em (ângulo, tempo), o tiro direto e o tiro por cima, e as bacias de atração."),
    "c": ("Sem arrasto, com arrasto linear (as coordenadas continuam separadas; o `sympy` resolve tudo) e com arrasto quadrático "
          "(o módulo da velocidade mistura as coordenadas); a queda vertical com `tanh`; RK4 vetorial com ordem 4 e balanço de "
          "energia exato; o ângulo ótimo cai de 45° para 32° a 30 m/s; a incerteza do lançamento vira uma elipse e depois uma "
          "\"banana\"; F e Q do plano = duas cópias das da reta."),
    "d": ("Pepé persegue Penélope numa arena. A régua: o KF do plano é exatamente dois KFs da reta. A intensidade do cheiro só "
          "diz a distância: o EKF a perde em 20 de 20 simulações. O gradiente diz a direção mas confunde perto com longe. "
          "Os dois juntos: erro mediano de 8 cm, sem nenhuma perda. O filtro de partículas mostra a forma de cada crença "
          "(um arco; uma nuvem que pode cair na distância gêmea; um ponto) e ajuda com a ambiguidade discreta, não com a contínua."),
}
linhas = []
for e, slug, titulo in ETAPAS:
    arq = arquivo("2", e)
    linhas.append(f"### ({e}) {titulo[0].upper() + titulo[1:]}\n{link(arq)}\n\n{resumo[e]}\n")
c.md("\n".join(linhas))

K3 = kalman("pepe_kalman_parte1_ekf_iekf.ipynb", "notebook 3")
K4 = kalman("pepe_kalman_parte2_imm.ipynb", "notebook 4")
K5 = kalman("kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", "notebook 5")
K6 = kalman("kalman_quadrotor_quaternions.ipynb", "notebook 6")
c.md(f"""
---
## Para quem vem da série Kalman

A etapa (d) deste domínio é o "Pepé Le Gambá" no plano, mais perto do original. Pontes com a série Kalman (pasta `kalman/` do mesmo repositório):
- o gradiente do cheiro é um dos sensores de Pepé no {K3}, e lá também é a combinação de sensores que resolve a perseguição;
- a propagação da incerteza pelo jacobiano ($\\mathbf J\\mathbf P\\mathbf J^\\top$) contra o Monte Carlo, na etapa (c), é a predição do EKF do {K5};
- a "banana" da etapa (c) é a razão de ser do estado de erro e do $\\boxplus$, que o {K5} e o {K6} usam com rotações;
- o filtro de partículas carrega várias hipóteses ao mesmo tempo, como o IMM do {K4} carrega vários modos.

**Domínio anterior:** {link(arquivo("1"), "ℝ, a reta")}. **Próximo domínio:** {link(arquivo("3"), "os números duais")}.
""")

c.salva()
