from comum import *

c = Caderno(arquivo("1"))
cabecalho(c, "Domínio I: os números reais",
          "Movimento de um ponto numa reta: o ponto de partida de toda a série",
          dom="1")

c.md(r"""
**O domínio.** O estado de um objeto que só anda numa reta é um número real: a posição $x\in\mathbb R$ (e, para a dinâmica, a velocidade $v\in\mathbb R$). Mover-se é **somar**: andar 3 m e depois 2 m é andar 5 m, em qualquer ordem. É o caso mais simples possível, e por isso o lugar certo para montar, com calma, as quatro ferramentas que a série vai levar até os quaternions tridentes.

**Os dois exemplos físicos deste domínio**
- **Um bloco com mola e atrito** numa mesa, sob uma força aplicada: o exemplo das etapas (b) e (c).
- **Pepé e Penélope num corredor:** ela foge, ele persegue, guiado por sensores com ruído. É o exemplo da etapa (d).

A animação abaixo mostra os dois. Ela é só para dar o tom: os detalhes estão nas folhas.
""")

c.code(FERRAMENTAS + r'''
from matplotlib import animation
from IPython.display import HTML
from scipy.linalg import expm

# bloco com mola e atrito viscoso (notebook c): solução exata por exponencial de matriz
A = np.array([[0.0, 1.0], [-4.0, -0.4]])
ts = np.linspace(0, 12, 60)
xb = np.array([(expm(A * t) @ np.array([1.0, 0.0]))[0] for t in ts])

# Pepé e Penélope: um trecho ilustrativo (o mundo completo está no notebook d)
rng = np.random.default_rng(5)
xa, va, xp, xa_l, xp_l = 40.0, 0.0, 30.0, [], []
for t in ts:
    for _ in range(4):
        d = xa - xp
        f = 1.5 * rng.standard_normal() + 3.0 * np.sign(d) * max(0, 1 - abs(d) / 8) - 50 * min(xa, 0) - 50 * max(xa - 60, 0)
        va += (f - va) * 0.05; xa += va * 0.05
        xp += np.clip(1.5 * (xa - xp), -2.5, 2.5) * 0.05
    xa_l.append(xa); xp_l.append(xp)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 3.0), dpi=60)
for a in (a1, a2):
    a.set_yticks([]); a.grid(False)
a1.set_xlim(-1.6, 1.6); a1.set_ylim(-0.5, 0.5); a1.set_title("bloco com mola e atrito", fontsize=9)
a1.axvline(-1.5, color="k", lw=3)
mola, = a1.plot([], [], color="gray"); bloco = a1.add_patch(plt.Rectangle((0, -0.2), 0.3, 0.4, color="tab:blue"))
a2.set_xlim(-2, 62); a2.set_ylim(-0.5, 0.5); a2.set_title("Pepé (roxo) persegue Penélope (preta) num corredor de 60 m", fontsize=9)
a2.axvline(0, color="k", lw=3); a2.axvline(60, color="k", lw=3)
pen, = a2.plot([], [], "o", color="k", ms=10); pep, = a2.plot([], [], "o", color="tab:purple", ms=10)
plt.tight_layout()

def quadro(i):
    x = xb[i]; s = np.linspace(-1.5, x, 60)
    mola.set_data(s, 0.1 * np.sin(np.linspace(0, 16 * np.pi, 60)))
    bloco.set_x(x)
    pen.set_data([xa_l[i]], [0]); pep.set_data([xp_l[i]], [0])
    return mola, bloco, pen, pep

anim = animation.FuncAnimation(fig, quadro, frames=len(ts), interval=120, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())
''')

c.md(r"""
---
## O que este domínio estabelece para a série

Em ℝ tudo é fácil, e isso permite fixar com clareza quatro ideias que vão ficar difíceis nos próximos domínios.

| ideia | em ℝ | o que muda depois |
|---|---|---|
| **compor movimentos** | somar: $g+h$ | a ordem passa a importar (rotações não comutam, a partir do domínio V) |
| **incremento pequeno** $x\boxplus\delta$ | $x+\delta$ | $x\otimes\mathrm{Exp}(\delta)$: o estado não é um vetor, mas o incremento é |
| **integrar uma EDO** | RK4: somar inclinações | RKMK: as inclinações vivem na álgebra de Lie e entram no estado pela $\mathrm{Exp}$ |
| **filtro de Kalman** | KF/EKF com $\hat x\leftarrow\hat x+\mathbf K\,\nu$ | ESKF: a correção entra pelo $\boxplus$ |

E uma ideia que vale em **todos** os domínios: um filtro de Kalman é um filtro de Bayes com crença gaussiana, e falha quando a crença verdadeira tem várias corcovas.

---
## As quatro folhas deste domínio
""")

resumo = {
    "a": ("Os axiomas de corpo e o que cada domínio da série perde. O `float` contra `Fraction` e `sympy`: "
          "associatividade e distributividade falham em 30% a 45% das triplas sorteadas; cancelamento catastrófico; "
          "soma de Kahan; a exponencial como ponte entre soma e produto; o dilema do passo das derivadas numéricas. "
          "A interface de domínio (`compoe`, `inverso`, `Exp`, `Log`, `⊞`, `⊟`) e os testes que valerão para os oito domínios."),
    "b": ("Blocos e molas: a matriz de rigidez e a eliminação de Gauss com pivotamento; o condicionamento "
          "(o conjunto que quase desliza sem esforço perde 11 dígitos); bisseção, Newton e secante com as ordens medidas "
          "(1, 2 e 1,6); Newton em sistemas com jacobiano à mão, simbólico e numérico; mínimos quadrados e Gauss–Newton "
          "para identificar a mola, que é o passo do filtro de Kalman iterado."),
    "c": ("Cinemática e por que integrar um acelerômetro erra 1,3 m em um minuto (o erro cresce como t^1,5); "
          "o bloco com mola e atrito resolvido pelo `sympy` nos três regimes; Euler, Euler semi-implícito e RK4 com as ordens "
          "medidas e o que cada um faz com a energia; o atrito seco, sem fórmula fechada, conferido pela queda linear da amplitude; "
          "a discretização exata que dá as matrizes F e Q do filtro de Kalman."),
    "d": ("Pepé persegue Penélope num corredor. Com uma régua, o filtro de Kalman nunca a perde. Com o olfato, que mede só a "
          "distância e não o lado, o EKF e o IEKF perdem o rastro em todas as simulações: ao passar por ela, Pepé passa a perseguir "
          "o reflexo dela. O filtro de Bayes numa grade, que carrega as duas hipóteses, não a perde em nenhuma."),
}
linhas = []
for e, slug, titulo in ETAPAS:
    arq = arquivo("1", e)
    linhas.append(f"### ({e}) {titulo.capitalize()}\n{link(arq)}\n\n{resumo[e]}\n")
c.md("\n".join(linhas))

K1 = kalman("mq_hmm_kalman_duas_trilhas.ipynb", "notebook 1")
K3 = kalman("pepe_kalman_parte1_ekf_iekf.ipynb", "notebook 3")
K4 = kalman("pepe_kalman_parte2_imm.ipynb", "notebook 4")
K5 = kalman("kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", "notebook 5")
c.md(f"""
---
## Para quem vem da série Kalman

A etapa (d) deste domínio é o "Pepé Le Gambá" reduzido a uma dimensão. Ela reaproveita, em miniatura, ideias de quatro notebooks daquela série (no mesmo repositório, pasta `kalman/`):
- o filtro de Bayes numa grade ({K5}, seção 2);
- o IEKF de Bell e Cathey ({K1}, {K3} e {K5});
- mirar onde a presa estará ({K4});
- o excesso de confiança medido pelo NEES ({K3}).

**Próximo domínio:** ℝ², o plano.
""")

c.salva()
