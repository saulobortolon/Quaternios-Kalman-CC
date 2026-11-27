from comum import *

c = Caderno(arquivo("3"))
cabecalho(c, "Domínio III: os números duais",
          "Um infinitesimal ε com ε² = 0: giro e deslizamento no mesmo eixo, e derivadas exatas",
          dom="3", anterior=arquivo("2"))

c.md(r"""
**O domínio.** Um número dual é $a+\varepsilon b$, com $\varepsilon^2=0$. Ele entra na série por dois motivos:
- **movimento:** o **ângulo dual** $\hat\theta=\theta+\varepsilon d$ descreve um giro $\theta$ em torno de um eixo junto com um deslizamento $d$ ao longo do mesmo eixo: o movimento de um parafuso, de uma junta cilíndrica, de Pepé agarrado a um mastro. O grupo de movimentos é $SO(2)\times\mathbb R$, o produto do círculo pela reta, e o estado vive num **cilindro**;
- **cálculo:** $f(a+\varepsilon)=f(a)+\varepsilon f'(a)$, exatamente. Avaliar um programa num número dual devolve a derivada do programa: é a **derivação automática**, que a série passa a usar para os jacobianos.

E uma novidade que não vem de $\varepsilon$: o ângulo **dá a volta**. É o primeiro estado da série que não é um vetor, e o $\boxplus$ e o $\boxminus$ deixam de ser uma soma e uma subtração.

**Os exemplos físicos deste domínio**
- **Uma barra num eixo vertical**, que gira e desliza, puxada por um fio e presa a molas: o exemplo da etapa (b).
- **O pêndulo de Wilberforce**: uma massa numa mola helicoidal que, ao esticar, torce, e troca energia entre o sobe-e-desce e o giro. É o exemplo da etapa (c).
- **Pepé no mastro, Penélope na parede de um silo:** o exemplo da etapa (d).

A animação mostra o pêndulo de Wilberforce (à esquerda: a altura e uma marca que gira com a massa) e o silo visto de cima (à direita).
""")

c.code(FERRAMENTAS + r'''
from matplotlib import animation
from IPython.display import HTML
from scipy.linalg import expm

# pêndulo de Wilberforce (notebook c): solução exata por exponencial de matriz
M_, I_, K_, D_, E_ = 0.5, 1e-4, 5.0, 1e-3, 1e-2
Minv = np.diag([1 / M_, 1 / I_]); Kq = np.array([[K_, E_ / 2], [E_ / 2, D_]])
A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Minv @ Kq, np.zeros((2, 2))]])
ts = np.linspace(0, 28, 60)
Y = np.array([expm(A * t) @ np.array([0.1, 0, 0, 0]) for t in ts])

# o silo visto de cima: Penélope dá voltas na parede, Pepé gira no mastro atrás dela
rng = np.random.default_rng(3)
phi, w_, th, PHI, TH = 3.0, 0.0, 2.0, [], []
for t in ts:
    for _ in range(9):
        w_ += (0.4 * rng.standard_normal() - w_) * 0.05 + 0.02
        phi += w_ * 0.05
        erro = np.angle(np.exp(1j * (phi - th)))
        th += np.clip(1.5 * erro, -0.5, 0.5) * 0.05
    PHI.append(phi); TH.append(th)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.4), dpi=60)
for a in (a1, a2):
    a.set_xticks([]); a.set_yticks([]); a.grid(False); a.set_aspect("equal")
a1.set_xlim(-1, 1); a1.set_ylim(-0.6, 0.6); a1.set_title("Wilberforce: sobe, desce e gira", fontsize=9)
a1.plot([0, 0], [0.6, 0.15], color="gray")
massa = a1.add_patch(plt.Circle((0, 0), 0.12, color="tab:blue"))
marca, = a1.plot([], [], color="w", lw=3)
a2.set_xlim(-6, 6); a2.set_ylim(-6, 6); a2.set_title("o silo visto de cima", fontsize=9)
a2.add_patch(plt.Circle((0, 0), 5, fill=False, lw=2)); a2.plot(0, 0, "ko", ms=6)
pen, = a2.plot([], [], "o", color="k", ms=9); seta, = a2.plot([], [], color="tab:purple", lw=3)
plt.tight_layout()

def quadro(i):
    zc = -3 * Y[i, 0]
    massa.center = (0, zc)
    marca.set_data([-0.12 * np.cos(Y[i, 1]), 0.12 * np.cos(Y[i, 1])], [zc, zc])
    pen.set_data([5 * np.cos(PHI[i])], [5 * np.sin(PHI[i])])
    seta.set_data([0, 2.5 * np.cos(TH[i])], [0, 2.5 * np.sin(TH[i])])
    return massa, marca, pen, seta

anim = animation.FuncAnimation(fig, quadro, frames=len(ts), interval=150, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())
''')

c.md(r"""
---
## O que este domínio estabelece para a série

| ideia | domínios I e II | domínio III | o que muda depois |
|---|---|---|---|
| **o estado** | um vetor | um ponto do **cilindro**: o ângulo dá a volta | rotações no plano (V) e no espaço (VI–VIII) também dão a volta, de modos mais ricos |
| **$\boxplus$ e $\boxminus$** | soma e subtração | soma e subtração **enroladas** | composição com $\mathrm{Exp}$ e $\mathrm{Log}$ de grupos não comutativos |
| **$\mathrm{Exp}$** | invertível | invertível só para giros menores que meia volta | idem, para todos os grupos de rotação |
| **jacobianos** | à mão, `sympy` ou diferenças finitas | **números duais**: exatos e automáticos | usados em todos os domínios seguintes |
| **integrar uma EDO** | RK4 | RK4 com $\boxplus$ = RKMK, porque o grupo comuta | RKMK com colchetes de Lie (a partir do V) |
| **ângulo dual** | — | $\theta+\varepsilon d$: giro e deslizamento no mesmo eixo | o princípio da transferência leva rotações a parafusos (VII) |

---
## As quatro folhas deste domínio
""")

resumo = {
    "a": ("Os duais como anel comutativo com divisores de zero, pela representação $\\begin{bmatrix}a&b\\\\0&a\\end{bmatrix}$; "
          "a série de Taylor que para no primeiro termo; uma classe `Dual` que deriva até um laço `while`, com erro zero onde "
          "a diferença central erra $5\\times10^{-12}$; o ângulo dual e o princípio da transferência; o círculo: `wrap`, "
          "$\\boxminus$ e a média circular; a interface de domínio para $SO(2)\\times\\mathbb R$, em que $\\mathrm{Exp}$ deixa de ser invertível."),
    "b": ("Sistemas lineares duais dão a sensibilidade da solução a um parâmetro, com uma fatoração só; jacobianos por "
          "números duais; o equilíbrio de uma barra num eixo, puxada por um fio, com sete equilíbrios (a mola de torção conta "
          "as voltas, e o ângulo não se enrola); identificar as molas com Gauss–Newton passando os duais através do próprio "
          "Newton (o teorema da função implícita, sem escrevê-lo)."),
    "c": ("A hélice e o parafuso; o pêndulo de Wilberforce pelas equações de Lagrange no `sympy`, com batimento de 28 s e "
          "99,9% da energia passando para o giro; RK4 com ordem 4 e energia conservada; RKMK = RK4 enrolado, porque o grupo "
          "comuta; a sensibilidade da trajetória por duais pelo RK4 e pela exponencial de uma matriz dual (o truque de Van Loan, "
          "com outro nome)."),
    "d": ("Pepé no mastro, Penélope na parede do silo. Sem $\\boxminus$ na inovação, cada passagem por ±180° faz a estimativa "
          "disparar (5% do tempo com erro acima de 30°, invisível na mediana); sem $\\boxminus$ no controle, Pepé gira pelo "
          "caminho longo e a vê só 54% do tempo. Os olhos de Pepé, com jacobiano dual, erram 0,2°; com a câmera do teto a 1 Hz "
          "como reserva, nunca a perdem."),
}
linhas = []
for e, slug, titulo in ETAPAS:
    arq = arquivo("3", e)
    linhas.append(f"### ({e}) {titulo[0].upper() + titulo[1:]}\n{link(arq)}\n\n{resumo[e]}\n")
c.md("\n".join(linhas))

K5 = kalman("kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", "notebook 5")
K6 = kalman("kalman_quadrotor_quaternions.ipynb", "notebook 6")
K3 = kalman("pepe_kalman_parte1_ekf_iekf.ipynb", "notebook 3")
c.md(f"""
---
## Para quem vem da série Kalman

Pontes com a série Kalman (pasta `kalman/` do mesmo repositório):
- o $\\boxplus$ e o $\\boxminus$ enrolados deste domínio são o caso mais simples do estado de erro do ESKF ({K5}, seções 5 e 7): a correção é calculada num espaço vetorial e aplicada ao estado por $\\boxplus$;
- os jacobianos do {K3} foram conferidos lá por diferenças finitas e pelo passo complexo; aqui, os números duais dão o mesmo resultado, exato, sem passo;
- o ângulo dual e o princípio da transferência preparam os dual quaternions do {K6}.

**Domínio anterior:** {link(arquivo("2"), "ℝ², o plano")}. **Próximo domínio:** {link(arquivo("4"), "os números complexos")}, em que a rotação vira multiplicação por $e^{{i\\theta}}$, e o enrolamento desaparece da álgebra.
""")

c.salva()
