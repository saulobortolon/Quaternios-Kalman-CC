from comum import *

c = Caderno(arquivo("4"))
cabecalho(c, "Domínio IV: os números complexos",
          "O corpo que gira: rotações do plano como multiplicação por e^{iθ}",
          dom="4", anterior=arquivo("3"))

c.md(r"""
**O domínio.** Os complexos $z=a+ib$, com $i^2=-1$, são a escolha $\alpha=-1$ da família de produtos de ℝ² (notebook II(a)), e a única que dá um **corpo**. A geometria é o que interessa à série: **multiplicar por $e^{i\theta}$ é girar o plano de $\theta$**. A rotação deixa de ser um ângulo, que precisa ser enrolado a cada conta (domínio III), e passa a ser um **número de módulo 1**, que se compõe por multiplicação. O grupo das rotações do plano, $SO(2)$, aparece como $U(1)$, o círculo unitário de ℂ.

Três consequências que atravessam o domínio:
- o enrolamento **sai da composição** ($e^{i(\theta+2\pi)}=e^{i\theta}$ automaticamente) e fica só no $\mathrm{Log}$, como um corte no semieixo real negativo;
- em `float`, um complexo "de módulo 1" vai deixando de ter módulo 1, e o Runge–Kutta clássico **encolhe** as rotações; o RKMK, que multiplica, não;
- o $\boxplus$ passa a ser **multiplicativo**, $q\,e^{i\delta}$. Num grupo comutativo, isso coincide com somar ao ângulo; nos domínios seguintes, não coincidirá.

**Os exemplos físicos deste domínio**
- **Molas forçadas** por uma força senoidal, resolvidas com **fasores**: o exemplo da etapa (b).
- **O pêndulo de Foucault**: o plano de oscilação gira porque a Terra gira debaixo dele. É o exemplo da etapa (c).
- **Pepé com nariz**: um perseguidor que anda para a frente, vê com uma câmera presa ao corpo, e estima a própria orientação com um giroscópio que tem viés. É o exemplo da etapa (d).

A animação mostra o pêndulo de Foucault numa Terra fictícia que gira muito depressa (à esquerda) e Pepé, com o nariz, perseguindo Penélope (à direita).
""")

c.code(FERRAMENTAS + r'''
from matplotlib import animation
from IPython.display import HTML

# Foucault (notebook c), com a Terra 300× mais rápida, para ver a precessão
W0, OZ = np.sqrt(9.81 / 67), 300 * 7.2921e-5 * np.sin(np.radians(48.85))
s, zs = np.array([3.0 + 0j, 0j]), []
f = lambda s: np.array([s[1], -2j * OZ * s[1] - W0**2 * s[0]])
for i in range(60 * 40):
    h = 0.1
    k1 = f(s); k2 = f(s + h / 2 * k1); k3 = f(s + h / 2 * k2); k4 = f(s + h * k3)
    s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    zs.append(s[0])
zs = np.array(zs)

# Pepé com nariz persegue Penélope (um trecho ilustrativo; o mundo completo está no notebook d)
rng = np.random.default_rng(5)
a, va, fa, p, q, A_, P_, Q_ = 28 + 20j, 0j, 0j, 12 + 20j, 1 + 0j, [], [], []
for k in range(60):
    for _ in range(8):
        fa += -fa / 1.5 * 0.05 + 2 * np.sqrt(0.1 / 1.5) * (rng.standard_normal() + 1j * rng.standard_normal())
        d = a - p; r = abs(d)
        fr = fa + 3 * d / r * max(0, 1 - r / 8) - 50 * (min(a.real, 0) + max(a.real - 40, 0)) - 50j * (min(a.imag, 0) + max(a.imag - 40, 0))
        va += (fr - va) * 0.05; a += va * 0.05
        e = np.angle(np.conj(q) * (a - p))
        q *= np.exp(1j * np.clip(3 * e, -2, 2) * 0.05)
        p += np.clip(1.5 * r * max(np.cos(e), 0), 0, 2.5) * q * 0.05
    A_.append(a); P_.append(p); Q_.append(q)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6), dpi=60)
for ax in (a1, a2):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
a1.set_xlim(-3.3, 3.3); a1.set_ylim(-3.3, 3.3); a1.set_title("Foucault (Terra 300× mais rápida)", fontsize=9)
rastro, = a1.plot([], [], lw=0.6, color="tab:blue"); bola, = a1.plot([], [], "o", color="tab:blue", ms=7)
a2.set_xlim(-2, 42); a2.set_ylim(-2, 42); a2.set_title("Pepé com nariz", fontsize=9)
a2.add_patch(plt.Rectangle((0, 0), 40, 40, fill=False, lw=2))
pen, = a2.plot([], [], "ko", ms=8); pepe, = a2.plot([], [], "o", color="tab:purple", ms=9); nariz, = a2.plot([], [], color="tab:purple", lw=3)
plt.tight_layout()

def quadro(i):
    rastro.set_data(zs[:40 * (i + 1)].real, zs[:40 * (i + 1)].imag); bola.set_data([zs[40 * i + 39].real], [zs[40 * i + 39].imag])
    pen.set_data([A_[i].real], [A_[i].imag]); pepe.set_data([P_[i].real], [P_[i].imag])
    ponta = P_[i] + 3 * Q_[i]
    nariz.set_data([P_[i].real, ponta.real], [P_[i].imag, ponta.imag])
    return rastro, bola, pen, pepe, nariz

anim = animation.FuncAnimation(fig, quadro, frames=60, interval=150, blit=True)
plt.close(fig)
HTML(anim.to_jshtml())
''')

c.md(r"""
---
## O que este domínio estabelece para a série

| ideia | domínio III (ângulo dual) | domínio IV (complexos) | o que muda depois |
|---|---|---|---|
| **a rotação** | um ângulo que se enrola | um número de módulo 1, $e^{i\theta}$ | quaternions unitários no espaço (VI) |
| **compor** | $\mathrm{wrap}(\theta_1+\theta_2)$ | $q_1q_2$, sem exceções | $q_1q_2\neq q_2q_1$ no espaço |
| **$\boxplus$** | $\mathrm{wrap}(\theta+\delta)$ | $q\,e^{i\delta}$ (igual, porque comuta) | só a forma multiplicativa é correta |
| **em `float`** | o ângulo acumula erro de soma | o módulo escorrega; renormalizar | idem para quaternions |
| **integrar** | RK4 = RKMK | RK4 **encolhe** $q$; RKMK fica no círculo | RKMK com colchetes de Lie |
| **derivar** | números duais | o **passo complexo**, sem classe nenhuma | os dois, conforme o caso |
| **filtro** | $\boxminus$ na inovação e no controle | estado de erro $\delta\psi$ e viés do giroscópio | o ESKF completo (V a VIII) |

---
## As quatro folhas deste domínio
""")

resumo = {
    "a": ("ℂ como corpo (a representação $\\begin{bmatrix}a&-b\\\\b&a\\end{bmatrix}$) e por que não há ordem; a fórmula de Euler pelo `sympy`; "
          "multiplicar gira e estica, somar translada; $U(1)$ e o corte do logaritmo; um milhão de rotações em `float`: o módulo "
          "escorrega linearmente ($5\\times10^{-11}$), somar ângulos erra $10^{-6}$; a divisão do Python não transborda; o passo complexo, "
          "com erro zero em $h=10^{-200}$ (e errado, sem aviso, com `abs`); a interface de domínio em $U(1)$."),
    "b": ("Fasores: três blocos com molas forçados por uma senoide viram um sistema linear complexo $3\\times3$, igual ao real $6\\times6$ "
          "e à simulação depois do transitório; a curva de ressonância; Newton complexo e Cauchy–Riemann (o jacobiano é um número "
          "complexo); o fractal de Newton para $z^3-1$; raízes como autovalores da matriz companheira; o polinômio de Wilkinson, "
          "com raízes que saltam para fora do eixo real por uma mudança de $10^{-7}$ num coeficiente."),
    "c": ("Coriolis e centrífuga numa linha de derivação; o pêndulo de Foucault como uma EDO complexa resolvida pelo `sympy`: uma volta "
          "do plano em 31,8 h em Paris, 70 h a 20° de latitude, sentido invertido no hemisfério sul; a precessão medida numa simulação "
          "de uma hora concorda com a teoria em $5\\times10^{-4}$ grau; o RK4 encolhe a orientação ($\\lvert R(ix)\\rvert^2=1-x^6/72+\\dots$), "
          "o RKMK não; o estado de erro de orientação e viés para a etapa (d)."),
    "d": ("Pepé com nariz e câmera no corpo. Sem procura ativa, perde Penélope em 6 de 20 simulações; com procura, nunca. Com o "
          "giroscópio sem correção, o mapa gira (erro de 2 m) mas a perseguição não sofre, porque os erros se cancelam no referencial "
          "do corpo. Com faróis, o filtro conjunto estima orientação (0,2°) e viés, e o mapa volta a 25 cm. ⊞ multiplicativo = aditivo "
          "até $10^{-11}$."),
}
linhas = []
for e, slug, titulo in ETAPAS:
    arq = arquivo("4", e)
    linhas.append(f"### ({e}) {titulo[0].upper() + titulo[1:]}\n{link(arq)}\n\n{resumo[e]}\n")
c.md("\n".join(linhas))

K3 = kalman("pepe_kalman_parte1_ekf_iekf.ipynb", "notebook 3")
K5 = kalman("kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", "notebook 5")
K6 = kalman("kalman_quadrotor_quaternions.ipynb", "notebook 6")
c.md(f"""
---
## Para quem vem da série Kalman

Pontes com a série Kalman (pasta `kalman/` do mesmo repositório):
- o passo complexo da etapa (a) é a ferramenta que o {K3} e o {K5} usam para conferir jacobianos;
- o estado de erro $\\delta\\psi$ com $q=\\hat q\\,e^{{i\\delta\\psi}}$ e o viés do giroscópio da etapa (d) são o ESKF do {K5} em uma dimensão de rotação;
- o RK4 que encolhe a orientação e o RKMK que não encolhe preparam a integração de atitude com quaternions do {K6}.

**Domínio anterior:** {link(arquivo("3"), "os números duais")}. **Próximo domínio:** os números dual-complexos, que juntam $i$ e $\\varepsilon$ para descrever a **pose** no plano: posição e orientação num número só.
""")

c.salva()
