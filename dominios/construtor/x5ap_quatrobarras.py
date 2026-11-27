from comum import *
from dc_codigo import CLASSE_DC

c = Caderno(apendice("5", "quatrobarras"))
cabecalho(c, "Domínio V, apêndice: o mecanismo de quatro barras",
          "Uma manivela, uma biela, um balancim: a curva do acoplador e o centro instantâneo como ponto fixo de uma pose",
          dom="5", etapa=None, anterior=arquivo("5"))

c.md(r"""
**O mecanismo.** Quatro barras rígidas ligadas por quatro juntas de rotação formam um laço fechado: a **base** (fixa, de $O_2$ a $O_4$, comprimento $d$), a **manivela** (de $O_2$ a $A$, comprimento $a$), o **acoplador** (de $A$ a $B$, comprimento $b$) e o **balancim** (de $B$ a $O_4$, comprimento $c$). Gira-se a manivela, e o resto se move de um único jeito (escolhido o ramo de montagem). É o mecanismo mais estudado da cinemática: limpadores de para-brisa, pedais de bicicleta ergométrica, suspensões de carro, a perna do cavalo mecânico de Chebyshev.

**Em números complexos**, o laço fecha quando

$$a\,e^{i\theta_2}+b\,e^{i\theta_3}-c\,e^{i\theta_4}-d=0,$$

uma equação complexa (duas reais) para as duas incógnitas $\theta_3,\theta_4$, dado $\theta_2$. O **acoplador** é a barra interessante: ele não gira em torno de um ponto fixo, e a sua pose é um dual-complexo $q_3=\text{pose}(\theta_3,A)$. Um ponto preso a ele descreve a **curva do acoplador**, que pode ter formas surpreendentes (laços, cúspides, trechos quase retos).

**O que este notebook mostra.**
1. A posição por interseção de círculos, conferida pela equação do laço.
2. As velocidades angulares, por derivação da equação do laço, conferidas por diferenças finitas.
3. O **centro instantâneo** do acoplador como o **ponto fixo** da pose incremental $\Delta q=q_3(\theta_2+h)\,q_3(\theta_2)^{-1}$ (a dobradiça da porta do notebook V(a), seção 6), conferido pelo **teorema de Kennedy**: ele fica na interseção das retas $O_2A$ e $O_4B$.
4. A **condição de Grashof**, conferida por uma varredura numérica.
""")

c.code(FERRAMENTAS + CLASSE_DC + r'''
O2, O4 = 0j, 4.0 + 0j
a_, b_, c_, d_ = 1.0, 3.5, 3.0, 4.0

def posicao(t2, a=a_, b=b_, c=c_, d=d_, ramo=+1):
    """θ3, θ4 e os pontos A, B, pela interseção do círculo (A, b) com o círculo (O4, c). None se o laço não fecha."""
    A = a * np.exp(1j * t2); O4_ = d + 0j
    D = abs(O4_ - A)
    if D > b + c or D < abs(b - c):
        return None
    x = (b**2 - c**2 + D**2) / (2 * D)                  # distância de A ao pé da corda, ao longo de A→O4
    h = np.sqrt(max(b**2 - x**2, 0.0))
    u = (O4_ - A) / D
    B = A + x * u + ramo * h * 1j * u                   # ramo +1: montagem "aberta"
    return np.angle(B - A), np.angle(B - O4_), A, B

t3, t4, A, B = posicao(np.radians(60))
laco = a_ * np.exp(1j * np.radians(60)) + b_ * np.exp(1j * t3) - c_ * np.exp(1j * t4) - d_
confere("a equação do laço fecha (θ2 = 60°)", laco, 0, tol=1e-13)
confere("os comprimentos das barras se mantêm", [abs(B - A), abs(B - O4)], [b_, c_], tol=1e-13)
print(f"θ2 = 60°:  θ3 = {np.degrees(t3):.3f}°,  θ4 = {np.degrees(t4):.3f}°")
''')

c.md(r"""
## 1. Velocidades: derivar a equação do laço

Derivando a equação do laço no tempo, com $\omega_k=\dot\theta_k$:

$$i a\omega_2e^{i\theta_2}+i b\omega_3e^{i\theta_3}-i c\omega_4e^{i\theta_4}=0,$$

uma equação complexa **linear** em $\omega_3,\omega_4$. Separando parte real e imaginária, é um sistema $2\times2$. É o mesmo truque do jacobiano do braço (notebook V(b)): a cinemática de posição é não linear, a de velocidade é linear.
""")

c.code(r'''
def velocidades(t2, w2=1.0):
    t3, t4, A, B = posicao(t2)
    M = np.array([[(1j * b_ * np.exp(1j * t3)).real, (-1j * c_ * np.exp(1j * t4)).real],
                  [(1j * b_ * np.exp(1j * t3)).imag, (-1j * c_ * np.exp(1j * t4)).imag]])
    rhs = -1j * a_ * w2 * np.exp(1j * t2)
    return np.linalg.solve(M, [rhs.real, rhs.imag])

t2, h = np.radians(60), 1e-6
w3, w4 = velocidades(t2)
p_mais, p_menos = posicao(t2 + h), posicao(t2 - h)
w3_df = wrap(p_mais[0] - p_menos[0]) / (2 * h); w4_df = wrap(p_mais[1] - p_menos[1]) / (2 * h)
print(f"com a manivela a 1 rad/s:  ω3 = {w3:.6f},  ω4 = {w4:.6f} rad/s")
confere("ω3 e ω4 pela equação do laço = diferenças finitas", [w3, w4], [w3_df, w4_df], tol=1e-7)
''')

c.md(r"""
## 2. O centro instantâneo do acoplador: o ponto fixo de uma pose

Entre dois instantes próximos, o acoplador sofre o deslocamento rígido $\Delta q=q_3(\theta_2+h)\,q_3(\theta_2)^{-1}$. Todo deslocamento rígido do plano que gira (ângulo $\Delta\theta\neq0$) é uma rotação em torno de **um** ponto, o seu ponto fixo: $e^{i\Delta\theta}p+t=p$, ou seja, $p=t/(1-e^{i\Delta\theta})$. Quando $h\to0$, esse ponto tende ao **centro instantâneo de rotação** do acoplador: naquele instante, o acoplador inteiro gira em torno dele, como uma porta em torno da dobradiça.

O **teorema de Kennedy** diz onde ele está sem conta nenhuma: na interseção das retas $O_2A$ (o ponto $A$ se move perpendicularmente à manivela) e $O_4B$ (idem para $B$ e o balancim). Os dois caminhos têm de dar o mesmo ponto.
""")

c.code(r'''
def pose_acoplador(t2):
    t3, t4, A, B = posicao(t2)
    return pose(t3, A)

def centro_instantaneo(t2, h=1e-5):
    dq = pose_acoplador(t2 + h) * pose_acoplador(t2 - h).conj()
    return translacao(dq) / (1 - np.exp(1j * angulo(dq)))

def intersecao(p1, d1, p2, d2):
    """interseção das retas p1 + s·d1 e p2 + u·d2"""
    M = np.array([[d1.real, -d2.real], [d1.imag, -d2.imag]])
    s = np.linalg.solve(M, [(p2 - p1).real, (p2 - p1).imag])[0]
    return p1 + s * d1

for graus in [30, 60, 120, 200, 300]:
    t2 = np.radians(graus)
    _, _, A, B = posicao(t2)
    ci = centro_instantaneo(t2)
    kennedy = intersecao(O2, A - O2, O4, B - O4)
    print(f"θ2 = {graus:3d}°:  ponto fixo de Δq = {ci:.5f};   Kennedy = {kennedy:.5f}")
    confere(f"  ponto fixo = Kennedy em θ2 = {graus}°", ci, kennedy, tol=1e-5, relativo=True)
''')

c.md(r"""
## 3. A curva do acoplador

Um ponto preso ao acoplador, nas coordenadas locais dele (origem em $A$, eixo real ao longo de $AB$), é levado ao mundo pela pose: $P=q_3\cdot p_{\text{local}}$ (a função `aplica`). Varrendo a manivela por uma volta inteira, $P$ desenha a curva do acoplador.
""")

c.code(r'''
p_local = 1.2 + 1.6j                                 # um ponto do acoplador, fora da reta AB
angs = np.radians(np.linspace(0, 360, 721))
curva = np.array([aplica(pose_acoplador(t), p_local) for t in angs])
confere("a curva fecha depois de uma volta da manivela", curva[-1], curva[0], tol=1e-12)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(curva.real, curva.imag, "C3-", lw=1.5, label="curva do acoplador")
for graus, alfa in [(40, 1.0), (150, 0.55), (260, 0.3)]:
    t2 = np.radians(graus); _, _, A, B = posicao(t2)
    P = aplica(pose_acoplador(t2), p_local)
    ax.plot([O2.real, A.real, B.real, O4.real], [O2.imag, A.imag, B.imag, O4.imag], "k-o", ms=4, alpha=alfa, lw=2)
    ax.fill([A.real, B.real, P.real], [A.imag, B.imag, P.imag], color="C0", alpha=0.25 * alfa)
    ci = centro_instantaneo(t2)
    if abs(ci) < 12:
        ax.plot(ci.real, ci.imag, "x", color="C2", ms=9, alpha=alfa)
        ax.plot([A.real, ci.real], [A.imag, ci.imag], ":", color="C2", alpha=alfa)
        ax.plot([B.real, ci.real], [B.imag, ci.imag], ":", color="C2", alpha=alfa)
ax.plot([O2.real, O4.real], [O2.imag, O4.imag], "k-", lw=4, alpha=0.3)
ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower left")
ax.set_title("quatro barras (a=1, b=3,5, c=3, d=4): três posições, centros instantâneos (×) e a curva de um ponto do acoplador", fontsize=9)
plt.tight_layout(); plt.show()
''')

c.md(r"""
## 4. A condição de Grashof

Quando a manivela consegue dar a volta inteira? Com $s$ e $l$ a menor e a maior das quatro barras, e $p$, $q$ as outras duas, Grashof (1883) mostrou que **alguma** barra dá a volta completa se e só se

$$s+l\le p+q.$$

Se, além disso, a barra mais curta é a manivela, o mecanismo é **manivela–balancim** (a manivela gira, o balancim oscila); se a mais curta é a base, os dois braços giram (**dupla manivela**). Se a condição falha, nenhuma barra gira inteira: **duplo balancim**. A conferência abaixo varre $\theta_2$ e verifica, numericamente, se o laço fecha em todos os ângulos.
""")

c.code(r'''
def manivela_gira(a, b, c, d):
    return all(posicao(t, a, b, c, d) is not None for t in np.linspace(0, 2 * np.pi, 2001))

def grashof(a, b, c, d):
    L = sorted([a, b, c, d]); return L[0] + L[3] <= L[1] + L[2]

casos = [(1.0, 3.5, 3.0, 4.0), (1.0, 2.0, 2.0, 2.5), (2.0, 2.0, 2.0, 1.0), (2.0, 3.0, 2.5, 4.0), (1.5, 4.0, 2.0, 4.0), (3.0, 1.0, 3.5, 4.0)]
print(" a     b     c     d    Grashof   a manivela (a) gira?   tipo")
for a, b, c, d in casos:
    g, gira = grashof(a, b, c, d), manivela_gira(a, b, c, d)
    menor = min(a, b, c, d)
    tipo = ({a: "manivela–balancim", d: "dupla manivela", b: "o acoplador gira; a e c só oscilam", c: "balancim–manivela (quem gira é c)"}[menor]) if g else "duplo balancim"
    print(f"{a:4.1f}  {b:4.1f}  {c:4.1f}  {d:4.1f}   {'sim' if g else 'não':5s}     {'sim' if gira else 'não':5s}               {tipo}")
    confere(f"  varredura concorda com Grashof", gira, g and menor in (a, d))   # a gira inteira se a ou a base é a mais curta
''')

c.md(r"""
**Leitura.** A posição por interseção de círculos fecha a equação do laço com erro de arredondamento, e as velocidades pela equação derivada coincidem com as diferenças finitas. O centro instantâneo calculado como **ponto fixo** da pose incremental do acoplador, um cálculo de dual-complexos que não sabe nada de mecanismos, cai exatamente na interseção das retas da manivela e do balancim, como manda o teorema de Kennedy. Na figura, as linhas pontilhadas verdes, de $A$ e de $B$ até o centro, são os prolongamentos da manivela e do balancim. A tabela de Grashof concorda com a varredura numérica nos seis casos: a manivela $a$ dá a volta inteira exatamente quando a condição vale **e** a barra mais curta é ela ou a base. No último caso a condição vale, mas a mais curta é o acoplador: ele gira inteiro em relação às vizinhas, e a manivela só oscila.

## Exercícios

1. **A perna de Chebyshev.** O mecanismo de Chebyshev com $d=2$, $a=1$, $b=c=2{,}5$ e o ponto do acoplador no prolongamento de $AB$, a uma distância $5$ de $A$, tem uma curva do acoplador com um trecho quase reto. Desenhe-a e meça o quanto o trecho se afasta de uma reta.
2. **O ângulo de transmissão.** O ângulo $\mu$ entre o acoplador e o balancim mede a qualidade da transmissão de força: perto de $0°$ ou $180°$, a manivela empurra o balancim quase pela haste, e o mecanismo trava. Calcule $\mu$ ao longo da volta e o seu mínimo. Projetistas pedem $\mu_{\min}\ge40°$; este mecanismo passa?
3. **O outro ramo.** Troque `ramo=+1` por `ramo=-1` (a montagem "cruzada"). A curva do acoplador muda? E o centro instantâneo continua no ponto de Kennedy?
4. **Síntese.** Escolha três posições desejadas para o acoplador (três poses $q$) e encontre o mecanismo de quatro barras que passa por elas (o problema de Burmester, com três posições, tem solução de régua e compasso). Dica: para cada um dos pontos $A$ e $B$, as três posições ficam numa circunferência; o centro dela é $O_2$ (ou $O_4$).
""")

c.salva()
