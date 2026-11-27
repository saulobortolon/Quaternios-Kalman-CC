from comum import *
from q_codigo import FUNCOES_Q

c = Caderno(apendice("6", "bola"))
cabecalho(c, "Domínio VI, apêndice: a bola que rola",
          "Rolar uma bola por um caminho fechado e encontrá-la girada: o colchete de Lie que se sente com a mão",
          dom="6", etapa=None, anterior=arquivo("6"))

c.md(r"""
**A experiência.** Ponha uma bola sobre a mesa, com uma marca no topo. Role-a, sem deixar escorregar e sem girá-la em torno da vertical, ao longo de um quadrado: um palmo para a frente, um para a direita, um para trás, um para a esquerda. A bola volta ao mesmo lugar, mas **não** à mesma orientação: ela girou em torno da vertical. Rolar em $x$ e rolar em $y$ não comutam, e a diferença entre as duas ordens é um giro em torno de $z$, a direção que nenhum dos dois movimentos produz sozinho. É o colchete de Lie do notebook VI(c), $[\hat{\mathbf x},\hat{\mathbf y}]=\hat{\mathbf z}$, que se pode sentir com a mão.

**O modelo.** Uma esfera de raio $r$ rola sem escorregar sobre o plano $z=0$, com o centro andando com velocidade $\mathbf v$ (horizontal) e sem giro em torno da vertical. O ponto de contato tem velocidade zero, $\mathbf v+\boldsymbol\omega\times(-r\hat{\mathbf z})=0$, o que dá

$$\boldsymbol\omega=\frac{\hat{\mathbf z}\times\mathbf v}{r}\quad\text{(no mundo)},\qquad \dot q=\tfrac12\,(0,\boldsymbol\omega)\otimes q.$$

A velocidade angular está no referencial do **mundo**, e por isso multiplica $q$ pela **esquerda** (no notebook VI(c), o giroscópio media no corpo e multiplicava pela direita). Em cada pequeno trecho reto do caminho, $\boldsymbol\omega$ é constante, e o passo é exato: $q\leftarrow\mathrm{Exp}\big((\hat{\mathbf z}\times\Delta\mathbf c)/r\big)\,q$.

**O resultado a conferir.** Para um caminho fechado pequeno, que cerca uma área $A$ (no sentido anti-horário), a bola volta girada de

$$\lvert\theta\rvert\approx\frac A{r^2}$$

em torno da vertical, **no sentido contrário** ao do caminho (o sinal de menos abaixo), e **qualquer que seja a forma** do caminho. Com o sinal: $\theta\approx-A/r^2$. Para caminhos grandes, o eixo do giro final deixa de ser vertical.
""")

c.code(FERRAMENTAS + FUNCOES_Q + r'''
r = 1.0
Z = np.array([0, 0, 1.0])

def rola(pontos, q0=np.array([1.0, 0, 0, 0])):
    """Rola a bola pelos pontos (complexos) do caminho do centro; devolve as orientações."""
    qs = [q0]
    for p0, p1 in zip(pontos[:-1], pontos[1:]):
        d = p1 - p0
        qs.append(qmul(qexp(np.cross(Z, [d.real, d.imag, 0.0]) / r), qs[-1]))
    return np.array(qs)

# rolar em linha reta uma distância d gira a bola d/r em torno do eixo horizontal perpendicular
d = 0.7
q_reta = rola(np.linspace(0, d, 50) + 0j)[-1]
confere("rolar em reta: giro de d/r em torno de y (andando em +x)", qlog(q_reta), [0, d / r, 0], tol=1e-14)
# o ponto da bola que toca o chão (−r ẑ, no mundo) tem velocidade zero
v = np.array([0.3, -0.4, 0]); w = np.cross(Z, v) / r
confere("o ponto de contato não escorrega", v + np.cross(w, -r * Z), 0, tol=1e-16)
''')

c.md(r"""
## 1. O quadrado: um colchete de Lie

Rolar uma distância $a$ em $+x$, depois $+y$, depois $-x$, depois $-y$. São quatro rotações exatas: $\mathrm{Exp}(-\tfrac ar\hat{\mathbf x})\,\mathrm{Exp}(-\tfrac ar\hat{\mathbf y})\,\mathrm{Exp}(\tfrac ar\hat{\mathbf x})\,\mathrm{Exp}(\tfrac ar\hat{\mathbf y})$, um **comutador de grupo**. Para $a$ pequeno, ele vale $\mathrm{Exp}\big(\tfrac{a^2}{r^2}[\hat{\mathbf y},\hat{\mathbf x}]\big)+O(a^3)$, um giro de $a^2/r^2$ (a área do quadrado sobre $r^2$) em torno de $\pm\hat{\mathbf z}$.
""")

c.code(r'''
print("   a/r     giro em torno de z     área/r²      razão      inclinação do eixo")
razoes = []
for a in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    quadrado = np.array([0, a, a + 1j * a, 1j * a, 0])
    pts = np.concatenate([np.linspace(quadrado[k], quadrado[k + 1], 2) [:-1] for k in range(4)] + [[0j]])
    q = rola(pts)[-1]
    phi = qlog(q); giro = phi[2]; incl = np.degrees(np.arccos(abs(phi[2]) / np.linalg.norm(phi)))
    razoes.append(giro / (a**2 / r**2))
    print(f"{a:6.2f}      {giro:+.6e}        {a**2 / r**2:.6e}     {giro / (a**2 / r**2):.5f}      {incl:6.2f}°")
confere("para um quadrado pequeno (percorrido no sentido anti-horário), o giro em torno de z é −área/r²", razoes[0], -1, tol=1e-3)
''')

c.md(r"""
## 2. A forma não importa (para caminhos pequenos)

Quatro caminhos com a **mesma área** $A=0{,}004\,r^2$: um círculo, um quadrado, um triângulo equilátero e uma elipse alongada (5 por 1). Todos percorridos no sentido anti-horário, com 2000 pontos.
""")

c.code(r'''
A = 0.004 * r**2
s = np.linspace(0, 1, 2001)
def poligono(vertices, n=2000):
    v_ = np.array(vertices + [vertices[0]]); L_ = np.abs(np.diff(v_)); cum = np.concatenate([[0], np.cumsum(L_)]) / L_.sum()
    return np.interp(s, cum, v_.real) + 1j * np.interp(s, cum, v_.imag)
lado_tri = np.sqrt(4 * A / np.sqrt(3))
caminhos = {
    "círculo": np.sqrt(A / np.pi) * (np.exp(2j * np.pi * s) - 1),
    "quadrado": poligono([0, np.sqrt(A), np.sqrt(A) * (1 + 1j), 1j * np.sqrt(A)]),
    "triângulo": poligono([0, lado_tri, lado_tri * np.exp(1j * np.pi / 3)]),
    "elipse 5:1": np.sqrt(A / np.pi) * (np.sqrt(5) * (np.cos(2 * np.pi * s) - 1) + 1j * np.sin(2 * np.pi * s) / np.sqrt(5)),
}
giros = {}
for nome, cam in caminhos.items():
    area = 0.5 * np.sum((cam[:-1].conj() * cam[1:]).imag)       # fórmula do laço (shoelace), em complexos
    phi = qlog(rola(cam)[-1]); giros[nome] = phi[2]
    print(f"{nome:12s}: área {area:.5f};  giro em torno de z {phi[2]:.6f} rad  (área/r² = {A / r**2:.6f});  componentes horizontais {np.linalg.norm(phi[:2]):.1e}")
confere("os quatro caminhos dão o mesmo giro, −área/r², a menos de 1%", list(giros.values()), [-A / r**2] * 4, tol=0.01, relativo=True)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4))
for nome, cam in caminhos.items():
    axs[0].plot(cam.real, cam.imag, label=nome)
axs[0].set_aspect("equal"); axs[0].legend(fontsize=7); axs[0].set_title("quatro caminhos com a mesma área", fontsize=9)
# uma marca presa à bola (inicialmente apontando para +x, no equador), vista de cima, ao longo do quadrado
cam = caminhos["quadrado"] * 16                                 # um quadrado maior, para o giro aparecer
qs = rola(cam)
for k_ in range(0, len(cam), 100):
    m_ = qrot(qs[k_], [0.35, 0, 0])
    axs[1].add_patch(plt.Circle((cam[k_].real, cam[k_].imag), 0.08, fill=False, color="0.6", lw=0.6))
    axs[1].arrow(cam[k_].real, cam[k_].imag, m_[0], m_[1], head_width=0.03, color="C3" if k_ == 0 else ("C0" if k_ < len(cam) - 100 else "k"), lw=0.8)
m_fim = qrot(qs[-1], [0.35, 0, 0])
axs[1].arrow(cam[-1].real, cam[-1].imag, m_fim[0], m_fim[1], head_width=0.04, color="k", lw=1.5)
axs[1].plot(cam.real, cam.imag, "k:", lw=0.6); axs[1].set_aspect("equal")
axs[1].set_title(f"vista de cima: uma marca da bola (vermelha no início, preta no fim);\nquadrado de lado {abs(cam[500] - cam[0]):.2f} r", fontsize=8)
plt.tight_layout(); plt.show()
''')

c.md(r"""
## 3. Caminhos grandes

Para um círculo de raio $\rho$, a razão entre o giro e $A/r^2$ e a inclinação do eixo do giro final, à medida que $\rho$ cresce.
""")

c.code(r'''
rhos = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
dados = []
for rho in rhos:
    cam = rho * (np.exp(2j * np.pi * s) - 1)
    phi = qlog(rola(cam)[-1]); A_ = np.pi * rho**2
    dados.append((np.linalg.norm(phi) / (A_ / r**2), np.degrees(np.arccos(abs(phi[2]) / np.linalg.norm(phi)))))
dados = np.array(dados)
for rho, (raz, inc) in zip(rhos, dados):
    print(f"ρ = {rho:4.2f} r:  ângulo do giro / (A/r²) = {raz:.4f};  o eixo do giro está a {inc:5.2f}° da vertical")
confere("a lei da área vale no limite de caminhos pequenos", dados[0, 0], 1, tol=1e-3)
confere("e o eixo inclina quando o caminho cresce", dados[-1, 1] > 30, True)
''')

c.md(r"""
**Leitura.** Rolar em reta é só um giro horizontal de $d/r$, e o ponto de contato não escorrega. No quadrado, as quatro rotações exatas compõem um giro em torno da vertical que, para quadrados pequenos, é a área sobre $r^2$, com sinal trocado: o quadrado percorrido no sentido anti-horário deixa a bola girada no sentido horário (razão $-0{,}9999$ para $a=0{,}02\,r$). A correção cresce com o tamanho, e o eixo do giro final se inclina. Com a mesma área pequena, círculo, quadrado, triângulo e elipse dão o mesmo giro a menos de 1%: em primeira ordem, só a **área** importa, não a forma. Para caminhos grandes, a lei da área deixa de valer: com um círculo de raio igual ao da bola, o giro é 83% de $A/r^2$, e o eixo está a 45° da vertical.

É a mesma geometria do pêndulo de Foucault (notebook IV(c)) e da dupla cobertura: transportar algo ao longo de um laço fechado num espaço "curvo" (aqui, o grupo das rotações) e voltar diferente. Os físicos chamam isso de **fase geométrica**, ou holonomia. É também o princípio de robôs esféricos que se movem rolando, e de quebra-cabeças em que se gira uma esfera só rolando-a entre as mãos.

## Exercícios

1. **Girar a bola sem sair do lugar.** Encontre um caminho fechado que deixe a bola girada de exatamente 90° em torno da vertical. (Dica: repetir $n$ vezes um circuito pequeno multiplica o giro por $n$, aproximadamente; mas o eixo também precisa ficar vertical, e um laço em forma de oito, com as duas metades em sentidos opostos, ajuda a cancelar a inclinação.)
2. **Qualquer orientação.** Mostre que, rolando, dá para levar a bola de qualquer orientação a qualquer outra, voltando ao mesmo ponto da mesa. (É um teorema de controlabilidade: os movimentos em $x$ e $y$, mais o colchete deles, geram as três direções de $\mathfrak{so}(3)$.)
3. **Uma bola sobre outra.** Se a bola de raio $r$ rola sobre uma esfera fixa de raio $R$ (em vez de um plano), o giro em torno da vertical depois de um laço pequeno de área $A$ muda. Simule e descubra o fator.
""")

c.salva()
