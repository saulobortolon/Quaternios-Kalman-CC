from comum import *
from q_codigo import FUNCOES_Q

c = Caderno(arquivo("6"))
cabecalho(c, "Domínio VI: os quaternions",
          "A orientação no espaço num número de quatro partes: a primeira vez em que a própria rotação não comuta",
          dom="6", anterior=arquivo("5"))

c.md(r"""
**O domínio.** Os quaternions $q=w+x\mathbf i+y\mathbf j+z\mathbf k$, com $\mathbf i^2=\mathbf j^2=\mathbf k^2=\mathbf{ijk}=-1$, fazem pela rotação do espaço o que os complexos unitários fizeram pela do plano: o giro de $\theta$ em torno do eixo $\mathbf n$ é $q=(\cos\frac\theta2,\ \mathrm{sen}\frac\theta2\,\mathbf n)$, compor giros é multiplicar, desfazer é conjugar, e um vetor gira por $q\,\mathbf v\,q^*$. O grupo $SO(3)$ aparece como a esfera $S^3$ dos quaternions unitários, com $q$ e $-q$ representando a mesma rotação.

Três consequências atravessam o domínio:
- a **própria rotação não comuta**: girar um livro em torno de $x$ e depois de $y$ não é o mesmo que o contrário; o colchete de Lie de $\mathfrak{so}(3)$ é o produto vetorial, e ele aparece nos integradores, no erro de cone das centrais inerciais e na bola que rola;
- a **dupla cobertura** tem efeitos visíveis: uma volta de 360° não traz o quaternion de volta; um controlador que ignora isso gira o satélite 350° para corrigir 10°; uma câmera lenta vê a bola girar ao contrário;
- os **ângulos de Euler** têm singularidades (o *gimbal lock*), e os quaternions não: é por isso que a navegação, a animação e os jogos usam quaternions.

**Os exemplos físicos deste domínio**
- **Um satélite** que determina a própria orientação pelas estrelas, e nuvens de pontos 3D alinhadas pelo ICP, na etapa (b).
- **A porca de Dzhanibekov**, que capota sozinha no espaço, e o **erro de cone** das centrais inerciais, na etapa (c).
- **Pepé com um celular no capacete**: quatro filtros de atitude e o erro no corpo e no mundo, na etapa (d).
- **Cinco apêndices**: controle de atitude e *unwinding*, portas de um qubit, a bola que rola, realidade virtual e a bola de pingue-pongue com efeito.

A animação mostra a porca girando em torno do eixo intermediário, à esquerda (o momento angular, em vermelho, fica parado; a haste capota), e uma bola rolando num quadrado, vista de cima, à direita (ela volta ao ponto de partida girada).
""")

c.code(FERRAMENTAS + FUNCOES_Q + r'''
from matplotlib import animation
from IPython.display import HTML

# a porca: corpo livre com I = diag(1, 2, 3), girando perto do eixo intermediário
I = np.array([1.0, 2.0, 3.0]); dw = lambda w: np.cross(I * w, w) / I
def campo(s): return np.concatenate([0.5 * qmul(s[:4], np.concatenate([[0.0], s[4:]])), dw(s[4:])])
s = np.concatenate([[1.0, 0, 0, 0], [0.01, 2.0, 0.01]]); h = 0.01; Hs = [s]
for _ in range(2400):
    k1 = campo(s); k2 = campo(s + h / 2 * k1); k3 = campo(s + h / 2 * k2); k4 = campo(s + h * k3)
    s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4); Hs.append(s)
Hs = np.array(Hs)[::40]                                 # 61 quadros em 24 s
L = qmat(qnorm(Hs[0, :4])) @ (I * Hs[0, 4:]); L = L / np.linalg.norm(L)

# a bola: rola num quadrado de lado 0,9 r (vista de cima), marca no equador
lado = 0.9; cantos = np.array([0, lado, lado + 1j * lado, 1j * lado, 0])
cam = np.concatenate([np.linspace(cantos[k], cantos[k + 1], 16)[:-1] for k in range(4)] + [[0j]])
qb = [np.array([1.0, 0, 0, 0])]
for p0, p1 in zip(cam[:-1], cam[1:]):
    d = p1 - p0; qb.append(qmul(qexp(np.cross([0, 0, 1.0], [d.real, d.imag, 0.0])), qb[-1]))

fig = plt.figure(figsize=(7.6, 3.6), dpi=60)
a1 = fig.add_subplot(1, 2, 1, projection="3d"); a2 = fig.add_subplot(1, 2, 2)
a1.set_xlim(-1, 1); a1.set_ylim(-1, 1); a1.set_zlim(-1, 1); a1.set_box_aspect((1, 1, 1))
a1.set_xticks([]); a1.set_yticks([]); a1.set_zticks([]); a1.set_title("a porca de Dzhanibekov", fontsize=9)
a1.plot([0, L[0]], [0, L[1]], [0, L[2]], color="tab:red", lw=2)
haste, = a1.plot([], [], [], color="0.2", lw=4); barra, = a1.plot([], [], [], color="tab:blue", lw=4)
a2.set_xlim(-0.6, 1.5); a2.set_ylim(-0.6, 1.5); a2.set_aspect("equal"); a2.set_xticks([]); a2.set_yticks([]); a2.grid(False)
a2.set_title("a bola que rola (vista de cima)", fontsize=9)
a2.plot(cam.real, cam.imag, ":", color="0.5", lw=1)
bola = plt.Circle((0, 0), 0.3, fill=False, color="0.4"); a2.add_patch(bola)
marca, = a2.plot([], [], color="tab:red", lw=3); inicio = qrot(qb[0], [0.3, 0, 0])
a2.plot([0, inicio[0]], [0, inicio[1]], color="tab:red", lw=1, alpha=0.4)
plt.tight_layout()

def quadro(i):
    q = qnorm(Hs[i, :4])
    e2 = qrot(q, [0, 0.9, 0]); e1 = qrot(q, [0.5, 0, 0])
    haste.set_data_3d([-e2[0], e2[0]], [-e2[1], e2[1]], [-e2[2], e2[2]])
    barra.set_data_3d([e2[0] - e1[0], e2[0] + e1[0]], [e2[1] - e1[1], e2[1] + e1[1]], [e2[2] - e1[2], e2[2] + e1[2]])
    p = cam[i]; m = qrot(qb[i], [0.3, 0, 0])
    bola.center = (p.real, p.imag); marca.set_data([p.real, p.real + m[0]], [p.imag, p.imag + m[1]])
    return haste, barra, bola, marca

anim = animation.FuncAnimation(fig, quadro, frames=61, interval=150, blit=False)
plt.close(fig)
HTML(anim.to_jshtml())
''')

c.md(r"""
---
## O que este domínio estabelece para a série

| ideia | domínio V (dual-complexos) | domínio VI (quaternions) | o que muda depois |
|---|---|---|---|
| **o que se representa** | a pose no plano | a orientação no espaço, $(\cos\frac\theta2,\mathrm{sen}\frac\theta2\,\mathbf n)$ | a pose no espaço (VII) |
| **compor** | não comuta (por causa da translação) | não comuta **na própria rotação** | as duas não comutatividades juntas |
| **$\boxplus$** | corpo ≠ mundo, na translação | corpo ≠ mundo já na orientação: $\mathrm{Exp}(\delta)\,q=q\,\mathrm{Exp}(R^\top\delta)$ | $\mathrm{Ad}$ $6\times6$ |
| **$\mathrm{Exp}$** | um arco de círculo | a fórmula de Rodrigues; o ângulo por $\mathrm{atan2}$ | o parafuso de Chasles |
| **colchete** | $[X_1,X_2]$ mistura giro e translação | o produto vetorial: Magnus, cone, bola que rola | $\mathfrak{se}(3)$ |
| **dupla cobertura** | $q$ e $-q$ | $q$ e $-q$, com efeitos físicos: *unwinding*, serrilhado | o mesmo, com a parte dual |
| **sistemas** | Procrustes complexo | Wahba e Horn: um **autovetor** $4\times4$ | a pose de nuvens 3D, mão-olho |
| **filtro** | erro no corpo, no mundo, nas coordenadas | MEKF (corpo), IEKF (mundo), aditivo, Mahony | o ESKF de pose; o quadrirrotor |

---
## As quatro folhas deste domínio
""")

resumo = {
    "a": ("O produto de Hamilton, a identidade dos quatro quadrados e as matrizes de $SU(2)$, pelo `sympy`; o teste do livro (duas ordens, "
          "dois giros de 120° diferentes); a dupla cobertura, lendo laços de matrizes (360° termina em $-1$, 720° em $+1$); o *gimbal lock* "
          "(taxas de Euler de 70 rad/s para uma rotação de 0,6 rad/s, e um erro que salta sete ordens de grandeza); o ângulo por $\\mathrm{atan2}$, "
          "que o $\\arccos$ perde perto de zero; SLERP com velocidade constante, NLERP com velocidade desigual e Euler com um caminho 44% mais "
          "longo; rotações uniformes de Shoemake contra ângulos de Euler sorteados (rejeitados com valor-p de $10^{-91}$); 200 rotações exatas "
          "com quaternions inteiros; a interface de domínio em $SO(3)$."),
    "b": ("O problema de Wahba para um satélite: o método q de Davenport (um autovetor $4\\times4$) igual ao SVD, o TRIAD, e a covariância de "
          "Shuster conferida por Monte Carlo a menos de 3%, com o eixo da linha de visada das estrelas oito vezes pior que os outros; a orientação "
          "absoluta de Horn e o ICP em 3D, que acerta sempre com chutes de até 60° e erra quase sempre a partir de 120°; a média de rotações, "
          "em que somar componentes erra 36° e as médias de Markley e de Karcher erram 2,5°."),
    "c": ("Euler, RK4 nas componentes, Magnus 2 e Magnus 4 com e sem o colchete (ordens 1, 4, 2, 4 e 2); a porca de Dzhanibekov, que capota "
          "a cada 11 s com energia e momento angular no mundo conservados a $10^{-10}$; o erro de cone de uma central inercial, em que o algoritmo "
          "ingênuo deriva 960 °/h a 100 Hz e a correção de Miller, que é um colchete de Lie, reduz a deriva 50 vezes e sobe a ordem de 2 para 4."),
    "d": ("Pepé, com um celular no capacete, estima a orientação da cabeça com giroscópio, acelerômetro e magnetômetro. Partindo perto da verdade, "
          "o EKF aditivo, o MEKF (erro no corpo) e o IEKF (erro no mundo) empatam (0,78°, NEES 2,75), e o filtro de Mahony erra 2,1°. Ligando o "
          "celular sem saber o rumo, o IEKF converge mais rápido (9,6° depois de 1 s, contra 20° e 24°), mas nenhum dos três EKFs é consistente "
          "no começo: a lição é inicializar pelo problema de Wahba."),
}
linhas = []
for e, slug, titulo in ETAPAS:
    arq = arquivo("6", e)
    linhas.append(f"### ({e}) {titulo[0].upper() + titulo[1:]}\n{link(arq)}\n\n{resumo[e]}\n")
c.md("\n".join(linhas))

apendices = [
    ("unwinding", "o satélite que dá a volta à toa",
     "Um PD de atitude que ignora o sinal de $q$ gira o satélite 350° para corrigir 10°; escolher o sinal resolve, mas faz o torque chavear "
     "dezenas de vezes perto de 180° com ruído; a histerese elimina o chaveamento."),
    ("qubit", "as portas de um qubit são quaternions",
     "O dicionário com as matrizes de Pauli, as portas X, Y, Z, H, S e T como rotações, a esfera de Bloch, a decomposição ZYZ, e a "
     "aproximação de rotações quaisquer por palavras em H e T, com o erro caindo como $N^{-1/3}$."),
    ("bola", "a bola que rola",
     "Rolar uma bola num caminho fechado pequeno a deixa girada de $-A/r^2$ em torno da vertical, qualquer que seja a forma do caminho: o "
     "colchete de Lie que se sente com a mão. Em caminhos grandes, o eixo do giro se inclina."),
    ("vr", "óculos de realidade virtual",
     "Prever a orientação da cabeça 20 ms à frente: sem previsão, 1,8° de erro; com a velocidade do giroscópio e a $\\mathrm{Exp}$, 0,19°; com um "
     "filtro de Kalman de aceleração, 0,064°."),
    ("pingpong", "a bola de pingue-pongue com efeito",
     "A força de Magnus em três saques; medir a rotação pela marca da bola, e o serrilhado em $SO(3)$: abaixo de dois quadros por rotação, "
     "um *topspin* de 50 rps aparece como corte de 25 rps, e o quique previsto erra 70 cm."),
]
linhas = ["---\n## Apêndices\n\nAplicações que não cabiam nas quatro etapas, cada uma num notebook independente, com as ferramentas do domínio (quaternions, $\\mathrm{Exp}$ e $\\mathrm{Log}$, o colchete, filtros de atitude) e as próprias conferências.\n"]
for slug, titulo, texto in apendices:
    linhas.append(f"- **{titulo[0].upper() + titulo[1:]}.** {link(apendice('6', slug))}. {texto}")
c.md("\n".join(linhas))

K5 = kalman("kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", "notebook 5")
K6 = kalman("kalman_quadrotor_quaternions.ipynb", "notebook 6")
c.md(f"""
---
## Para quem vem da série Kalman

Pontes com a série Kalman (pasta `kalman/` do mesmo repositório):
- o MEKF e o IEKF da etapa (d) são o ESKF e o IEKF do {K5}, agora em $SO(3)$, com o viés do giroscópio no estado;
- a etapa (d) é a parte de atitude do quadrirrotor do {K6}: o mesmo estado de erro de 3 componentes, o mesmo $\\boxplus$ multiplicativo;
- a tabela de integradores da etapa (c) é o RK4 contra o RKMK do {K6}, com o colchete que separa a ordem 2 da ordem 4.

**Domínio anterior:** {link(arquivo("5"), "os números dual-complexos")}. **Próximo domínio:** os dual quaternions, que juntam a rotação do espaço e a translação num número só, a pose de um drone.
""")

c.salva()
