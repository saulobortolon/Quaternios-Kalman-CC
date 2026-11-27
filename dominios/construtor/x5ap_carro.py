from comum import *
from dc_codigo import CLASSE_DC

c = Caderno(apendice("5", "carro"))
cabecalho(c, "Domínio V, apêndice: odometria de um carro",
          "O modelo de bicicleta, o arco exato da Exp, e a calibração das rodas por dois circuitos fechados",
          dom="5", etapa=None, anterior=arquivo("5"))

c.md(r"""
**O modelo de bicicleta.** Para a cinemática em baixa velocidade (sem derrapagem), um carro se comporta como uma bicicleta: junta-se as duas rodas de cada eixo numa só, no meio do eixo. Com o ponto de referência no meio do eixo traseiro, a velocidade de avanço $v$, o ângulo de esterçamento $\delta$ das rodas dianteiras e a distância entre eixos $L$, a velocidade no corpo é

$$\omega=\frac{v\tan\delta}{L},\qquad \mathbf v_{\text{corpo}}=v\ (\text{para a frente, sem componente lateral}).$$

Com $v$ e $\delta$ constantes durante um intervalo $\Delta t$, o carro percorre **exatamente** um arco de círculo, e a pose avança por $q\leftarrow q\,\mathrm{Exp}(\omega\Delta t,\ v\Delta t)$, sem erro de integração nenhum (notebook V(a), seção 5). O passo de Euler, $x\mathrel{+}=v\cos\theta\,\Delta t$, $y\mathrel{+}=v\,\mathrm{sen}\,\theta\,\Delta t$, $\theta\mathrel{+}=\omega\Delta t$, troca o arco por um segmento de reta.

**Odometria pelas rodas.** Na prática, $v$ e $\omega$ vêm de **encoders** nas rodas traseiras: contando os pulsos de cada roda, sabe-se quanto ela girou, e, com o raio $r$, quanto andou. Com as distâncias $s_e$ e $s_d$ das rodas esquerda e direita e a bitola $B$ (a distância entre elas): $v\Delta t=(s_e+s_d)/2$ e $\omega\Delta t=(s_d-s_e)/B$. Se os raios ou a bitola estiverem errados, mesmo que por meio por cento, o erro vira um **desvio sistemático** que se acumula sem parar.

**O que este notebook mostra.**
1. O arco da $\mathrm{Exp}$ é a trajetória exata do modelo, e o seu ponto fixo é o centro de curva de **Ackermann**, sobre a reta do eixo traseiro.
2. Euler tem erro de ordem 1, o ponto médio de ordem 2, a $\mathrm{Exp}$ é exata.
3. A calibração: com dois circuitos fechados, um em cada sentido (a ideia do *UMBmark* de Borenstein e Feng), um Gauss–Newton no grupo estima a diferença entre os raios e a bitola, **mas não a escala**: um circuito fechado não sabe o próprio tamanho. Com uma reta de comprimento conhecido, os três parâmetros saem certos.
""")

c.code(FERRAMENTAS + r'''
rng = np.random.default_rng(55)
''' + CLASSE_DC + r'''
L, BITOLA = 2.7, 1.6                       # distância entre eixos e bitola (m)

def velocidade_corpo(v, delta):
    return v * np.tan(delta) / L, v        # (ω, v)
''')

c.md(r"""
## 1. Ackermann: o centro de curva é o ponto fixo da Exp

Com esterçamento constante, o carro gira em torno de um ponto fixo do mundo, a uma distância $R=L/\tan\delta$ do meio do eixo traseiro, **sobre a reta do eixo traseiro** (para a esquerda, se $\delta>0$). É o princípio de Ackermann: todas as rodas giram em torno desse mesmo ponto, e nenhuma escorrega de lado. Duas conferências: o ponto fixo do passo $\mathrm{Exp}(\omega\Delta t,v\Delta t)$, levado ao mundo, é esse centro; e a roda dianteira, no ponto $L$ do corpo, se move exatamente na direção para onde está esterçada.
""")

c.code(r'''
v, delta, dt = 5.0, np.radians(12), 0.1
w, vv = velocidade_corpo(v, delta)
R = L / np.tan(delta)
q = pose(np.radians(30), 3 + 2j)                      # uma pose qualquer
passo = Exp(w * dt, vv * dt)
centro = aplica(q, 1j * R)                           # R para a esquerda, sobre a reta do eixo traseiro
# ponto fixo do deslocamento no mundo, q·passo·q⁻¹
dq = q * passo * q.conj()
fixo = translacao(dq) / (1 - np.exp(1j * angulo(dq)))
print(f"raio de curva R = L/tan δ = {R:.4f} m;  centro esperado {centro:.5f};  ponto fixo do passo {fixo:.5f}")
confere("o ponto fixo da Exp é o centro de Ackermann", fixo, centro, tol=1e-12)

# a roda dianteira: velocidade do ponto L do corpo = v + ω·i·L (no corpo); a direção dela tem de ser δ
vel_dianteira = vv + w * 1j * L
confere("a roda dianteira anda para onde está esterçada (sem derrapar)", np.angle(vel_dianteira), delta, tol=1e-14)

# 200 passos: todos os pontos à distância R do centro
cam = [q]
for _ in range(200):
    cam.append(cam[-1] * passo)
confere("200 passos depois, o carro continua no círculo de raio R", max(abs(abs(translacao(p) - centro) - R) for p in cam), 0, tol=1e-11)
''')

c.md(r"""
## 2. Euler, ponto médio e Exp

Meia volta na rotatória ($v=5$ m/s, $\delta=12°$): o carro deveria chegar ao ponto diametralmente oposto, $2Ri$. A tabela mostra o erro de posição na chegada para vários $\Delta t$. (Por que meia volta e não uma volta inteira? Com $\omega$ constante, o Euler percorre um polígono regular, que **fecha** depois de uma volta: o erro na chegada seria zero, por simetria, e esconderia o erro do caminho.)
""")

c.code(r'''
def integra(metodo, dt, v=5.0, delta=np.radians(12)):
    w, vv = velocidade_corpo(v, delta)
    T = np.pi / w; n = int(round(T / dt)); dt = T / n              # meia volta
    x, y, th = 0.0, 0.0, 0.0
    q = pose(0.0, 0j)
    for _ in range(n):
        if metodo == "Euler":
            x += vv * np.cos(th) * dt; y += vv * np.sin(th) * dt; th += w * dt
        elif metodo == "ponto médio":
            tm = th + w * dt / 2; x += vv * np.cos(tm) * dt; y += vv * np.sin(tm) * dt; th += w * dt
        else:
            q = q * Exp(w * dt, vv * dt)
    alvo = 2j * L / np.tan(delta)
    return abs(complex(x, y) - alvo) if metodo != "Exp" else abs(translacao(q) - alvo)

passos = [0.4, 0.2, 0.1, 0.05]
print("   Δt      Euler        ponto médio     Exp")
tab = {m: [integra(m, h) for h in passos] for m in ["Euler", "ponto médio", "Exp"]}
for i, h in enumerate(passos):
    print(f"{h:6.2f}   {tab['Euler'][i]:9.2e}     {tab['ponto médio'][i]:9.2e}     {tab['Exp'][i]:9.2e}")
ordem = lambda e: np.polyfit(np.log(passos), np.log(e), 1)[0]
print(f"ordem observada: Euler {ordem(tab['Euler']):.2f}, ponto médio {ordem(tab['ponto médio']):.2f}")
confere("Euler tem ordem 1", ordem(tab["Euler"]), 1, tol=0.05)
confere("ponto médio tem ordem 2", ordem(tab["ponto médio"]), 2, tol=0.05)
confere("a Exp é exata (erro de arredondamento)", max(tab["Exp"]), 0, tol=1e-10)
''')

c.md(r"""
## 3. Calibrar as rodas com dois circuitos

O carro tem raios de roda nominais de 30 cm e bitola nominal de 1,6 m. Na verdade, a roda esquerda tem 30,15 cm (pneu mais cheio), a direita 29,94 cm, e a bitola efetiva é 1,62 m (o ponto de contato do pneu não é onde a ficha técnica diz). O carro percorre um circuito retangular de 20 m × 10 m, voltando ao ponto de partida, uma vez no sentido anti-horário e uma vez no horário. Os encoders têm 2048 pulsos por volta (a medida é arredondada para o pulso inteiro).

Um erro nos raios **iguais** muda a escala de tudo; uma **diferença** entre os raios faz o carro curvar quando acha que vai reto; um erro na bitola muda o quanto ele acha que girou nas curvas. Num circuito só, a diferença de raios e a bitola se confundem; nos dois sentidos, os efeitos da bitola se invertem e os da diferença de raios não, e eles se separam (é o argumento do UMBmark).

O resíduo de cada circuito é $\mathrm{Log}(q_{\text{odo}}^{-1}\,q_{\text{verdade}})$ na chegada, onde $q_{\text{verdade}}$ é a pose de chegada verdadeira (praticamente a de partida: o carro volta ao mesmo lugar, conferido por um marco no chão). Os parâmetros são três: os dois raios e a bitola. Vale a pena prever, antes de rodar: dois circuitos dão seis números; três parâmetros; deve dar certo?
""")

c.code(r'''
PULSOS = 2048
r_ver = np.array([0.3015, 0.2994]); B_ver = 1.62           # verdade: raios esquerdo, direito; bitola
r_nom = np.array([0.30, 0.30]);     B_nom = 1.60           # o que o carro acha

def circuito(sentido, dt=0.05, v=3.0):
    """Os controles (v, δ) de um retângulo 20 × 10 com curvas de raio 4 m, no sentido +1 (anti-horário) ou −1."""
    R = 4.0; delta = sentido * np.arctan(L / R)
    w = v / R
    trechos = []                                   # (v, δ, Δt): o Δt de cada trecho é ajustado para o trecho caber num número inteiro de passos
    for reto in [20 - 2 * R, 10 - 2 * R] * 2:
        n = int(round(reto / v / dt)); trechos += [(v, 0.0, reto / v / n)] * n
        n = int(round((np.pi / 2) / w / dt)); trechos += [(v, delta, (np.pi / 2) / w / n)] * n
    return trechos

def pulsos_das_rodas(trechos):
    """O carro anda de verdade; cada roda traseira gira s/r_ver; o encoder conta pulsos inteiros (acumulados)."""
    q = pose(0.0, 0j); ang = np.zeros(2); contagens = [np.zeros(2, dtype=int)]
    for v, delta, dt in trechos:
        w, vv = velocidade_corpo(v, delta)
        q = q * Exp(w * dt, vv * dt)
        s = np.array([vv * dt - w * dt * B_ver / 2, vv * dt + w * dt * B_ver / 2])   # distâncias das rodas e, d
        ang += s / r_ver
        contagens.append(np.floor(ang / (2 * np.pi) * PULSOS).astype(int))
    return q, np.array(contagens)

def odometria(contagens, r, B):
    q = pose(0.0, 0j)
    for d in np.diff(contagens, axis=0):
        s = d / PULSOS * 2 * np.pi * r
        q = q * Exp((s[1] - s[0]) / B, (s[0] + s[1]) / 2 + 0j)
    return q

dados = {}
for sentido in [+1, -1]:
    q_fim, cont = pulsos_das_rodas(circuito(sentido))
    dados[sentido] = cont
    print(f"sentido {'anti-horário' if sentido > 0 else 'horário     '}: a verdade termina a {abs(translacao(q_fim)):.1e} m da partida (o circuito fecha)")
    q_odo = odometria(cont, r_nom, B_nom)
    print(f"   a odometria com os parâmetros nominais acha que terminou a {abs(translacao(q_odo) - translacao(q_fim)):.2f} m da verdade, "
          f"com {np.degrees(wrap(angulo(q_odo) - angulo(q_fim))):+.1f}° de erro de rumo")
    dados[sentido] = (cont, q_fim)
''')

c.code(r'''
def Log3(q):
    w, v = Log(q); return np.array([w, v.real, v.imag])

def residuo(p, experimentos):
    r = p[:2]; B = p[2]
    return np.concatenate([Log3(odometria(cont, r, B).conj() * q_fim) for cont, q_fim in experimentos])

def calibra(experimentos, n_iter=6):
    p = np.array([*r_nom, B_nom])
    print(" it    raio esq. (cm)   raio dir. (cm)   bitola (m)    |resíduo|")
    for it in range(n_iter):
        r0 = residuo(p, experimentos)
        print(f"{it:3d}    {p[0] * 100:9.4f}        {p[1] * 100:9.4f}       {p[2]:8.5f}     {np.linalg.norm(r0):.2e}")
        J = np.zeros((len(r0), 3)); h = 1e-7
        for k in range(3):
            d = np.zeros(3); d[k] = h
            J[:, k] = (residuo(p + d, experimentos) - residuo(p - d, experimentos)) / (2 * h)
        p = p - np.linalg.lstsq(J, r0, rcond=None)[0]
    print(f"verdade:   {r_ver[0] * 100:9.4f}        {r_ver[1] * 100:9.4f}       {B_ver:8.5f}")
    print(f"número de condição do jacobiano: {np.linalg.cond(J):.3g}")
    return p

p2 = calibra([dados[+1], dados[-1]])
razoes = np.array([p2[0] / r_ver[0], p2[1] / r_ver[1], p2[2] / B_ver])
print(f"\nestimado / verdadeiro, para os três parâmetros: {razoes}")
confere("com dois circuitos fechados, os três parâmetros saem errados pelo MESMO fator", razoes.max() - razoes.min(), 0, tol=5e-4)
p_ver = np.array([*r_ver, B_ver])
for k in [0.5, 1.0, 2.0]:
    print(f"resíduo com os parâmetros verdadeiros multiplicados por {k}:  {np.linalg.norm(residuo(k * p_ver, [dados[+1], dados[-1]])):.2e}")
''')

c.md(r"""
**O que deu errado.** O Gauss–Newton fechou os circuitos em duas iterações (o resíduo caiu ao nível da quantização dos encoders), mas depois **saiu passeando**: os três parâmetros encolhem juntos, iteração após iteração, sempre na mesma proporção entre si, e chegam a raios de décimos de milímetro, sem que o resíduo reaja. O número de condição do jacobiano é da ordem de $10^7$. E os parâmetros verdadeiros multiplicados por $0{,}5$ ou por $2$ dão resíduos tão pequenos quanto os verdadeiros. O motivo é geométrico: se os dois raios e a bitola forem multiplicados pelo mesmo fator $k$, a odometria calcula um circuito $k$ vezes maior, com os mesmos ângulos, e um circuito que volta ao ponto de partida continua voltando ao ponto de partida. Um circuito fechado não tem **escala**: a escala é uma direção do espaço dos parâmetros em que o resíduo não muda (só a quantização dos encoders quebra a simetria, de leve, e é isso que deixa o número de condição finito; num problema sem quantização, o jacobiano seria exatamente singular). É o mesmo fenômeno da primeira pose livre no SLAM por grafo, e da escala do SLAM com uma câmera só.

A correção é uma medida com **escala**: o carro anda em linha reta até um marco a 20 m, conhecido com precisão.
""")

c.code(r'''
# o terceiro experimento: 20 m em linha reta até um marco
reta = [(3.0, 0.0, 20 / 3.0 / 400)] * 400
q_fim, cont = pulsos_das_rodas(reta)
dados[0] = (cont, q_fim)
print(f"a verdade anda {abs(translacao(q_fim)):.3f} m; a odometria nominal acha que andou {abs(translacao(odometria(cont, r_nom, B_nom))):.3f} m\n")
p = calibra([dados[+1], dados[-1], dados[0]])
confere("agora os raios estimados batem com os verdadeiros (a menos de 0,1 mm)", p[:2], r_ver, tol=1e-4)
confere("e a bitola estimada bate com a verdadeira (a menos de 1 mm)", p[2], B_ver, tol=1e-3)
''')

c.code(r'''
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, sentido in zip(axs, [+1, -1]):
    cont, q_fim = dados[sentido]
    for nome, r, B, est in [("nominal", r_nom, B_nom, "r-"), ("calibrado", p[:2], p[2], "b--")]:
        q = pose(0.0, 0j); pts = [0j]
        for d in np.diff(cont, axis=0):
            s = d / PULSOS * 2 * np.pi * r
            q = q * Exp((s[1] - s[0]) / B, (s[0] + s[1]) / 2 + 0j); pts.append(translacao(q))
        pts = np.array(pts); ax.plot(pts.real, pts.imag, est, lw=1.3, label=f"odometria {nome}")
    ax.plot(0, 0, "ko", ms=6, label="partida e chegada")
    ax.set_aspect("equal"); ax.legend(fontsize=7)
    ax.set_title(f"circuito {'anti-horário' if sentido > 0 else 'horário'}", fontsize=10)
plt.tight_layout(); plt.show()
''')

c.md(r"""
**Leitura.** O passo da $\mathrm{Exp}$ tem por ponto fixo o centro de Ackermann, a roda dianteira anda na direção em que aponta, e, depois de 200 passos, o carro continua no círculo com erro de arredondamento: o arco é a solução exata do modelo de bicicleta. Euler e ponto médio mostram as ordens 1 e 2 esperadas.

Na calibração, diferenças de poucas décimas de milímetro nos raios e de 2 cm na bitola bastam para a odometria nominal terminar os circuitos com erros de 1,3 m e 2,2 m na posição e de 9° e 17° no rumo. Com os dois circuitos, o Gauss–Newton no grupo fecha os circuitos, mas deixa a escala solta: os três parâmetros derivam juntos, na proporção certa e no tamanho errado. Com a reta de 20 m, a escala aparece, o número de condição cai de cerca de $10^7$ para cerca de $10^2$, e os três parâmetros saem certos, até onde a quantização dos encoders permite. Na figura, a odometria calibrada (tracejada) fecha os dois circuitos.

## Exercícios

1. **Um circuito e a reta.** Calibre usando só o circuito anti-horário e a reta. O jacobiano fica $6\times3$; calcule o número de condição e compare com o dos três experimentos. Que combinação de parâmetros fica mal determinada agora?
2. **A banana do carro.** Acrescente ruído aos pulsos (por exemplo, roda que escorrega um pouco: some $\pm1$ pulso aleatório a cada passo) e repita a odometria mil vezes. Desenhe a nuvem das poses finais: é uma banana (notebook V(c), seção 4)?
3. **Derrapagem.** Em alta velocidade, o modelo de bicicleta cinemático falha: os pneus escorregam de lado, e aparece uma velocidade lateral $v_y\neq0$ no corpo. A $\mathrm{Exp}$ continua exata para $(\omega,v_x+iv_y)$ constantes? E o ponto fixo, onde fica agora?
""")

c.salva()
