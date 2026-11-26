from comum import *

c = Caderno(arquivo())
c.md(f"""
{selo(c.arq)}

# Números, movimento e Kalman
## De ℝ aos quaternions tridentes, em oito domínios e quatro etapas

**O que é esta série.** Uma sequência de notebooks em que a **mesma história** é contada oito vezes, cada vez num "domínio" de números mais rico: dos reais aos quaternions tridentes. Em cada domínio, quatro etapas:

- **(a) álgebra:** as operações, as propriedades que valem e as que se perdem, e a implementação em Python (inclusive com `sympy`);
- **(b) sistemas de equações** lineares e não lineares naquele domínio;
- **(c) cinemática e dinâmica:** as equações diferenciais do movimento, resolvidas simbolicamente pelo `sympy` e numericamente por Runge–Kutta (e, quando há rotações, por RKMK, o Runge–Kutta de Munthe-Kaas);
- **(d) perseguição:** um perseguidor estima, com um filtro de Kalman, onde está um alvo que só observa por sensores com ruído.

**Por que assim.** Quaternions, dual quaternions e quaternions tridentes assustam quando aparecem de uma vez. Aqui eles chegam aos poucos: cada domínio acrescenta **uma** ideia ao anterior (uma segunda direção, um infinitesimal, uma rotação, uma translação, uma velocidade), e as quatro etapas mostram o que essa ideia muda na conta, na equação, na simulação e no filtro.

**Para quem.** Alunos de Ciência da Computação. Supõe-se Python, NumPy, álgebra linear e probabilidade básicas. Não se supõe Física além de $F=ma$, nem Engenharia de Controle. Quem fez a série *O Filtro de Kalman para Cientistas da Computação* reconhecerá o Pepé e a Penélope, mas ela não é pré-requisito.
""")

c.code(r'''
# nada a executar aqui além da verificação no final; esta célula só confirma o ambiente
import sys, numpy, matplotlib
print("Python", sys.version.split()[0], "| NumPy", numpy.__version__, "| Matplotlib", matplotlib.__version__)
try:
    import sympy; print("SymPy", sympy.__version__)
except ImportError:
    print("SymPy não encontrado: no Colab ele já vem instalado; localmente, use  pip install sympy")
''')

# matriz
cab = "| domínio | nó do domínio | " + " | ".join(f"({e}) {t.split(':')[0]}" for e, _, t in ETAPAS) + " |"
sep = "|---|---|" + "---|" * len(ETAPAS)
linhas = [cab, sep]
for cod, nome, numero, grupo, ex, pers in DOMINIOS:
    romano = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"][int(cod) - 1]
    if cod in PRONTOS:
        no = f"[{arquivo(cod)}]({arquivo(cod)})"
        cel = [f"[{e}]({arquivo(cod, e)}) · [Colab]({colab(arquivo(cod, e))})" for e, _, _ in ETAPAS]
    else:
        no = "*em preparação*"; cel = ["—"] * len(ETAPAS)
    linhas.append(f"| **{romano}** · {numero} | {no} | " + " | ".join(cel) + " |")
c.md("""
---
## 1. O mapa: oito domínios × quatro etapas

A série é uma **árvore**: esta raiz (nível 0), um nó por domínio (nível I, que apresenta o domínio e os exemplos) e quatro folhas por domínio (nível II, as etapas a–d). Na prática, as folhas formam uma **matriz**, e ela pode ser lida de dois jeitos:
- **por linha** (um domínio inteiro, de (a) a (d)): é a ordem natural de uma primeira leitura;
- **por coluna** (uma etapa através dos domínios): por exemplo, a coluna (c) mostra o Runge–Kutta virando RKMK, e a coluna (d) mostra o filtro de Kalman virando o ESKF.

""" + "\n".join(linhas))

esc = ["| | números | grupo de movimentos | exemplo físico (b, c) | perseguição (d) |", "|---|---|---|---|---|"]
for cod, nome, numero, grupo, ex, pers in DOMINIOS:
    romano = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"][int(cod) - 1]
    esc.append(f"| **{romano}** | {numero} | {grupo} | {ex} | {pers} |")
c.md("""
---
## 2. A escada de domínios

""" + "\n".join(esc) + r"""

**O que cada degrau acrescenta, e o que perde.**

| domínio | estrutura algébrica | acrescenta | perde em relação ao anterior |
|---|---|---|---|
| I · ℝ | corpo ordenado | — | — |
| II · ℝ² | espaço vetorial | uma segunda direção | o produto entre elementos e a ordem |
| III · duais $a+\varepsilon b$, $\varepsilon^2=0$ | anel comutativo | um infinitesimal: giro e deslizamento no mesmo eixo; derivação automática | inversos ($\varepsilon$ não tem inverso) |
| IV · ℂ | corpo | a rotação como multiplicação | a ordem |
| V · dual-complexos | anel não comutativo | posição **e** atitude no plano | a comutatividade |
| VI · ℍ | anel de divisão | rotações no espaço | (volta a ter inversos, mantém a não comutatividade) |
| VII · dual quaternions | anel não comutativo | pose no espaço (parafuso) | os inversos |
| VIII · quaternions tridentes | anel não comutativo | pose **e velocidade** no espaço | — |

O degrau VIII é o destino: um quaternion tridente é um quaternion cujos coeficientes são "números tridentes" $a+\varepsilon_1b+\varepsilon_2c$, com $\varepsilon_1^2=\varepsilon_2^2=\varepsilon_1\varepsilon_2=0$. Para quem chegou até ali pela escada, ele é um dual quaternion com um infinitesimal a mais.
""")

c.md(r"""
---
## 3. A interface que atravessa a série

Em todos os domínios, o estado de um objeto em movimento vive num conjunto com seis operações: compor dois movimentos, inverter um movimento, a identidade, $\mathrm{Exp}$ (o movimento produzido por uma velocidade constante), $\mathrm{Log}$ (a velocidade que produz um movimento), e o par $\boxplus$, $\boxminus$ (somar e medir incrementos pequenos). A etapa (a) de cada domínio implementa a interface e roda **os mesmos testes**.

| domínio | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$x\boxplus\delta$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;integrador da etapa (c)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;filtro da etapa (d)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|---|---|---|---|
| ℝ, ℝ² | $x+\delta$ | RK4 | KF, EKF, IEKF, filtro em grade |
| duais | $(\theta+\delta_\theta)+\varepsilon(d+\delta_d)$ | RK4 | EKF com jacobianos por números duais |
| ℂ | $z+\delta$ ou $z\,e^{i\delta}$ | RK4 = RKMK (comutativo) | EKF |
| dual-complexos | $\hat z\,\mathrm{Exp}(\delta)$ | RKMK | ESKF |
| quaternions | $q\otimes\mathrm{Exp}(\delta)$ | RKMK | ESKF |
| dual quaternions | $\hat q\otimes\mathrm{Exp}(\delta)$ | RKMK | ESKF |
| tridentes | $\tilde q\otimes\mathrm{Exp}(\delta)$ | RKMK | ESKF tridente |
""")

c.md(f"""
---
## 4. Como usar

**No Colab, sem instalar nada.** Cada notebook tem, no topo, um botão "Abrir no Colab", e a matriz da seção 1 tem um link "Colab" em cada célula. Os notebooks são **autossuficientes**: cada um repete, numa célula inicial de ferramentas, o pouco que precisa das etapas anteriores, e roda sozinho.

**O repositório.** Todos os arquivos estão em [github.com/{REPO}](https://github.com/{REPO}):
- `dominios/notebooks/`: os notebooks desta série, já executados (com figuras e resultados);
- `dominios/construtor/`: o código Python que **gera** os notebooks. Os notebooks não são editados à mão: cada correção é feita no construtor, e a série inteira é regenerada e reexecutada.

**Convenções**
- Toda afirmação numérica importante é conferida por uma célula que imprime `OK` (ou interrompe o notebook com uma mensagem clara).
- As simulações de Monte Carlo têm um parâmetro `N_MC` no topo do notebook: diminua-o para rodar mais rápido em aula.
- As leituras dos resultados ("**Leitura.**") foram escritas a partir dos números obtidos, e dizem o que os números dizem, inclusive quando contrariam a expectativa.

**A série irmã.** *O Filtro de Kalman para Cientistas da Computação* (Pepé Le Gambá, IMM, notas de Im, quadrotor) está no mesmo repositório, na pasta `kalman/`. Comece pelo guia: {kalman(GUIA_KALMAN, "guia da série Kalman")}. As duas séries se cruzam: cada nó de domínio desta série tem uma seção "Para quem vem da série Kalman", e o guia da série Kalman tem uma tabela de pontes para cá.
""")

c.md(r"""
---
## 5. Verificação: quais notebooks estão nesta pasta?
""")

lista = [(arquivo(),)] + [(arquivo(d[0]),) for d in DOMINIOS if d[0] in PRONTOS] + \
        [(arquivo(d[0], e[0]),) for d in DOMINIOS if d[0] in PRONTOS for e in ETAPAS]
c.code("import os\nfrom IPython.display import Markdown, display\nSERIE = " + repr([x[0] for x in lista]) + r'''
linhas = ["| notebook | nesta pasta? |", "|---|---|"]
for arq in SERIE:
    linhas.append(f"| {arq} | {'sim' if os.path.exists(arq) else '**não encontrado**'} |")
display(Markdown("\n".join(linhas)))
print(f"{sum(os.path.exists(a) for a in SERIE)} de {len(SERIE)} notebooks encontrados em {os.getcwd()}")
''')

c.salva()
