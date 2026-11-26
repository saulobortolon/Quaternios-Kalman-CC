# Números, movimento e Kalman: de ℝ aos quaternions tridentes

Série de notebooks para alunos de Ciência da Computação, criada por Saulo Bortolon em parceria com Claude.

A mesma história é contada em oito domínios de números, cada um mais rico que o anterior. Em cada domínio, quatro etapas:

| etapa | conteúdo |
|---|---|
| (a) álgebra | operações, propriedades que valem e que se perdem, implementação em Python (com `sympy`) |
| (b) sistemas | equações lineares e não lineares no domínio |
| (c) EDOs | cinemática e dinâmica ($F=ma$), solução simbólica (`sympy`) e numérica (Runge–Kutta, RKMK) |
| (d) perseguição | um perseguidor estima a posição de um alvo com um filtro de Kalman, a partir de sensores com ruído |

| | domínio | estado |
|---|---|---|
| I | números reais ℝ | **pronto** |
| II | vetores de ℝ² | em preparação |
| III | números duais | em preparação |
| IV | números complexos ℂ | em preparação |
| V | números dual-complexos | em preparação |
| VI | quaternions | em preparação |
| VII | dual quaternions | em preparação |
| VIII | quaternions tridentes | em preparação |

## Por onde começar

Abra [`notebooks/X0_raiz.ipynb`](notebooks/X0_raiz.ipynb), ou diretamente no Colab:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X0_raiz.ipynb)

Cada notebook é autossuficiente e roda sozinho no Colab, sem instalar nada.

## Estrutura (dentro de `dominios/`)

```
notebooks/     os notebooks, já executados (figuras e resultados incluídos)
construtor/    o código que GERA os notebooks
  comum.py       funções comuns, lista de domínios e etapas, células de ferramentas
  x0.py, x1.py   a raiz e o nó de cada domínio
  x1a.py ...     as folhas (domínio 1, etapas a–d)
  build.py       gera e executa tudo:  python build.py   (ou  python build.py x1c  para um só)
```

Os notebooks **não são editados à mão**. Cada correção é feita no construtor, e a série é regenerada e reexecutada. Toda afirmação numérica importante é conferida por uma célula que imprime `OK`.

## Requisitos para regenerar

Python 3.10+, `numpy`, `scipy`, `matplotlib`, `sympy`, `pandas`, `nbformat`, `nbconvert`, `jupyter`.

## Série irmã

*O Filtro de Kalman para Cientistas da Computação* (Pepé Le Gambá, IMM, notas de Im, quadrotor com quaternions tridentes), no mesmo repositório: [`kalman/`](../kalman/). O guia dela tem uma tabela de pontes entre as duas séries.
