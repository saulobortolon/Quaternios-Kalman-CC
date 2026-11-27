# Quaternios-Kalman-CC

Duas séries de notebooks para alunos de Ciência da Computação, criadas por Saulo Bortolon em parceria com Claude. Todos rodam no Google Colab sem instalar nada.

## As duas séries

**[O Filtro de Kalman para Cientistas da Computação](kalman/)**: do Kalman como aprendizado online (mínimos quadrados e HMM) ao EKF, ao Kalman iterado, ao IMM, ao ESKF/IESKF e a um quadrotor com quaternions tridentes. Personagens: Pepé Le Gambá e Penélope.
Comece pelo guia: [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/kalman/notebooks/kalman_guia_da_serie.ipynb)

**[Números, movimento e Kalman: de ℝ aos quaternions tridentes](dominios/)**: a mesma história contada em oito domínios de números (ℝ, ℝ², duais, complexos, dual-complexos, quaternions, dual quaternions, tridentes), cada um em quatro etapas: álgebra, sistemas de equações, EDOs e uma perseguição com Kalman. Domínios I a IV (reais, vetores do plano, duais e complexos) prontos.
Comece pela raiz: [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X0_raiz.ipynb)

## Como as séries se ligam

A série *Domínios* dá a base numérica (ponto flutuante, condicionamento, Newton e Gauss–Newton, Runge–Kutta e RKMK, discretização exata do ruído) que a série Kalman supõe; a série Kalman dá a teoria dos filtros que a etapa (d) de cada domínio usa em miniatura. O guia da série Kalman (seção 7) tem a tabela de pontes, e cada nó de domínio tem uma seção "Para quem vem da série Kalman".

## Estrutura

```
kalman/      notebooks, construtor, vídeos e fontes da série Kalman
dominios/    notebooks e construtor da série Domínios
```

Em cada série, a pasta `notebooks/` tem os notebooks já executados e a pasta `construtor/` tem o código Python que os gera. Os notebooks não são editados à mão: cada correção é feita no construtor, e a série é regenerada e reexecutada (`python build.py` dentro de `construtor/`).
