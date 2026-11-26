# O Filtro de Kalman para Cientistas da Computação

Série de notebooks para alunos de Ciência da Computação, criada por Saulo Bortolon em parceria com Claude. O Filtro de Kalman é apresentado como **aprendizado online de uma variável escondida**, por dois caminhos que se encontram: os mínimos quadrados e as cadeias de Markov escondidas. Depois vêm o EKF e o Kalman iterado (Pepé Le Gambá), o IMM, as notas de Im (KF, EKF, ESKF, IEKF, IESKF) e um quadrotor com quaternions, dual quaternions e quaternions tridentes.

## Por onde começar

Abra o guia, [`notebooks/kalman_guia_da_serie.ipynb`](notebooks/kalman_guia_da_serie.ipynb), ou diretamente no Colab:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/kalman/notebooks/kalman_guia_da_serie.ipynb)

O guia diz o que é cada notebook, em que ordem lê-los, onde aparece cada conceito, e traz as pontes com a série irmã ([`../dominios/`](../dominios/)). Cada notebook tem, no topo, o botão "Abrir no Colab" e um link de volta ao guia.

| # | notebook |
|---|---|
| 1 | `mq_hmm_kalman_duas_trilhas.ipynb`: duas trilhas até o Filtro de Kalman |
| 2 | `kalman_passo_a_passo.ipynb`: o Filtro de Kalman passo a passo |
| 2b | `kalman_produto_de_gaussianas_faragher.ipynb`: o Kalman como produto de gaussianas (Faragher, 2012) |
| 3 | `pepe_kalman_parte1_ekf_iekf.ipynb`: Pepé Le Gambá, Parte 1 (EKF e Kalman iterado) |
| 3v | `pepe_kalman_gerador_video.ipynb`: gerador do vídeo da Parte 1 |
| 4 | `pepe_kalman_parte2_imm.ipynb`: Pepé Le Gambá, Parte 2 (IMM) |
| 4v | `pepe_imm_gerador_video.ipynb`: gerador do vídeo do IMM |
| 5 | `kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb`: notas sobre o Filtro de Kalman (tradução comentada de G. E. Im, 2024, CC BY) |
| 6 | `kalman_quadrotor_quaternions.ipynb`: Kalman para um quadrotor com quaternions, dual quaternions e tridentes |

## Estrutura (dentro de `kalman/`)

```
notebooks/     os notebooks, já executados (os dois geradores de vídeo ficam sem executar)
videos/        os dois vídeos prontos (MP4)
fontes/im/     figuras e texto LaTeX das notas de Im (arXiv:2406.06427, licença CC BY)
construtor/    o código que GERA os notebooks
  k0_guia.py ... k6_quadrotor/   um construtor por notebook
  build.py       gera e executa tudo:  python build.py   (ou  python build.py k3  para um só)
```

Os notebooks **não são editados à mão**: cada correção é feita no construtor, e a série é regenerada e reexecutada. Toda equivalência afirmada no texto é conferida por uma célula `confere(...)` que imprime `OK`.

## Requisitos para regenerar

Python 3.10+, `numpy`, `scipy`, `matplotlib`, `pandas`, `pillow`, `nbformat`, `nbconvert`, `jupyter`; `ffmpeg` para os geradores de vídeo.
