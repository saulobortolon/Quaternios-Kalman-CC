import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# O Filtro de Kalman para Cientistas da Computação
## Guia da série de notebooks

Este é o notebook **raiz** da série. Ele não ensina nada de novo: diz **o que é cada notebook**, **em que ordem** lê-los, **o que cada um supõe** que o leitor já sabe e **onde** cada conceito aparece. Os próximos notebooks da série serão acrescentados aqui, seguindo o mesmo modelo (seção 6).

**Para quem é a série.** Alunos de Ciência da Computação numa disciplina de Matemática para Aprendizado de Máquina. Supõe-se que o leitor conhece Python, NumPy, álgebra linear (produto de matrizes, inversa, autovalores) e probabilidade básica (gaussiana, regra de Bayes). **Não** se supõe nenhuma base de Sistemas Dinâmicos nem de Engenharia de Controle: tudo o que vem dessas áreas é explicado do zero, e sempre que possível traduzido para a linguagem de aprendizado de máquina.

**A ideia central da série.** O Filtro de Kalman é **aprendizado online de uma variável escondida**:

- o estado é um parâmetro desconhecido que se aprende a cada nova medida;
- o *prior* funciona como regularização;
- o ganho é uma taxa de aprendizado ótima;
- a atualização é um passo de mínimos quadrados (ou de Gauss–Newton, no caso não linear).

A série chega a essa ideia por dois caminhos que se encontram: os **mínimos quadrados** (variáveis contínuas) e as **cadeias de Markov escondidas** (variáveis discretas).

> **Onde estão os notebooks.** No repositório GitHub **[github.com/saulobortolon/Quaternios-Kalman-CC](https://github.com/saulobortolon/Quaternios-Kalman-CC)**, pasta `kalman/notebooks/` (já executados, com figuras e resultados). O mesmo repositório guarda a série irmã *Números, movimento e Kalman* (pasta `dominios/`); as pontes entre as duas estão na seção 7.
>
> **Para usar no Colab:** cada notebook tem, no topo, um botão **"Abrir no Colab"** e um link de volta para este guia; a tabela da seção 8 tem o link de cada um. Alterações feitas no Colab ficam só na sua sessão: para guardá-las, use "Arquivo → Salvar uma cópia no Drive".
>
> **Vídeos.** Os dois vídeos prontos estão em [`kalman/videos/`](https://github.com/saulobortolon/Quaternios-Kalman-CC/tree/main/kalman/videos); os notebooks geradores refazem os vídeos.
>
> **Cópia no Google Drive.** A [pasta antiga do Drive](https://drive.google.com/drive/folders/1dtp9t1VQoMD-0hpIQoRVyFf0MT8C0M0f?usp=sharing) continua disponível, mas a versão de referência é a do repositório.
""")

md(r"""
---
# 1. Mapa da série

```
                     ┌────────────────────────────────────────────┐
  (opcional)         │ 1. mq_hmm_kalman_duas_trilhas              │
  notebook de HMM ──▶│    mínimos quadrados ─┐                    │
  do passeio do      │    HMM ───────────────┴─▶ Kalman, RTS, EM,  │
  bêbado             │                            Gauss–Newton    │
                     └───────────────┬────────────────────────────┘
                                     │ "o que cada linha faz?"
                                     ▼
                     ┌────────────────────────────────────────────┐
                     │ 2. kalman_passo_a_passo                    │
                     │    um programa explicado linha a linha     │
                     │    + glossário                             │◀── 2b. produto de gaussianas
                     └───────────────┬────────────────────────────┘     (Faragher, 2012)
                                     │ "e num problema de verdade?"
                                     ▼
                     ┌────────────────────────────────────────────┐
                     │ 3. pepe_kalman_parte1_ekf_iekf             │──▶ gerador de vídeo da Parte 1
                     │    EKF, Kalman iterado, fusão de sensores  │
                     └───────────────┬────────────────────────────┘
                                     │ "e se o alvo muda de comportamento?"
                                     ▼
                     ┌────────────────────────────────────────────┐
                     │ 4. pepe_kalman_parte2_imm                  │──▶ gerador de vídeo do IMM
                     │    IMM: HMM sobre modos + Kalman por modo  │
                     └───────────────┬────────────────────────────┘
                                     ▼
                         próximos notebooks (seção 6)

  Referência teórica, para consultar a qualquer momento:
  5. kalman_notas_im_kf_ekf_eskf_iekf_ieskf
     KF, EKF, ESKF, IEKF e IESKF com a mesma notação (tradução comentada
     das notas de G. E. Im), com rotações, MAP e Gauss–Newton
                │
                ▼
  6. kalman_quadrotor_quaternions
     quaternion × dual quaternion × quaternion tridente num quadrotor
```

**Ordem sugerida.** 1 → 2 → 3 → 4, com o notebook 5 como referência ao lado (as seções 3 e 4 dele acompanham os notebooks 2 e 3; as seções 5 a 7 vão além do resto da série). Há também um atalho para quem quer começar pela intuição: ler o notebook 2 primeiro, voltar ao 1 para a teoria e seguir com o 3 e o 4.
""")

# ---------------------------------------------------------------------
md(r"""
---
# 2. Os notebooks

## 2.1 `mq_hmm_kalman_duas_trilhas.ipynb`: Duas trilhas até o Filtro de Kalman

**Pergunta que responde.** De onde vem o Filtro de Kalman, e por que ele se parece tanto com coisas que o aluno já conhece?

**O que contém.** Duas linhas de raciocínio independentes que se encontram no Kalman.

- **Trilha A, contínua.** Média recursiva (ganho $1/k$) → mínimos quadrados em lote → ridge como *prior* gaussiano (o χ² de Thacker & Lacey) → mínimos quadrados recursivos (RLS, com Sherman–Morrison e fator de esquecimento) → funções de base. A última etapa usa a Função de Franke com rede RBF e mostra que ridge, RLS e processo gaussiano dão o **mesmo** ajuste.
- **Trilha B, discreta.** Passeio do bêbado: matriz de transição → matriz de emissão → HMM com *forward–backward* → Viterbi → Baum–Welch. Mostra que o **passo M é um problema de mínimos quadrados** com momentos esperados. Termina com a máxima verossimilhança por otimização direta (L-BFGS, com gradiente pela identidade de Fisher).
- **Parte C, o encontro:**
  - o Kalman é o *forward* com estado contínuo;
  - com o estado parado, o Kalman **é** o RLS;
  - o suavizador RTS é **um único** problema de mínimos quadrados na trajetória inteira;
  - o EM de Shumway–Stoffer usa a **mesma função** de mínimos quadrados do Baum–Welch;
  - no caso não linear, EKF = 1 passo de Gauss–Newton, Kalman iterado = Gauss–Newton (Bell & Cathey);
  - seção C5b: o exemplo em que o EKF erra com certeza, e um exercício de "caça ao erro" nas notas de Im (2024).

**Pré-requisitos.** Álgebra linear e probabilidade básica. Ajuda (mas não é indispensável) ter visto o notebook de HMM do passeio do bêbado.

**Resultados que vale guardar.** 31 conferências numéricas de que coisas "diferentes" são iguais (RLS = ridge, GP = KRR, contagens = mínimos quadrados *one-hot*, RTS = mínimos quadrados em lote, IEKF = Gauss–Newton…). E duas surpresas honestas: no HMM, EM e L-BFGS param em máximos locais diferentes; no Kalman, um EM que usa só as médias suavizadas (o erro de Benhamou) se desvia por completo.

**Tempo de execução.** Cerca de 1,5 min.

---

## 2.2 `kalman_passo_a_passo.ipynb`: O Filtro de Kalman passo a passo

**Pergunta que responde.** O que cada linha de um programa de Kalman faz, e o que entra e sai de cada função a cada iteração?

**O que contém.** Um único programa (o passeio do bêbado contínuo, um modelo de nível local), explicado em cinco etapas:

1. a simulação, desmontada operação por operação (`standard_normal`, `cumsum`, `[:, None]`, matrizes 1×1);
2. as funções `kf_predicao`, `kf_atualizacao` e `kalman`, linha a linha (por que `solve` em vez de inversa, por que simetrizar `P`, o que é `slogdet`);
3. um **glossário** dos termos do algoritmo, cada um com uma analogia em computação (estado e crença, covariância, *prior* e *posterior*, inovação, $S$, **ganho**, $Q$ e $R$, $F$ e $H$, *prior* difuso, verossimilhança, Riccati);
4. tabelas com as **entradas de cada função em cada iteração**;
5. **onze quadros** do filtro em tempo real, uma medida de cada vez, com as contas daquele passo impressas e a atualização vista como produto de gaussianas.

**Pré-requisitos.** Nenhum além de Python e NumPy. É o melhor ponto de entrada para quem nunca viu um Kalman.

**Resultados que vale guardar.**
- Processar as medidas uma a uma dá exatamente o mesmo que a chamada em lote.
- O ganho converge ao valor da equação de Riccati.
- A verdade fica dentro da faixa de ±2σ em cerca de 93% do tempo.

**Tempo de execução.** Poucos segundos.

---

## 2.2b `kalman_produto_de_gaussianas_faragher.ipynb`: o Kalman como produto de gaussianas (complemento do notebook 2)

**Pergunta que responde.** Por que a atualização do Kalman tem a forma que tem, explicada sem nada além de multiplicar duas gaussianas?

**O que contém.** Acompanha, na mesma ordem e com o mesmo exemplo (um trem num trilho), as *lecture notes* de R. Faragher, "Understanding the Basis of the Kalman Filter Via a Simple and Intuitive Derivation" (*IEEE Signal Processing Magazine*, 2012). O texto do notebook é próprio: o artigo deve ser lido ao lado.

- a **entrada de controle** $Bu_t$ (o maquinista acelera e freia), que não aparece no resto da série;
- a dedução de $P^-=FPF^\top+Q$, conferida por Monte Carlo;
- as quatro figuras do artigo refeitas (*prior*, predição, medida, fusão);
- o produto de duas gaussianas, com a **constante de normalização que o artigo omite**: é a verossimilhança da inovação, o `ll` dos outros notebooks;
- o tempo de voo como medida, e o aparecimento de $H=1/c$;
- **além do artigo**: o produto de gaussianas em duas dimensões (elipse inclinada × faixa), que mostra como medir só a posição corrige a velocidade; e o trem completo, com e sem o controle;
- notas críticas, com as limitações e os deslizes do artigo.

**Pré-requisitos.** Nenhum além de probabilidade básica. Pode ser lido antes ou depois do notebook 2.

**Resultados que vale guardar.** As precisões se somam ($1/\sigma^2=1/\sigma_1^2+1/\sigma_2^2$). A área do produto é a verossimilhança da medida. A receita escalar de "converter para o domínio da medida e voltar" só vale com $H$ inversível. Conhecer o controle reduz o erro de velocidade do trem de 1,32 para 0,16 m/s.

**Tempo de execução.** Poucos segundos.

---

## 2.3 `pepe_kalman_parte1_ekf_iekf.ipynb`: Pepé Le Gambá, Parte 1

**Pergunta que responde.** Como estimar a posição de algo que não se vê, a partir de medidas **não lineares** e ruidosas de vários sensores?

**O que contém.** Pepé Le Gambá persegue Penélope, que se esconde e foge. Ele não a vê: sente o **gradiente do cheiro** (Fase 1), também a **intensidade** (Fase 2) e, de perto, **ouve o coração** dela (Fase 3). O notebook abre com a história do personagem e por que ele é hoje considerado politicamente incorreto. O Pepé do desenho é usado como exemplo de **mau estimador**: uma crença com variância zero, que nenhuma evidência corrige. Os conteúdos:

- jacobianos das medidas, conferidos pela **derivada de passo complexo**;
- EKF e Kalman iterado comparados nas três fases;
- **integração de sensores**: empilhar medidas = atualizar em sequência = somar informação. Uma tabela mostra que o gradiente só informa a **direção** e as intensidades só informam a **distância**;
- oito quadros da perseguição, com as contas de cada passo;
- comparação em 40 perseguições, incluindo o **Pepé teimoso** ($q\approx0$).

**Pré-requisitos.** Notebooks 1 (seção C5) e 2.

**Resultados que vale guardar.** O erro mediano cai de 1,36 (só gradiente) para 0,19 (nariz + ouvidos) com o Kalman iterado. O **EKF com intensidade diverge**: confiante demais logo no primeiro passo, ele "inventa" uma fuga, como no exemplo de Bell & Cathey. O Pepé teimoso se perde a cada fuga.

**Tempo de execução.** Cerca de 30 s.

**Material complementar.** O vídeo pronto é [`kalman/videos/pepe_kalman_7_configuracoes.mp4`](https://github.com/saulobortolon/Quaternios-Kalman-CC/blob/main/kalman/videos/pepe_kalman_7_configuracoes.mp4). `pepe_kalman_gerador_video.ipynb` gera esse vídeo com um trecho para cada uma das sete configurações e uma cartela-resumo calculada. Os parâmetros ficam todos num bloco, para os alunos experimentarem.

---

## 2.4 `pepe_kalman_parte2_imm.ipynb`: Pepé Le Gambá, Parte 2 (IMM)

**Pergunta que responde.** E se o alvo **muda de comportamento**? Como estimar ao mesmo tempo **onde** ele está e **em que modo** está, e como **prever** onde estará?

**O que contém.**

- **Penélope com três modos:** escondida, fugindo em linha reta e procurando esconderijo com corridinhas. As fugas são longas o bastante para Pepé deixar de ouvir o coração.
- **Pepé com uma cadeia de Markov sobre os modos.** A matriz de transição sai das durações médias.
- **Um modelo de movimento por modo:** parada (com velocidade de arranque incerta), velocidade constante e modelo de Singer.
- **O IMM em cinco etapas:** misturar, prever, atualizar, pesar os modos, combinar. A etapa de pesar os modos é **literalmente o *forward* do HMM**, com a verossimilhança de cada Kalman no papel da emissão.
- **Duas estratégias:** mirar onde ela **está** e onde ela **estará** (prevendo com a cadeia).
- **Seis quadros** com probabilidades de modo, verossimilhanças e pesos da mistura; uma matriz de confusão dos modos; o erro de previsão a 2 e 4 s; uma comparação em mundos de 2 e de 3 modos.

**Pré-requisitos.** Notebook 3 e, para a ligação com o HMM, a Trilha B do notebook 1.

**Resultados que vale guardar.**
- **Prever é onde o IMM ganha:** a 4 s, o erro de previsão durante uma fuga cai de cerca de 6,5 para cerca de 4.
- Só o IMM de 3 modos é **honesto** no mundo de 3 modos (elipse de 2σ perto dos 86% ideais).
- Mirar onde ela estará **não** aumenta os encontros: o gargalo é a velocidade de Pepé, não a previsão.
- Com ela escondida, o modelo único prevê melhor, porque nenhum modelo sabe que é Pepé quem dispara a fuga.

**Tempo de execução.** Cerca de 2,5 min.

**Material complementar.** O vídeo pronto é [`kalman/videos/pepe_imm_configuracoes.mp4`](https://github.com/saulobortolon/Quaternios-Kalman-CC/blob/main/kalman/videos/pepe_imm_configuracoes.mp4). `pepe_imm_gerador_video.ipynb` gera o vídeo do IMM, com uma bolinha por filtro de tamanho proporcional à probabilidade do modo, o destino de Pepé, as probabilidades de modo empilhadas e uma cartela-resumo com o erro de previsão a 4 s.

---

## 2.5 `kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb`: Notas sobre o Filtro de Kalman (KF, EKF, ESKF, IEKF, IESKF)

**Pergunta que responde.** Como se relacionam os cinco filtros da família, com uma notação única, e o que fazer quando o estado nem é um vetor (uma rotação, por exemplo)?

**O que contém.** Tradução comentada para o português, com as figuras originais, das notas de G. E. Im (*Notes on Kalman Filter*, arXiv:2406.06427v3, 2024, licença CC BY), mais código, conferências e exercícios:
- filtro de Bayes recursivo, inclusive implementado **numa grade** (e por que o caso gaussiano cabe em dois números);
- KF, com o efeito de um $\mathbf P_0$ mal escolhido, o KF 1D e a figura das densidades refeita;
- EKF com **jacobianos dos ruídos** ($\mathbf F_{\mathbf w}$, $\mathbf H_{\mathbf v}$), num robô de duas rodas com faróis;
- **ESKF**: para estados vetoriais, idêntico ao EKF; na **atitude de um corpo rígido** (giroscópio, acelerômetro, magnetômetro), com $\boxplus$, $\mathrm{Exp}/\mathrm{Log}$ de $SO(3)$ e o reset $\mathbf G=\mathbf J_r$;
- **IEKF**: a forma impressa no original diverge, a forma de Bell e Cathey chega ao MAP;
- **IESKF**: a mesma correção iterada, na variedade, conferida contra o MAP exato;
- deduções do KF (marginalização, forma de informação, gaussiana condicional), MAP = Gauss–Newton = IEKF, e a dedução do IESKF, cada passo conferido;
- uma lista de **19 notas do tradutor** com os erros do original e onde cada um foi verificado.

**Pré-requisitos.** Notebook 2 para o KF; notebooks 1 (C5) e 3 para EKF e IEKF. As seções de ESKF e IESKF pedem só álgebra linear e a ideia de rotação.

**Resultados que vale guardar.**
- O EKF é Gauss–Newton com uma iteração; o IEKF, Gauss–Newton até convergir; o IESKF, Gauss–Newton numa variedade.
- O "estado de erro" só muda alguma coisa quando o estado não é vetor; com rotações, o nominal nunca sai de $SO(3)$.
- Num caso difícil (40° de erro de atitude), uma correção deixa 6,6° de erro; o IESKF chega ao MAP (1,2°) em 3 ou 4 iterações.
- Sem o reset $\mathbf G$, o ESKF converge mais devagar e fica mais otimista por vários segundos.

**Tempo de execução.** Poucos segundos.

---

## 2.6 `kalman_quadrotor_quaternions.ipynb`: Kalman para um quadrotor com quaternions, dual quaternions e quaternions tridentes

**Pergunta que responde.** Que objeto deve guardar o estado de um drone guiado por IMU (quaternion + vetores, dual quaternion + velocidade, ou um único quaternion tridente), e isso muda o filtro de Kalman?

**O que contém.**
- As três álgebras, conferidas contra os grupos de matrizes $SO(3)$, $SE(3)$ e $SE_2(3)$; o quaternion tridente de Ouyang e Wu (2022), com duas unidades duais.
- A navegação inercial numa única equação de tridentes, $\tilde q^+=\Gamma\otimes\Phi(\tilde q)\otimes\Upsilon$ (gravidade à esquerda, IMU à direita).
- Um quadrotor em voo de Lissajous, com IMU simulada.
- O mesmo ESKF com quatro escolhas de $\boxplus$: quaternion + vetores, dual quaternion + velocidade (Filipe et al.), tridente com erro à esquerda e à direita (LQEKF e RQEKF de Ouyang e Wu). Todos os jacobianos são deduzidos à mão e conferidos numericamente.
- Experimento 1: pose a 10 Hz e 0,5 Hz (a pergunta de Filipe et al.). Experimento 2: só GPS, com a guinada inicial errada de 10° a 150°.
- O quaternion dentro do estado (Jeon et al., 2022) contra o ESKF, e a não observabilidade de "massa × inércia × coeficientes dos rotores" (Böhm et al., 2021).

**Pré-requisitos.** Quaternions e dual quaternions; notebook 5, seções 5 e 7.

**Resultados que vale guardar.**
- Com erros pequenos, as quatro representações dão os mesmos números; a vantagem do dual quaternion a 0,5 Hz relatada por Filipe et al. não se reproduz.
- Com o tridente, o erro tem uma dinâmica que **não depende da estimativa**. A 150° de guinada errada, os tridentes convergem em 23 de 24 simulações e continuam honestos (NEES perto do ideal); quaternion + vetores converge em 5 de 12, e já a 45° se declara mais certo do que está. O dual quaternion fica no meio (10 de 12, mas igualmente otimista), porque deixa a velocidade fora do grupo.
- O erro à direita só funciona com a $\mathbf P_0$ escrita no seu referencial ($\mathrm{Ad}\,\mathbf P_0\,\mathrm{Ad}^\top$); com a $\mathbf P_0$ ingênua, é o pior de todos.
- O EKF com o quaternion no estado acerta a atitude tanto quanto o ESKF, mas carrega uma variância radial sem significado.

**Tempo de execução.** Cerca de 6 min (ajustável por `N_MC`).
""")

# ---------------------------------------------------------------------
md(r"""
---
# 3. Onde está cada conceito

| conceito | 1. duas trilhas | 2. passo a passo | 3. Pepé, Parte 1 | 4. Pepé, Parte 2 | 5. notas de Im | 6. quadrotor |
|---|---|---|---|---|---|---|
| predição → atualização (filtro bayesiano) | C1 | seções 2 e 4 | seção 4 | seção 5 | seções 2 e 3 (inclusive numa grade) | seção 5 |
| mínimos quadrados, ridge = *prior* | A2, A3 | — | seção 4 (o χ²) | — | 1.2 (peso), 8.3 (regressão = ganho) | — |
| RLS, Sherman–Morrison, Woodbury | A4, C2 | — | — | — | seções 8 e 10 | — |
| funções de base, processo gaussiano | A5 | — | — | — | — | — |
| cadeia de Markov, HMM, *forward* | B1–B3 | — | — | seções 4 e 5 | 1.4 (o grafo) | — |
| Viterbi | B4 | — | — | — | — | — |
| EM: Baum–Welch e Shumway–Stoffer | B5, C4 | — | — | exercício 3 | — | — |
| máxima verossimilhança por otimização | B6, C4 | — | — | — | seção 9 (MAP como MLE) | — |
| ganho de Kalman, $Q$, $R$, Riccati | C1 | glossário, seção 5 | seção 2 | seção 3 | 3.2 | — |
| produto de gaussianas, entrada de controle $Bu$ | — | seção 4 (quadros) | — | — | seção 3 | — |
| suavizador RTS = mínimos quadrados em lote | C3 | — | — | — | — | — |
| EKF, Kalman iterado = Gauss–Newton | C5, C5b | — | seções 4–6 | seção 5 | seções 4, 6 e 9 | exercício 2 |
| jacobianos (e passo complexo) | — | — | seção 3 | seção 2 | seção 4 (inclusive dos ruídos) | seção 5 (conferidos na álgebra) |
| estado de erro, rotações, $\boxplus$ (ESKF, IESKF) | — | — | — | — | seções 5, 7 e 10 | todo o notebook (SE(3), SE₂(3)) |
| integração de sensores | — | — | seção 7 | seção 2 | 5.4 (giroscópio + acelerômetro + magnetômetro) | seções 6–8 |
| consistência (elipse de 2σ, NEES) | — | seção 5 | seção 9 | seções 7 e 9 | seções 3, 4 e 5.4 | seções 6 e 7 (NEES) |
| IMM, modelos com chaveamento | — | — | exercício 6 | todo o notebook | — | — |
| prever o futuro com o modelo | — | — | exercício 5 | seções 6 e 8 | — | — |
| leitura crítica de textos | C5b (Im), C4 (Benhamou) | — | — | — | todo o notebook (19 notas) | seções 6–8 (Filipe, Jeon, Böhm) |
| dual quaternions, quaternions tridentes | — | — | — | — | — | seções 2 e 3 |
| filtro de Kalman invariante (erro independente da estimativa) | — | — | — | — | — | seções 5 e 7 |
""")

md(r"""
---
# 4. Convenções da série

- **Língua.** Texto e comentários em português; nomes de variáveis em português quando possível (`predicao`, `atualizacao`, `ganho`), com exceção de símbolos consagrados ($F$, $H$, $Q$, $R$, $K$, $P$).
- **Notação.**
  - $m$ e $P$ são a média e a covariância da crença;
  - $m^-$ e $P^-$ são a predição, antes da medida;
  - $v$ é a inovação, $S$ a sua covariância e $K$ o ganho;
  - no HMM, $A$ é a matriz de transição, $B$ a de emissão e $\alpha,\beta,\gamma,\xi$ as quantidades do *forward–backward*;
  - no IMM, $\Pi$ é a matriz de transição dos modos e $\mu$ as probabilidades de modo.
- **Conferências.** Toda equivalência afirmada no texto é verificada numericamente por uma função `confere(nome, a, b, tol)`, que imprime `OK` ou `FALHOU` e interrompe a execução se falhar. Um notebook que roda até o fim sem `FALHOU` confirma todas as suas afirmações.
- **Bibliotecas.** Só NumPy, SciPy, pandas e Matplotlib (e `ffmpeg` para os geradores de vídeo). Todos rodam no Google Colab sem instalar nada.
- **Sementes.** Toda simulação usa uma semente explícita. Mudá-la é sempre um bom exercício: os números mudam, as conclusões devem se manter.
- **Honestidade dos resultados.** Quando um resultado contraria o que se esperava (o EKF que diverge, o IMM que prevê pior quando ela está escondida, a perseguição que não melhora), o texto diz isso e explica por quê.
""")

md(r"""
---
# 5. Textos de referência usados até agora

| texto | usado em | observação |
|---|---|---|
| G. Welch, G. Bishop. *An Introduction to the Kalman Filter* (2001) | 1 | introdução clássica; exemplo da constante aleatória |
| R. Faragher. Understanding the Basis of the Kalman Filter Via a Simple and Intuitive Derivation (*IEEE SPM*, 2012) | 2b | dedução pelo produto de gaussianas; omite a normalização e a regra de Bayes, e generaliza para matrizes só por analogia |
| N. Thacker, A. Lacey. *The Likelihood Interpretation of the Kalman Filter* (Tina Memo) | 1, 3 | o Kalman como minimização de um χ² |
| S. Roweis, Z. Ghahramani. A Unifying Review of Linear Gaussian Models (1999) | 1 | unifica PCA, FA, Kalman, HMM; EM do sistema linear |
| F. Stulp, O. Sigaud. Many regression algorithms, one unified model (2015) | 1 | funções de base; a eq. (58) tem um erro, discutido em A5 |
| E. Benhamou. *Kalman Filter Demystified* (2019) | 1 | **leitura crítica**: vários erros, inclusive no passo M do EM |
| B. Bell, F. Cathey. The iterated Kalman filter update as a Gauss–Newton method (1993) | 1, 3 | IEKF = Gauss–Newton; exemplo do EKF enviesado |
| G. E. Im. *Notes on Kalman Filter* (arXiv:2406.06427, 2024, CC BY) | 1, 5 | traduzido e comentado no notebook 5; bom mapa da família; a correção do IEKF impressa diverge (corrigida pelo próprio texto nas seções 7 e 9) |
| J. Solà. *Quaternion kinematics for the error-state Kalman filter* (arXiv:1711.02508, 2017) | 5 | referência do ESKF com rotações |
| D. He, W. Xu, F. Zhang. Kalman filters on differentiable manifolds (arXiv:2102.03804, 2021) | 5 | IESKF em variedades |
| W. Ouyang, Y. Wu. A trident quaternion framework for inertial-based navigation, Partes I e II (*IEEE TAES* 2022; arXiv:2102.12217 e 2102.12220) | 6 | o quaternion tridente; LQEKF e RQEKF; convergência com guinada arbitrária (reproduzida no drone) |
| N. Filipe, M. Kontitsis, P. Tsiotras. EKF for spacecraft pose estimation using dual quaternions (*JGCD*, 2015) | 6 | DQ-MEKF; a vantagem a 0,5 Hz não se reproduz no quadrotor com IMU |
| S. Incicco, J. I. Giribet, L. Colombo. Enhanced UAV navigation systems through sensor fusion with trident quaternions (arXiv:2504.14133) | 6 | o tridente num multirrotor real, com GPS |
| H. Jeon et al. Quaternion-based IEKF for vision/IMU 6-DOF displacement monitoring (*IEEE Sensors J.*, 2022) | 6 | quaternion no estado; IEKF na forma correta |
| C. Böhm et al. Combined system identification and state estimation for a quadrotor UAV (ICRA 2021) | 6 | rotores como entrada; parâmetros no estado; observabilidade |
| A. Barrau, S. Bonnabel. The invariant extended Kalman filter as a stable observer (*IEEE TAC*, 2017) | 6 | o filtro invariante e os sistemas afins no grupo |
| L. Rabiner. A tutorial on hidden Markov models (1989) | 1 | HMM |
| R. Shumway, D. Stoffer (1982) | 1 | EM para o sistema linear |
| H. Blom, Y. Bar-Shalom (1988); Bar-Shalom, Li, Kirubarajan (2001) | 4 | IMM |
| R. Singer (1970) | 4 | modelo de alvo que manobra |
""")

# ---------------------------------------------------------------------
md(r"""
---
# 6. Como acrescentar um novo notebook à série

Os próximos notebooks (sobre outros temas do Filtro de Kalman para cientistas da computação) entram neste guia de três maneiras:

1. **Uma nova subseção na seção 2**, com os mesmos campos dos anteriores:
   - **Pergunta que responde**: uma frase;
   - **O que contém**: tópicos principais;
   - **Pré-requisitos**: quais notebooks da série;
   - **Resultados que vale guardar**: os números ou fatos que o aluno deve levar;
   - **Tempo de execução**;
   - **Material complementar**, se houver.
2. **Uma nova coluna na tabela da seção 3**, marcando onde cada conceito aparece, e novas linhas para os conceitos que o notebook introduz.
3. **Uma nova entrada no dicionário `SERIE`** da última célula, para que a verificação da seção 8 o encontre.

As referências novas vão para a tabela da seção 5.

## Temas candidatos para os próximos notebooks

Lista aberta, só para orientar:

- **Filtro de Kalman *unscented* (UKF) e filtro de partículas:** o que fazer quando linearizar não basta (multimodalidade, como no gradiente de Pepé longe de Penélope).
- **Perdas robustas:** Huber e IRLS dentro do Gauss–Newton, para medidas com *outliers*.
- **Vieses da IMU e o filtro invariante "imperfeito":** o estado de 15 dimensões do notebook 6, com vieses fora do grupo; odometria visual-inercial (FAST-LIO, EuRoC).
- **Rotores no modelo:** o quadrotor de Böhm et al., com as velocidades dos rotores como entrada e massa e inércias no estado.
- **Aprender o modelo:** EM e otimização para $Q$, $R$ e para a matriz $\Pi$ do IMM; ligação com redes neurais recorrentes.
- **Kalman e aprendizado profundo:** o Kalman como camada diferenciável e o "KalmanNet".
- **Séries temporais:** modelos estruturais (nível, tendência, sazonalidade) em forma de espaço de estados.
""")

md(r"""
---
# 7. Pontes com a série *Números, movimento e Kalman*

A série irmã, no mesmo repositório (pasta `dominios/`), conta uma única história em oito domínios de números (ℝ, ℝ², duais, complexos, dual-complexos, quaternions, dual quaternions, quaternions tridentes). Cada domínio tem quatro etapas: (a) álgebra, (b) sistemas de equações, (c) EDOs e (d) uma perseguição com Kalman. A raiz dela é [X0_raiz](../../dominios/notebooks/X0_raiz.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X0_raiz.ipynb)).

As duas séries se ajudam nos dois sentidos:

- a série *Domínios* dá a **base numérica** que esta série supõe sem dizer: ponto flutuante, condicionamento, Newton e Gauss–Newton, Runge–Kutta, discretização exata de $Q$;
- esta série dá a **teoria dos filtros** que a etapa (d) de cada domínio usa em miniatura.

Hoje estão prontos os domínios I (números reais) e II (vetores do plano); as linhas abaixo crescem com a série.

| notebook desta série | ponte na série *Domínios* | o que liga os dois |
|---|---|---|
| 1. duas trilhas (mínimos quadrados, Gauss–Newton, IEKF) | [X1b](../../dominios/notebooks/X1b_reais_sistemas.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1b_reais_sistemas.ipynb)) | mínimos quadrados e Gauss–Newton em ℝ, com condicionamento e as falhas de Newton |
| 1 e 3 (EKF × Kalman iterado) | [X1d](../../dominios/notebooks/X1d_reais_perseguicao.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1d_reais_perseguicao.ipynb)) | o IEKF de Bell e Cathey numa perseguição 1D; no caso linear ele coincide com o Kalman (conferido) |
| 2. passo a passo; 2b. produto de gaussianas | [X1d](../../dominios/notebooks/X1d_reais_perseguicao.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1d_reais_perseguicao.ipynb)) | o Kalman 1D com sensor linear (a régua) e o filtro na grade, que reproduz o Kalman nesse caso |
| 3. Pepé, Parte 1 (o EKF que diverge com o olfato) | [X1d](../../dominios/notebooks/X1d_reais_perseguicao.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1d_reais_perseguicao.ipynb)) | a mesma lição em 1D: com o sensor de olfato, EKF e IEKF se perdem e o filtro na grade não |
| 5. notas de Im ($\boxplus$, filtro numa grade, ruído de processo) | [X1a](../../dominios/notebooks/X1a_reais_algebra.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1a_reais_algebra.ipynb)), [X1c](../../dominios/notebooks/X1c_reais_edo.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1c_reais_edo.ipynb)) | $\boxplus$ = compor com $\operatorname{Exp}$ (trivial em ℝ); $F$, $G$ e $Q$ exatos pelo método de Van Loan |
| 6. quadrotor (quaternions, dual quaternions, tridentes; RK e grupos de Lie) | [X1c](../../dominios/notebooks/X1c_reais_edo.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1c_reais_edo.ipynb)); depois os domínios VI, VII e VIII | Runge–Kutta em ℝ e a passagem para RKMK, que integra no grupo |
| todos (ponto flutuante, `confere` com tolerância) | [X1a](../../dominios/notebooks/X1a_reais_algebra.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X1a_reais_algebra.ipynb)) | por que as conferências usam tolerâncias: associatividade que falha, cancelamento, soma de Kahan |
| 3. Pepé, Parte 1 (gradiente do cheiro, sensores que se completam) | [X2d](../../dominios/notebooks/X2d_vetores2d_perseguicao.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X2d_vetores2d_perseguicao.ipynb)) | no plano, a intensidade só diz a distância e o gradiente confunde perto com longe; juntos, o EKF não perde Penélope |
| 4. IMM (várias hipóteses ao mesmo tempo) | [X2d](../../dominios/notebooks/X2d_vetores2d_perseguicao.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X2d_vetores2d_perseguicao.ipynb)) | o filtro de partículas carrega várias hipóteses de posição; ajuda com a ambiguidade discreta (perto/longe), não com a contínua (o círculo) |
| 5. notas de Im (predição do EKF, $\mathbf J\mathbf P\mathbf J^\top$) | [X2c](../../dominios/notebooks/X2c_vetores2d_edo.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X2c_vetores2d_edo.ipynb)) | a incerteza de um lançamento: elipse linearizada contra Monte Carlo, e a "banana" quando a incerteza cresce |
| 1 (mínimos quadrados, Gauss–Newton) | [X2b](../../dominios/notebooks/X2b_vetores2d_sistemas.ipynb) ([Colab](https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/dominios/notebooks/X2b_vetores2d_sistemas.ipynb)) | trilateração com Gauss–Newton na cota $\sigma^2(\mathbf J^\top\mathbf J)^{-1}$, GDOP, e o reflexo que ela não vê |
""")

md(r"""
---
# 8. Verificação: quais notebooks estão nesta pasta?

A célula abaixo procura os arquivos da série na pasta atual e mostra o que encontrou. No Jupyter local, os nomes viram **links** que abrem o notebook.

**No Colab.** Para que a célula encontre os arquivos, traga o repositório para a sessão do Colab antes de rodá-la:

```python
!git clone -q https://github.com/saulobortolon/Quaternios-Kalman-CC
%cd Quaternios-Kalman-CC/kalman/notebooks
```

Mesmo sem isso, a coluna "Colab" da tabela abre cada notebook diretamente do GitHub.
""")

code(r"""
import os
REPO = "https://github.com/saulobortolon/Quaternios-Kalman-CC"
COLAB = "https://colab.research.google.com/github/saulobortolon/Quaternios-Kalman-CC/blob/main/kalman/notebooks/"
from IPython.display import Markdown, display

SERIE = [
    # (ordem, arquivo, título, tempo aproximado de execução)
    ("1",  "mq_hmm_kalman_duas_trilhas.ipynb",  "Duas trilhas até o Filtro de Kalman",           "~1,5 min"),
    ("2",  "kalman_passo_a_passo.ipynb",        "O Filtro de Kalman passo a passo",               "segundos"),
    ("2b", "kalman_produto_de_gaussianas_faragher.ipynb", "O Kalman como produto de gaussianas (Faragher, 2012)", "segundos"),
    ("3",  "pepe_kalman_parte1_ekf_iekf.ipynb", "Pepé Le Gambá, Parte 1: EKF e Kalman iterado",   "~30 s"),
    ("3v", "pepe_kalman_gerador_video.ipynb",   "Gerador de vídeo da Parte 1",                    "~5–10 min"),
    ("4",  "pepe_kalman_parte2_imm.ipynb",      "Pepé Le Gambá, Parte 2: IMM",                    "~2,5 min"),
    ("4v", "pepe_imm_gerador_video.ipynb",      "Gerador de vídeo do IMM",                        "~5–10 min"),
    ("5",  "kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", "Notas sobre o Filtro de Kalman: KF, EKF, ESKF, IEKF, IESKF (Im, 2024)", "segundos"),
    ("6",  "kalman_quadrotor_quaternions.ipynb", "Kalman para um quadrotor: quaternions, dual quaternions e tridentes", "~6 min"),
    # novos notebooks entram aqui
]

linhas = ["| # | notebook | título | execução | Colab | nesta pasta? |", "|---|---|---|---|---|---|"]
for ordem, arq, titulo, tempo in SERIE:
    existe = os.path.exists(arq)
    nome = f"[{arq}]({arq})" if existe else f"`{arq}`"
    linhas.append(f"| {ordem} | {nome} | {titulo} | {tempo} | [abrir]({COLAB}{arq}) | {'sim' if existe else '**não encontrado**'} |")
display(Markdown("\n".join(linhas)))
faltam = [arq for _, arq, _, _ in SERIE if not os.path.exists(arq)]
print(f"{len(SERIE) - len(faltam)} de {len(SERIE)} notebooks encontrados na pasta {os.getcwd()}")
if faltam:
    display(Markdown(f"Os notebooks que faltam estão no repositório, pasta `kalman/notebooks/`: [{REPO}]({REPO}/tree/main/kalman/notebooks)."))
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "kalman_guia_da_serie.ipynb")
print(len(cells), "células")
