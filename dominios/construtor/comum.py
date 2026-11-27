"""Funções comuns do construtor da série "Domínios, movimento e Kalman".

Cada notebook é gerado por um arquivo xNN.py que chama md(), code() e salva().
Os notebooks são autossuficientes: as ferramentas de que precisam são repetidas
por estas funções, e não importadas de um módulo externo.
"""
import os, re
import nbformat as nbf

REPO = "saulobortolon/Quaternios-Kalman-CC"
PASTA_NB = "dominios/notebooks"
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks")

# ------------------------------------------------------------------ a série
DOMINIOS = [
    # (código, nome curto, número, grupo de movimentos, exemplo físico, perseguição)
    ("1", "reais", "números reais ℝ", "ℝ (translações na reta)",
     "bloco com mola, atrito e força aplicada", "ponto persegue ponto na reta, medindo uma função da distância"),
    ("2", "vetores2d", "vetores de ℝ²", "ℝ² (translações no plano)",
     "projétil com arrasto", "Pepé e Penélope no plano"),
    ("3", "duais", "números duais", "SO(2)×ℝ (giro e deslizamento no mesmo eixo)",
     "barra que gira e desliza num eixo", "barra que se alinha a um alvo no eixo"),
    ("4", "complexos", "números complexos ℂ", "ℂ (plano, com rotações por multiplicação)",
     "pêndulo de Foucault", "perseguição no plano em notação complexa"),
    ("5", "dualcomplexos", "números dual-complexos", "SE(2) (posição e atitude no plano)",
     "barra ou carrinho no plano", "veículo no plano que persegue e aponta"),
    ("6", "quaternions", "quaternions", "SO(3) (atitude no espaço)",
     "corpo girando livre (efeito da raquete de tênis)", "câmera que gira para manter o alvo centrado"),
    ("7", "dualquaternions", "dual quaternions", "SE(3) (pose no espaço)",
     "paralelepípedo lançado girando", "drone persegue drone"),
    ("8", "tridentes", "quaternions tridentes", "SE₂(3) (pose e velocidade no espaço)",
     "paralelepípedo ou drone com IMU", "drone interceptador que persegue, aponta e lança uma rede"),
]
ETAPAS = [
    ("a", "algebra", "álgebra: operações, propriedades e implementação"),
    ("b", "sistemas", "sistemas de equações lineares e não lineares"),
    ("c", "edo", "cinemática e dinâmica: equações diferenciais"),
    ("d", "perseguicao", "perseguição com filtro de Kalman"),
]
PRONTOS = {"1", "2", "3", "4", "5"}          # domínios já escritos

def arquivo(dom=None, etapa=None):
    if dom is None:
        return "X0_raiz.ipynb"
    nome = [d for d in DOMINIOS if d[0] == dom][0][1]
    if etapa is None:
        return f"X{dom}_{nome}.ipynb"
    sufixo = [e for e in ETAPAS if e[0] == etapa][0][1]
    return f"X{dom}{etapa}_{nome}_{sufixo}.ipynb"

def apendice(dom, slug):
    """Notebooks extras de um domínio (aplicações fora da sequência a–d)."""
    return f"X{dom}_ap_{slug}.ipynb"

APENDICES = {                                   # domínio → [(slug, título curto)]
    "5": [("slam", "SLAM por grafo de poses"), ("quatrobarras", "quatro barras"), ("carro", "odometria de um carro"),
          ("curling", "curling"), ("robocup", "futebol de robôs"), ("estilingue", "estilingue gravitacional"),
          ("trelica", "treliça de ponte")],
}

def colab(arq):
    return f"https://colab.research.google.com/github/{REPO}/blob/main/{PASTA_NB}/{arq}"

def link(arq, texto=None):
    """Link relativo (Jupyter, GitHub) mais o atalho para o Colab."""
    return f"[{texto or arq}]({arq}) ([Colab]({colab(arq)}))"

# a série irmã (Kalman para Cientistas da Computação), no mesmo repositório
PASTA_KALMAN = "kalman/notebooks"
GUIA_KALMAN = "kalman_guia_da_serie.ipynb"

def kalman(arq, texto=None):
    """Link para um notebook da série Kalman: relativo (Jupyter, GitHub) mais o atalho para o Colab."""
    rel = f"../../{PASTA_KALMAN}/{arq}"
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/{PASTA_KALMAN}/{arq}"
    return f"[{texto or arq}]({rel}) ([Colab]({url}))"

def selo(arq):
    return f'<a href="{colab(arq)}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab"/></a>'

# ------------------------------------------------------------------ construção
class Caderno:
    def __init__(self, arq):
        self.arq = arq
        self.cells = []

    @staticmethod
    def _partes(s):
        s = s.strip("\n")
        pedacos = re.split(r"\n(?=#{2,3} |> \*\*(?:Nota|Atenção|Dica|Para pensar))", s)
        return [p.strip("\n") for p in pedacos if p.strip()]

    def md(self, s):
        for p in self._partes(s):
            self.cells.append(nbf.v4.new_markdown_cell(p))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    def salva(self):
        nb = nbf.v4.new_notebook(cells=self.cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nb.metadata["colab"] = {"provenance": []}
        os.makedirs(SAIDA, exist_ok=True)
        caminho = os.path.join(SAIDA, self.arq)
        nbf.write(nb, caminho)
        print(f"{self.arq}: {len(self.cells)} células")
        return caminho

def cabecalho(c, titulo, subtitulo, dom=None, etapa=None, anterior=None, proximo=None):
    nav = []
    if dom is not None:
        nav.append(f"**Série:** {link(arquivo(), 'raiz')}")
        if etapa is not None:
            nav.append(f"**Domínio:** {link(arquivo(dom), 'nó ' + dom)}")
    if anterior:
        nav.append(f"**Anterior:** {link(anterior)}")
    if proximo:
        nav.append(f"**Próximo:** {link(proximo)}")
    c.md(f"""
{selo(c.arq)}

# {titulo}
## {subtitulo}

{' · '.join(nav)}
""")

FERRAMENTAS = r'''
# Ferramentas comuns da série (repetidas em todos os notebooks, para que cada um rode sozinho no Colab)
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 100,
                     "axes.spines.top": False, "axes.spines.right": False})

def confere(nome, obtido, esperado, tol=1e-9, relativo=False):
    """Compara duas quantidades. Imprime OK ou interrompe o notebook com uma mensagem clara."""
    a, b = np.asarray(obtido), np.asarray(esperado)
    tipo = complex if (np.iscomplexobj(a) or np.iscomplexobj(b)) else float      # aceita complexos (domínio IV)
    a, b = a.astype(tipo), b.astype(tipo)
    escala = max(np.max(np.abs(b)), 1e-300) if relativo else 1.0
    erro = float(np.max(np.abs(a - b)) / escala)
    ok = erro <= tol
    print(f"{'OK ' if ok else 'FALHOU'}  {nome}:  diferença {'relativa ' if relativo else ''}máxima = {erro:.2e}")
    assert ok, f"conferência falhou: {nome}"
'''
