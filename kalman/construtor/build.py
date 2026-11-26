"""Gera (e executa) os notebooks da série "O Filtro de Kalman para Cientistas da Computação".

Uso, a partir desta pasta:
    python build.py              gera e executa todos
    python build.py k3 k4        só os indicados (pelo prefixo do construtor)
    EXECUTAR=0 python build.py   só gera, sem executar

Cada construtor escreve o seu notebook na pasta atual; este programa roda cada um
dentro de ../notebooks, acrescenta no topo o botão "Abrir no Colab" e a navegação
(volta ao guia, repositório, série irmã) e executa o notebook com o nbconvert.
Os dois geradores de vídeo NÃO são executados aqui (cada um leva vários minutos
e produz um MP4; os vídeos prontos estão em ../videos).
"""
import os, subprocess, sys, time
import nbformat as nbf

AQUI = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.normpath(os.path.join(AQUI, "..", "notebooks"))
REPO = "saulobortolon/Quaternios-Kalman-CC"
PASTA = "kalman/notebooks"
GUIA = "kalman_guia_da_serie.ipynb"

# (prefixo, construtor, notebook gerado, executar?)
CONSTRUTORES = [
    ("k1",  "k1_duas_trilhas.py",       "mq_hmm_kalman_duas_trilhas.ipynb",            True),
    ("k2",  "k2_passo_a_passo.py",      "kalman_passo_a_passo.ipynb",                  True),
    ("k2b", "k2b_faragher.py",          "kalman_produto_de_gaussianas_faragher.ipynb", True),
    ("k3",  "k3_pepe_parte1.py",        "pepe_kalman_parte1_ekf_iekf.ipynb",           True),
    ("k3v", "k3v_gerador_video.py",     "pepe_kalman_gerador_video.ipynb",             False),
    ("k4",  "k4_pepe_parte2_imm.py",    "pepe_kalman_parte2_imm.ipynb",                True),
    ("k4v", "k4v_gerador_video_imm.py", "pepe_imm_gerador_video.ipynb",                False),
    ("k5",  "k5_im/build.py",           "kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb", True),
    ("k6",  "k6_quadrotor/build.py",    "kalman_quadrotor_quaternions.ipynb",          True),
    ("k0",  "k0_guia.py",               GUIA,                                          True),  # por último: confere os outros
]


def colab(arq, pasta=PASTA):
    return f"https://colab.research.google.com/github/{REPO}/blob/main/{pasta}/{arq}"


def navegacao(arq):
    selo = (f'<a href="{colab(arq)}" target="_parent"><img '
            'src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab"/></a>')
    repo = f"[repositório](https://github.com/{REPO})"
    irma = (f"série irmã: [*Números, movimento e Kalman*](../../dominios/notebooks/X0_raiz.ipynb) "
            f"([Colab]({colab('X0_raiz.ipynb', 'dominios/notebooks')}))")
    if arq == GUIA:
        linha = f"*O Filtro de Kalman para Cientistas da Computação* · {repo} · {irma}"
    else:
        linha = (f"*O Filtro de Kalman para Cientistas da Computação* · "
                 f"[← guia da série]({GUIA}) ([Colab]({colab(GUIA)})) · {repo} · {irma}")
    c = nbf.v4.new_markdown_cell(f"{selo}\n\n{linha}")
    c.metadata["navegacao"] = True
    return c


def gera(prefixo, construtor, arq, executar):
    t0 = time.time()
    subprocess.run([sys.executable, os.path.join(AQUI, construtor)], cwd=SAIDA, check=True)
    caminho = os.path.join(SAIDA, arq)
    nb = nbf.read(caminho, as_version=4)
    nb.cells = [c for c in nb.cells if not c.metadata.get("navegacao")]
    nb.cells.insert(0, navegacao(arq))
    nbf.write(nb, caminho)
    if executar and os.environ.get("EXECUTAR", "1") != "0":
        subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=3600", arq], cwd=SAIDA, check=True)
    print(f"{prefixo:4s} {arq}  ({time.time() - t0:.0f} s{'' if executar else ', não executado'})", flush=True)


if __name__ == "__main__":
    os.makedirs(SAIDA, exist_ok=True)
    pedidos = sys.argv[1:]
    for prefixo, construtor, arq, executar in CONSTRUTORES:
        if not pedidos or prefixo in pedidos:
            gera(prefixo, construtor, arq, executar)
