"""Gera e executa todos os notebooks da série.  Uso:  python build.py [x1a x1b ...]"""
import os, sys, subprocess
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.chdir(AQUI)

TODOS = ["x1a", "x1b", "x1c", "x1d", "x1", "x2a", "x2b", "x2c", "x2d", "x2", "x3a", "x3b", "x3c", "x3d", "x3", "x4a", "x4b", "x4c", "x4d", "x4", "x0"]   # a raiz por último: a verificação dela procura os outros
alvo = sys.argv[1:] or TODOS
executar = os.environ.get("EXECUTAR", "1") == "1"

for nome in alvo:
    caminho = None
    ns = {"__name__": "__construtor__"}
    exec(open(os.path.join(AQUI, nome + ".py"), encoding="utf-8").read(), ns)
    caminho = os.path.join(AQUI, "..", "notebooks", ns["c"].arq)
    if executar:
        r = subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=1200", caminho], capture_output=True, text=True)
        print("   executado" if r.returncode == 0 else "   ERRO NA EXECUÇÃO\n" + r.stderr[-3000:])
