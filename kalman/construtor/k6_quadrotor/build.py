import os, re
import nbformat as nbf

AQUI = os.path.dirname(os.path.abspath(__file__))
cells = []

def _partes(s):
    s = s.strip("\n")
    pedacos = re.split(r"\n(?=#{2,3} |> \*\*(?:Nota|Dica|Atenção))", s)
    return [p.strip("\n") for p in pedacos if p.strip()]

def md(s):
    for p in _partes(s):
        cells.append(nbf.v4.new_markdown_cell(p))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))

ALG = open(os.path.join(AQUI, "c_algebra.py"), encoding="utf-8").read()
FIL = open(os.path.join(AQUI, "c_filtros.py"), encoding="utf-8").read()

def trecho(texto, inicio, fim=None):
    i = texto.index(inicio)
    j = texto.index(fim, i + 1) if fim else len(texto)
    return texto[i:j].strip("\n")

exec(open(os.path.join(AQUI, "p_texto.py"), encoding="utf-8").read())

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "kalman_quadrotor_quaternions.ipynb")
print(len(cells), "células")
