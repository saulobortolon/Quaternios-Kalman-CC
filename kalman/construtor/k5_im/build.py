import base64, io, os
import nbformat as nbf
from PIL import Image

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fontes", "im")
cells = []

import re as _re
def _partes(s):
    # quebra células longas em títulos (## / ###) e antes das caixas "Nota do tradutor"/"Dica"
    s = s.strip("\n")
    pedacos = _re.split(r"\n(?=#{2,3} |> \*\*(?:Nota do tradutor|Dica))", s)
    return [p.strip("\n") for p in pedacos if p.strip()]

def md(s):
    for p in _partes(s):
        cells.append(nbf.v4.new_markdown_cell(p))

def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))

_cache = {}
def _png(nome, largura_max=1600):
    if nome not in _cache:
        im = Image.open(os.path.join(SRC, nome)).convert("RGB")
        if im.width > largura_max:
            im = im.resize((largura_max, round(im.height * largura_max / im.width)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
        _cache[nome] = base64.b64encode(buf.getvalue()).decode()
    return _cache[nome]

def mdfig(s, *figs):
    """Célula markdown com figuras do original embutidas como anexos (attachment:NN.png)."""
    for p in _partes(s):
        c = nbf.v4.new_markdown_cell(p)
        usadas = [f for f in figs if "attachment:" + f in p]
        if usadas:
            c["attachments"] = {f: {"image/png": _png(f)} for f in usadas}
        cells.append(c)

for parte in ["p1.py", "p2.py"]:
    exec(open(os.path.join(os.path.dirname(__file__), parte), encoding="utf-8").read())

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "kalman_notas_im_kf_ekf_eskf_iekf_ieskf.ipynb")
print(len(cells), "células")
