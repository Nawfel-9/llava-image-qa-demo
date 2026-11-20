"""Question-réponse sur image avec LLaVA-1.5-7B : interface web (Gradio).

    python app.py        ->  http://127.0.0.1:7860

Onglets :
  1. Question-réponse : la réponse du modèle et ses « coulisses » (ce qu'il reçoit réellement) ;
  2. Résultats : l'évaluation produite par eval.py ;
  3. Code : les fonctions principales de llava_core.py.
"""
import html
import inspect
import json
import textwrap
from pathlib import Path

import gradio as gr
from PIL import Image

import llava_core
from llava_core import LLaVA

HERE = Path(__file__).parent
PHOTO = HERE / "assets" / "coco-53236.jpg"

llava = LLaVA()

# Couleurs : une couleur par concept (image en bleu, projection en orange, texte en vert, réponse en violet)
CSS = """
:root { --bg:#F7F6F2; --panel:#ECEAE4; --line:#DDDAD2; --rule:#7D8593; --ink:#16181D; --ink2:#4A505C;
  --vision:#1F5FD1; --proj:#E8590C; --text:#2B8A3E; --answer:#9C36B5; --error:#C92A2A; }
html, body, gradio-app, .main, .wrap, .contain, .gradio-container { background: var(--bg) !important; }
.gradio-container { max-width: none !important; padding: 24px 40px !important; color: var(--ink);
  font-family: "Inter", "Segoe UI", sans-serif !important; }
footer { display: none !important; }

#head h1 { font-size: 40px; font-weight: 650; margin: 0; letter-spacing: -0.01em; color: var(--ink); }
#head p { font-size: 20px; color: var(--ink2); margin: 6px 0 0; }
button[role="tab"] { font-size: 20px !important; font-weight: 600 !important; }

#q textarea { font-size: 24px !important; line-height: 1.35 !important; color: var(--text) !important; font-weight: 500; }
#ask { min-height: 56px; font-size: 22px !important; font-weight: 600 !important; color: #fff !important;
  background: var(--ink) !important; border: none !important; }

.k { font-size: 16px; font-weight: 600; color: var(--ink2); letter-spacing: .02em; text-transform: uppercase; }
#answer { background: #fff; border-radius: 16px; padding: 20px 24px; min-height: 150px; }
#answer .a { margin-top: 8px; font-size: 26px; line-height: 1.4; font-weight: 500; color: var(--answer); }

#stats { background: var(--panel); border-radius: 16px; padding: 20px 24px; }
#stats .k { margin-bottom: 10px; }
#stats table { width: 100%; border-collapse: collapse; font-size: 19px; }
#stats td { padding: 5px 0; vertical-align: top; color: var(--ink); }
#stats td:first-child { width: 46%; color: var(--ink2); }
#stats .v { color: var(--vision); font-weight: 600; }
#stats .p { color: var(--proj); font-weight: 600; }
#stats .t { color: var(--text); font-weight: 600; }
#stats .l { color: var(--answer); font-weight: 600; }
#stats pre { margin: 6px 0 0; padding: 10px 12px; border-radius: 10px; background: #fff; color: var(--ink);
  font-size: 16px; white-space: pre-wrap; }

#results table { width: 100%; border-collapse: collapse; font-size: 18px; }
#results th { padding: 8px 10px; text-align: left; font-weight: 600; color: var(--ink2); }
#results td { padding: 8px 10px; vertical-align: top; color: var(--ink); }
#results td:first-child { color: var(--ink2); }
#results .q { color: var(--text); }
#results .ans { color: var(--answer); }
#results .ok { color: var(--text); font-weight: 700; }
#results .err { color: var(--error); font-weight: 700; }
#results .score { margin: 4px 0 14px; font-size: 30px; font-weight: 650; color: var(--ink); }
#results h3 { margin: 26px 0 8px; font-size: 22px; color: var(--ink); }

/* Tableaux : uniquement des filets horizontaux */
#stats table, #stats tr, #stats td, #results table, #results tr, #results th, #results td {
  border-left: none !important; border-right: none !important; border-top: none !important; }
#stats td, #results td { border-bottom: 1px solid var(--line) !important; }
#results th { border-bottom: 2px solid var(--rule) !important; }

#code .cm-content, #code .cm-gutters { font-size: 21px !important; line-height: 1.45 !important; }
#code .cm-content { filter: brightness(0.62) saturate(1.4); font-weight: 500; }
"""

# Thème clair uniquement : Gradio active le mode sombre en ajoutant la classe « dark » ;
# on la retire au chargement et on empêche qu'elle revienne.
LIGHT_ONLY = """() => {
  const strip = () => [document.documentElement, document.body].forEach((el) => el.classList.remove('dark'));
  strip();
  new MutationObserver(strip).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  new MutationObserver(strip).observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
}"""


def esc(s):
    return html.escape(str(s))


def answer_html(text, status=""):
    return f'<div class="k">Réponse de LLaVA {esc(status)}</div><div class="a">{esc(text) or "…"}</div>'


def stats_html(st):
    if not st:
        return ('<div class="k">Coulisses du modèle</div>'
                '<p style="font-size:19px;color:#4A505C">Posez une question pour voir ce que le modèle reçoit.</p>')
    prompt = st["prompt"].replace("<image>", "<image> (576 tokens)")
    seconds = f'{st["seconds"]:.1f}'.replace(".", ",")
    rows = [
        ("Image d'entrée", f'<span class="v">{st["pixel_values"][2]} × {st["pixel_values"][3]} px</span>'),
        ("Tokens visuels", f'<span class="v">{st["image_tokens"]}</span> (24 × 24 patchs de 14 px)'),
        ("Couche de CLIP", f'<span class="v">{st["vision_layer"]}</span> : couche 23 sur 24, CLS retiré'),
        ("Projection (MLP)", f'<span class="v">{st["projector_in"][1]} × {st["projector_in"][2]}</span> → '
                             f'<span class="p">{st["projector_out"][1]} × {st["projector_out"][2]}</span>'),
        ("Tokens de texte", f'<span class="t">{st["text_tokens"]}</span>'),
        ("Tokens générés", f'<span class="l">{st["new_tokens"]}</span>'),
        ("Temps de réponse", f"{seconds} s"),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (f'<div class="k">Coulisses du modèle</div><table>{body}</table>'
            f'<div class="k" style="margin-top:14px">Prompt envoyé à Vicuna</div><pre>{esc(prompt)}</pre>')


def on_ask(image, question):
    """Affiche la réponse au fur et à mesure, puis les coulisses une fois la génération terminée."""
    if image is None or not question.strip():
        yield answer_html(""), stats_html(None)
        return
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    yield answer_html("", "(génération…)"), stats_html(None)
    for text, st in llava.stream(image.convert("RGB"), question):
        if st is None:
            yield answer_html(text, "(génération…)"), gr.skip()
        else:
            yield answer_html(text), stats_html(st)


def results_html():
    path = HERE / "results.json"
    if not path.exists():
        return "<p>Lancez d'abord <code>python eval.py</code>.</p>"
    r = json.loads(path.read_text(encoding="utf-8"))
    n, total = r["score"]
    rows = "".join(
        f'<tr><td>{esc(x["cat"])}</td><td class="q">{esc(x["q"])}</td><td>{esc(x["expected"])}</td>'
        f'<td class="ans">{esc(x["answer"])}</td>'
        f'<td class="{"ok" if x["correct"] else "err"}">{"✓" if x["correct"] else "✗"}</td></tr>'
        for x in r["rows"])
    variants = "".join(
        f'<tr><td>{esc(x["cat"])}</td><td class="q">{esc(x["q"])}</td><td class="ans" colspan="3">{esc(x["answer"])}</td></tr>'
        for x in r["variants"])
    return (f'<div class="score">{n} / {total} réponses correctes</div>'
            f'<table><tr><th>Type</th><th>Question</th><th>Attendu</th><th>Réponse de LLaVA</th><th></th></tr>{rows}</table>'
            f'<h3>Même demande, autre formulation</h3>'
            f'<table><tr><th>Type</th><th>Question</th><th colspan="3">Réponse de LLaVA</th></tr>{variants}</table>')


CODE = "\n".join(textwrap.dedent(inspect.getsource(f)) for f in (
    llava_core.LLaVA.__init__, llava_core.LLaVA.prompt, llava_core.LLaVA.inputs, llava_core.LLaVA.ask))

with gr.Blocks(title="Démonstration LLaVA") as demo:
    gr.HTML('<div id="head"><h1>Question-réponse sur image avec LLaVA-1.5-7B</h1>'
            '<p>CLIP ViT-L/14-336 · projection MLP · Vicuna-7B — exécuté en local sur GPU</p></div>')
    with gr.Tabs():
        with gr.Tab("1 · Question-réponse"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    img = gr.Image(value=str(PHOTO), type="pil", label="Image", height=470)
                    q = gr.Textbox(label="Question", lines=2, elem_id="q", placeholder="Posez une question sur l'image…")
                    ask = gr.Button("Demander à LLaVA", elem_id="ask")
                with gr.Column(scale=6):
                    ans = gr.HTML(answer_html(""), elem_id="answer")
                    st = gr.HTML(stats_html(None), elem_id="stats")
            ask.click(on_ask, [img, q], [ans, st])
            q.submit(on_ask, [img, q], [ans, st])
        with gr.Tab("2 · Résultats"):
            res = gr.HTML(results_html(), elem_id="results")
            gr.Button("Actualiser", elem_id="refresh").click(results_html, None, res)
        with gr.Tab("3 · Code"):
            gr.Code(CODE, language="python", elem_id="code")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, css=CSS, js=LIGHT_ONLY, theme=gr.themes.Base())
