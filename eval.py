"""Évaluation de LLaVA-1.5-7B sur la photo COCO 53236.

Chaque question a une réponse attendue, vérifiée à la main sur la photo. Une réponse est jugée
correcte si elle contient l'un des mots-clés acceptés et ne commence pas par un mot refusé
(utile pour les questions oui / non). Les variantes de formulation ne sont pas notées : elles
servent à comparer les réponses.

    python eval.py        ->  results.json (affiché dans l'onglet « Résultats » de app.py)
"""
import json
from pathlib import Path

from PIL import Image

from llava_core import MODEL_ID, LLaVA

HERE = Path(__file__).parent
PHOTO = HERE / "assets" / "coco-53236.jpg"

# (catégorie, question, réponse attendue, mots-clés acceptés, premiers mots refusés)
QUESTIONS = [
    ("VQA", "What type of table is the food on?", "une table en bois", ["wood"], []),
    ("VQA", "What fruit is in the bowl on the left?", "du raisin", ["grape"], []),
    ("VQA", "What color is the flower?", "jaune", ["yellow", "orange"], []),
    ("VQA", "What food is on the rectangular plate on the right?", "des croissants", ["croissant"], []),
    ("VQA", "What is on top of the cake pieces?", "des fraises", ["strawberr"], []),
    ("VQA", "How many pieces of cake are on the blue plate? Answer with a number.", "6", ["6", "six"], []),
    ("Oui / non", "Is there a cup on the table? Answer yes or no.", "oui", ["yes"], ["no"]),
    ("Oui / non", "Is there a bottle of Coca-Cola on the table? Answer yes or no.", "non", ["no"], ["yes"]),
    ("Oui / non", "Is there a dog in the image? Answer yes or no.", "non", ["no"], ["yes"]),
    ("Oui / non", "Is there a laptop on the table? Answer yes or no.", "non", ["no"], ["yes"]),
    ("Raisonnement", "What meal is this table most likely set for?", "petit-déjeuner ou brunch", ["breakfast", "brunch"], []),
    ("Raisonnement", "Which object on the table is used to make coffee?", "la cafetière italienne",
     ["moka", "coffee maker", "coffee pot", "espresso", "percolator"], []),
    ("Détail fin", "What brand is the coffee maker?", "illisible sur la photo",
     ["not", "cannot", "can't", "unclear", "unknown", "unable", "difficult"], []),
]

# Le même fait, demandé autrement : sensibilité aux consignes
VARIANTS = [
    ("Question orientée", "Is there a Coca-Cola on the table?"),
    ("Question orientée", "How many cans of Coca-Cola are on the table?"),
    ("Question orientée", "What is the dog eating?"),
    ("Format", "Describe the image in one sentence."),
    ("Format", "Answer with a single word: what fruit is in the bowl?"),
    ("Langue", "Décris l'image en une phrase."),
    ("Langue", "Quel fruit y a-t-il dans le bol ?"),
]


def score(answer: str, accepted: list[str], refused_first: list[str]) -> bool:
    text = answer.lower()
    words = text.split()
    first = words[0].strip(".,!") if words else ""
    if first in refused_first:
        return False
    return any(k in text for k in accepted)


def main():
    llava = LLaVA()
    image = Image.open(PHOTO).convert("RGB")

    rows = []
    for cat, question, expected, accepted, refused in QUESTIONS:
        answer = llava.ask(image, question, max_new_tokens=80)
        ok = score(answer, accepted, refused)
        rows.append({"cat": cat, "q": question, "expected": expected, "answer": answer, "correct": ok})
        print("OK " if ok else "ERR", question, "->", answer)

    variants = []
    for cat, question in VARIANTS:
        answer = llava.ask(image, question, max_new_tokens=160)
        variants.append({"cat": cat, "q": question, "answer": answer})
        print("VAR", question, "->", answer)

    n = sum(r["correct"] for r in rows)
    result = {"model": MODEL_ID, "photo": "COCO 53236", "score": [n, len(rows)], "rows": rows, "variants": variants}
    (HERE / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"score {n}/{len(rows)}")


if __name__ == "__main__":
    main()
