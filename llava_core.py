"""LLaVA-1.5-7B : chargement du modèle, question-réponse sur image et « coulisses ».

Utilisé par app.py (interface web) et eval.py (évaluation).
Modèle : llava-hf/llava-1.5-7b-hf, en float16 sur GPU (environ 14 Go de mémoire).
"""
from __future__ import annotations

import time
from threading import Thread

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, TextIteratorStreamer

MODEL_ID = "llava-hf/llava-1.5-7b-hf"


class LLaVA:
    def __init__(self, model_id: str = MODEL_ID):
        # Téléchargé depuis Hugging Face au premier lancement, puis lu depuis le cache local.
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16, device_map="cuda"
        ).eval()
        self.image_token_id = self.model.config.image_token_id
        # Mémorise les dimensions en entrée et en sortie de la projection (CLIP -> LLM).
        self._shapes = {}
        self.model.model.multi_modal_projector.register_forward_hook(self._record_shapes)

    def _record_shapes(self, module, inputs, output):
        self._shapes = {"in": tuple(inputs[0].shape), "out": tuple(output.shape)}

    # ------------------------------------------------------------------ prompt
    def prompt(self, question: str) -> str:
        """Format de conversation de LLaVA-1.5 : « USER: <image>\\nquestion ASSISTANT: »."""
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True)

    def inputs(self, image: Image.Image, question: str):
        """Image redimensionnée à 336 × 336 et balise <image> remplacée par 576 tokens visuels."""
        return self.processor(images=image, text=self.prompt(question), return_tensors="pt").to("cuda", torch.float16)

    # ------------------------------------------------------------------ réponses
    @torch.inference_mode()
    def ask(self, image: Image.Image, question: str, max_new_tokens: int = 200) -> str:
        """Réponse complète, en décodage glouton (reproductible)."""
        inputs = self.inputs(image, question)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def stream(self, image: Image.Image, question: str, max_new_tokens: int = 200):
        """Produit (réponse partielle, None) token par token, puis (réponse, statistiques)."""
        inputs = self.inputs(image, question)
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
        t0 = time.perf_counter()
        thread = Thread(target=self._generate, args=(kwargs,))
        thread.start()
        text = ""
        for piece in streamer:
            text += piece
            yield text.strip(), None
        thread.join()
        seconds = time.perf_counter() - t0
        n_new = len(self.processor.tokenizer(text, add_special_tokens=False)["input_ids"])
        yield text.strip(), self.stats(inputs, question, n_new, seconds)

    @torch.inference_mode()
    def _generate(self, kwargs):
        self.model.generate(**kwargs)

    # ------------------------------------------------------------------ coulisses
    def stats(self, inputs, question: str, n_new: int, seconds: float) -> dict:
        """Ce que le modèle a réellement reçu et produit pour une question."""
        ids = inputs["input_ids"][0]
        n_img = int((ids == self.image_token_id).sum())
        cfg = self.model.config
        return {
            "pixel_values": tuple(inputs["pixel_values"].shape),
            "image_tokens": n_img,
            "text_tokens": int(ids.numel()) - n_img,
            "projector_in": self._shapes.get("in"),
            "projector_out": self._shapes.get("out"),
            "vision_layer": cfg.vision_feature_layer,
            "prompt": self.prompt(question),
            "new_tokens": n_new,
            "seconds": seconds,
        }
