"""Benchmark loading and multimodal prompt construction.

FROZEN CONTRACT: ``SYSTEM_PROMPT`` and ``build_multimodal_messages`` are copied
verbatim from the research pipeline. Changing either breaks reproduction of the
published numbers.

Each request is a single user turn:
    [system prompt]
    === Context i (Person i) === [image] [audio] text     for i = 0..3
    === Query ===                [image] [audio] query_text
"""

import base64
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

SCENARIOS = (
    "visual_identity",
    "voice_identity",
    "same_modal_semantic",
    "cross_modal_semantic_bridge",
)

SYSTEM_PROMPT = """You are a personal memory assistant for a multimodal personalization benchmark.

You will receive memory contexts about four different people (Context 0 through Context 3). Each context may contain some combination of:
- A photo of the person (image)
- A voice recording or spoken clip of the person (audio)
- A text description with personal details such as name, job, hobbies, personality, location, etc.

After the four memory contexts, you will receive a Query section that may include:
- A query image showing a person (use it to identify which context person this is)
- A query audio clip of a person speaking (use it to identify which context person this is)
- A question asking for specific information about that person

Your task:
1. Carefully examine ALL four contexts, attending to every modality (image, audio, text).
2. Use the query image or audio to identify which context (Person 0–3) the query is referring to.
   - If the query includes an image: match the face/appearance to the context images.
   - If the query includes an audio clip: match the voice to the context audio recordings.
   - If the query is text-only: use the semantic description to identify the correct person.
3. Answer the question using ONLY the information from the matching context.

Response rules:
- Before answering, briefly reason through which person matches the query and what relevant information their context contains.
- Provide a specific, informative answer — include all relevant details available in the context (e.g. name, job, location, hobby as appropriate). Aim for 1–3 sentences.
- Base your answer solely on what is stated in the provided contexts — do not hallucinate or infer beyond what is given.
- Do not repeat the question.
- Use "I cannot determine that from the provided context." ONLY as a last resort — when the specific information asked is genuinely absent from ALL modalities (image, audio, and text) of the matching person's context. If any modality provides a partial or indirect answer, use it. Do not abstain simply because you are uncertain or the context is ambiguous."""


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load the benchmark file. Accepts the .json array or the .jsonl twin."""
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stratified_subset(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Deterministic subset covering all 4 scenarios x {answerable, unanswerable}.

    Round-robins over the 8 strata (in benchmark order) so a small ``limit``
    still exercises every scenario and both answerability classes.
    """
    if limit <= 0 or limit >= len(items):
        return items
    strata: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        answerable = bool(it.get("answer_supervision", {}).get("answerable_from_context"))
        strata[(it.get("scenario"), answerable)].append(it)
    keys = sorted(strata, key=lambda k: (SCENARIOS.index(k[0]) if k[0] in SCENARIOS else 99, not k[1]))
    picked: List[Dict[str, Any]] = []
    idx = 0
    while len(picked) < limit:
        added = False
        for k in keys:
            if idx < len(strata[k]):
                picked.append(strata[k][idx])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        idx += 1
    order = {id(it): i for i, it in enumerate(items)}
    return sorted(picked, key=lambda it: order[id(it)])


def _encode(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_content(b64: str) -> Dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def audio_content(b64: str) -> Dict:
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


def text_content(text: str) -> Dict:
    return {"type": "text", "text": text}


def build_multimodal_messages(
    item: Dict[str, Any],
    asset_root: Path,
    skip_audio: bool = False,
    max_images: Optional[int] = None,
) -> List[Dict]:
    """Build the messages list with interleaved image/audio/text per concept.

    Args:
        item: one benchmark record.
        asset_root: directory holding ``sample_<id>/concept_<k>.png|wav``.
        skip_audio: Drop all audio content blocks. Use for models that only
                    accept text/image (e.g. gpt-4o without audio preview).
        max_images: Hard cap on total image content blocks. When set, context images
                    are included in order and one slot is reserved for the query image
                    (if the query has one). Use max_images=4 for Phi-4-multimodal.

    Missing assets (None path or non-existent file) are silently skipped.
    """
    content: List[Dict] = []

    # Pre-compute how many context image slots are available
    # voice_identity task: query identification is done via audio only — no query image
    is_voice_identity = item.get("query_modality") == "voice"
    has_query_image = bool(item.get("query_assets", {}).get("image")) and not is_voice_identity
    if max_images is not None:
        context_image_slots = max(0, max_images - (1 if has_query_image else 0))
    else:
        context_image_slots = None  # unlimited

    context_images_used = 0

    for i, ctx in enumerate(item["contexts"]):
        # Section header comes FIRST so the model knows which person the following
        # image/audio/text belong to before it encounters the modality tokens.
        content.append(text_content(f"=== Context {i} (Person {i}) ==="))

        img_rel = ctx.get("image")
        if img_rel:
            if context_image_slots is None or context_images_used < context_image_slots:
                b64 = _encode(asset_root / img_rel)
                if b64:
                    content.append(image_content(b64))
                    context_images_used += 1

        if not skip_audio:
            aud_rel = ctx.get("audio")
            if aud_rel:
                b64 = _encode(asset_root / aud_rel)
                if b64:
                    content.append(audio_content(b64))

        txt = ctx.get("text", "").strip()
        if txt:
            content.append(text_content(txt))

    content.append(text_content("=== Query ==="))

    # query assets — image or audio that identifies the target person
    # voice query: skip query image so the model cannot cheat via face matching
    q_img_rel = item.get("query_assets", {}).get("image")
    if q_img_rel and not is_voice_identity:
        b64 = _encode(asset_root / q_img_rel)
        if b64:
            content.append(image_content(b64))

    if not skip_audio:
        q_aud_rel = item.get("query_assets", {}).get("audio")
        if q_aud_rel:
            b64 = _encode(asset_root / q_aud_rel)
            if b64:
                content.append(audio_content(b64))

    content.append(text_content(item["query_text"].strip()))

    # Prepend system prompt as the first text block in the user message,
    # matching the training data format.
    user_content = [text_content(SYSTEM_PROMPT)] + content

    return [{"role": "user", "content": user_content}]
