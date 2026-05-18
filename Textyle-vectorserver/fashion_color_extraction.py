from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


DENIM_CONTEXT_TERMS = (
    "denim",
    "jean",
    "jeans",
    "raw denim",
    "dark denim",
    "\ub370\ub2d8",
    "\uccad\ubc14\uc9c0",
    "\ud751\uccad",
    "\uc9c4\uccad",
    "\uc911\uccad",
    "\uc5f0\uccad",
)

PATTERN_CONTEXT_TERMS = (
    "check",
    "checked",
    "checkered",
    "plaid",
    "tartan",
    "gingham",
    "stripe",
    "striped",
    "pattern",
    "patterns",
    "dot",
    "dotted",
    "\uccb4\ud06c",
    "\uccb4\ud06c\ubb34\ub2ac",
    "\uc2a4\ud2b8\ub77c\uc774\ud504",
    "\uc904\ubb34\ub2ac",
    "\ud328\ud134",
    "\ub3c4\ud2b8",
)


@dataclass
class ColorExtractionResult:
    color: str = ""
    confidence: str = "low"
    reason: str = "grounding_dino_sam_not_implemented"
    dominant_ratio: float = 0.0
    second_ratio: float = 0.0
    candidates: list[dict[str, Any]] = field(default_factory=list)
    secondary_colors: list[str] = field(default_factory=list)
    is_mixed_color: bool = False
    pattern: str = ""
    search_color_weights: dict[str, float] = field(default_factory=dict)
    segmentation_used: bool = False
    candidate_pixel_count: int = 0
    ignored_pixel_count: int = 0
    background_pixel_count: int = 0


def _compact_text(*values: Any) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    compact = []
    for char in text:
        if char.isascii() and char.isalnum():
            compact.append(char)
        elif "\uac00" <= char <= "\ud7a3":
            compact.append(char)
    return "".join(compact)


def is_denim_context_from_text(*values: Any) -> bool:
    compact_text = _compact_text(*values)
    return any(_compact_text(term) in compact_text for term in DENIM_CONTEXT_TERMS)


def should_run_pattern_classifier(text: str) -> bool:
    compact_text = _compact_text(text)
    return any(_compact_text(term) in compact_text for term in PATTERN_CONTEXT_TERMS)


def extract_dominant_color_result(
    image_obj: Image.Image,
    denim_context: bool = False,
    pattern_context_text: str = "",
    product_id: str = "",
    product_name: str = "",
    segmentation_mask: Image.Image | None = None,
) -> ColorExtractionResult:
    del image_obj, denim_context, pattern_context_text, product_id, product_name, segmentation_mask
    return ColorExtractionResult()


def extract_dominant_color(
    image_obj: Image.Image,
    denim_context: bool = False,
    pattern_context_text: str = "",
    product_id: str = "",
    product_name: str = "",
) -> str:
    result = extract_dominant_color_result(
        image_obj,
        denim_context=denim_context,
        pattern_context_text=pattern_context_text,
        product_id=product_id,
        product_name=product_name,
    )
    return result.color
