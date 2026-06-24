"""Down-sample images and prune text before routing and inference."""

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from my_routing_agent.config import CompressorConfig
from my_routing_agent.utils.tokenizer import TokenCounter


SYSTEM_FLUFF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^#{1,6}\s+", re.MULTILINE),
    re.compile(r"^\s*[-*]\s+", re.MULTILINE),
    re.compile(r"\b(?:please|kindly|note that|for your information|fyi)\b", re.IGNORECASE),
    re.compile(r"\b(?:as an ai(?: language model)?|i(?:'m| am) an ai)\b", re.IGNORECASE),
    re.compile(r"```[\s\S]*?```"),
    re.compile(r"<system>[\s\S]*?</system>", re.IGNORECASE),
)

WHITESPACE_PATTERN = re.compile(r"[ \t]+")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


@dataclass
class ProcessedInput:
    """Normalized payload ready for routing and inference."""

    text: str
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    pre_optimization_tokens: int = 0
    post_optimization_tokens: int = 0

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def is_multimodal(self) -> bool:
        return self.has_images


class InputCompressor:
    """CPU-only preprocessor for text pruning and image compression."""

    SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

    def __init__(
        self,
        config: CompressorConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config or CompressorConfig()
        self._tokens = token_counter or TokenCounter()

    def process(
        self,
        text: str = "",
        *,
        image_paths: list[str | Path] | None = None,
        image_data: list[str | bytes] | None = None,
    ) -> ProcessedInput:
        raw_text = text or ""
        pre_tokens = self._tokens.count(raw_text)

        cleaned_text = self._prune_text(raw_text)
        if self._config.max_text_chars and len(cleaned_text) > self._config.max_text_chars:
            cleaned_text = cleaned_text[: self._config.max_text_chars].rstrip() + "…"

        images: list[str] = []
        image_meta: list[dict[str, Any]] = []

        for path in image_paths or []:
            encoded, meta = self._compress_image_file(Path(path))
            images.append(encoded)
            image_meta.append(meta)

        for item in image_data or []:
            encoded, meta = self._compress_image_bytes(self._coerce_bytes(item))
            images.append(encoded)
            image_meta.append(meta)

        post_tokens = self._tokens.count(cleaned_text) + len(images) * 85

        return ProcessedInput(
            text=cleaned_text,
            images=images,
            metadata={"images": image_meta},
            pre_optimization_tokens=pre_tokens,
            post_optimization_tokens=post_tokens,
        )

    def _prune_text(self, text: str) -> str:
        result = text.strip()
        if self._config.strip_system_fluff:
            for pattern in SYSTEM_FLUFF_PATTERNS:
                result = pattern.sub(" ", result)
        if self._config.collapse_whitespace:
            result = WHITESPACE_PATTERN.sub(" ", result)
            result = MULTI_NEWLINE_PATTERN.sub("\n\n", result)
        return result.strip()

    def _compress_image_file(self, path: Path) -> tuple[str, dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            return self._compress_pil_image(img, source=str(path))

    def _compress_image_bytes(self, data: bytes) -> tuple[str, dict[str, Any]]:
        with Image.open(io.BytesIO(data)) as img:
            return self._compress_pil_image(img, source="bytes")

    def _compress_pil_image(self, image: Image.Image, *, source: str) -> tuple[str, dict[str, Any]]:
        original_size = image.size
        resized = self._downsample(image)
        buffer = io.BytesIO()
        fmt = "JPEG" if resized.mode in {"RGB", "L"} else "PNG"
        save_kwargs: dict[str, Any] = {"optimize": self._config.png_optimize}
        if fmt == "JPEG":
            if resized.mode != "RGB":
                resized = resized.convert("RGB")
            save_kwargs["quality"] = self._config.jpeg_quality
        resized.save(buffer, format=fmt, **save_kwargs)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return encoded, {
            "source": source,
            "original_size": original_size,
            "compressed_size": resized.size,
            "format": fmt,
            "bytes": len(buffer.getvalue()),
        }

    def _downsample(self, image: Image.Image) -> Image.Image:
        max_dim = self._config.max_image_dimension
        width, height = image.size
        if max(width, height) <= max_dim:
            return image.copy()
        scale = max_dim / float(max(width, height))
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def _coerce_bytes(data: str | bytes) -> bytes:
        if isinstance(data, bytes):
            return data
        if data.startswith("data:"):
            _, _, payload = data.partition(",")
            return base64.b64decode(payload)
        try:
            return base64.b64decode(data, validate=True)
        except Exception:
            return data.encode("utf-8")
