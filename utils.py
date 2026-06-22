"""
APVD image and tensor utilities.
"""
from __future__ import annotations

import difflib
import math
import random
import re
import tarfile
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", "*.mkv", ".webm", ".m4v", ".wmv"}
MODEL_EXTENSIONS = {".pt", ".pth"}
DEFAULT_TARGET_SIZE = (256, 256)
GENERIC_PROMPT_WORDS = {
    "a", "an", "the", "some", "random", "show", "me", "something",
    "just", "any", "please", "model", "models", "image", "images",
    "picture", "pictures", "of",
}
CATEGORY_ALIASES = {
    "game": {"game", "games"},
    "object": {"object", "objects"},
    "person": {"person", "people", "persons"},
    "place": {"place", "places"},
}

def _archive_format(path: Path) -> str:
    name_lower = path.name.lower()
    if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
        return "tar"
    if path.suffix.lower() == ".tar":
        return "tar"
    if path.suffix.lower() == ".zip":
        return "zip"
    raise ValueError(f"Unsupported archive type: {path}")

def _is_safe_archive_member(member_name: str) -> bool:
    norm = member_name.replace("\\", "/").strip("/")
    if not norm:
        return False
    parts = norm.split("/")
    if ".." in parts:
        return False
    if parts[0].endswith(":"):
        return False
    return True

def _skip_macosx_path(member_name: str) -> bool:
    parts = member_name.replace("\\", "/").split("/")
    return any(p == "__MACOSX" for p in parts)

def _member_is_image_file(member_name: str) -> bool:
    if not member_name or member_name.endswith("/"):
        return False
    return Path(member_name).suffix.lower() in IMAGE_EXTENSIONS

def list_image_members(archive: Path) -> list[str]:
    archive = Path(archive)
    if not archive.is_file():
        return []
    try:
        fmt = _archive_format(archive)
    except ValueError:
        return []
    out: list[str] = []
    if fmt == "zip":
        with zipfile.ZipFile(archive, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if _skip_macosx_path(name) or not _is_safe_archive_member(name):
                    continue
                if _member_is_image_file(name):
                    out.append(name)
    else:
        with tarfile.open(archive, "r:*") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if _skip_macosx_path(name) or not _is_safe_archive_member(name):
                    continue
                if _member_is_image_file(name):
                    out.append(name)
    return sorted(out)

def load_training_images_from_archive_entries(
    entries: Iterable[tuple[Path, str]],
    device: torch.device | None = None,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"Invalid target size: {target_size}")
    entries_list = [(Path(a), m) for a, m in entries]
    if not entries_list:
        raise ValueError("No archive entries provided.")
    total = len(entries_list)
    by_arch: dict[Path, list[str]] = defaultdict(list)
    arch_order: list[Path] = []
    for ap, mem in entries_list:
        if ap not in by_arch:
            arch_order.append(ap)
        by_arch[ap].append(mem)
    tensors: list[torch.Tensor] = []
    bad_files = 0
    for archive_path in arch_order:
        members = by_arch[archive_path]
        fmt = _archive_format(archive_path)
        if fmt == "zip":
            try:
                with zipfile.ZipFile(archive_path, "r") as zf:
                    for member in members:
                        if not _is_safe_archive_member(member) or _skip_macosx_path(member):
                            bad_files += 1
                            continue
                        try:
                            with zf.open(member, "r") as f:
                                data = f.read()
                            with Image.open(BytesIO(data)) as img:
                                tensors.append(_pil_to_tensor(img, target_size=target_size))
                            if progress_callback is not None and total > 0:
                                progress_callback(len(tensors), total)
                        except Exception:
                            bad_files += 1
            except Exception:
                bad_files += len(members)
        else:
            try:
                with tarfile.open(archive_path, "r:*") as tf:
                    for member in members:
                        if not _is_safe_archive_member(member) or _skip_macosx_path(member):
                            bad_files += 1
                            continue
                        try:
                            info = tf.getmember(member)
                            if not info.isfile():
                                bad_files += 1
                                continue
                            reader = tf.extractfile(info)
                            if reader is None:
                                bad_files += 1
                                continue
                            data = reader.read()
                            with Image.open(BytesIO(data)) as img:
                                tensors.append(_pil_to_tensor(img, target_size=target_size))
                            if progress_callback is not None and total > 0:
                                progress_callback(len(tensors), total)
                        except Exception:
                            bad_files += 1
            except Exception:
                bad_files += len(members)
    if not tensors:
        raise ValueError("No valid images could be loaded from the selected archive(s).")
    batch = torch.stack(tensors, dim=0)
    if device is not None:
        batch = batch.to(device)
    if bad_files:
        print(f"Warning: skipped {bad_files} unreadable or unsafe archive member(s).")
    return batch

def get_image_paths(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)

def list_model_paths(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS]
    return sorted(paths)

def preprocess_prompt(prompt: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", prompt.lower())

def clean_model_filename(path: Path) -> list[str]:
    name = path.stem.lower()
    name = re.sub(r"(_?model|_?checkpoint|_?ckpt)$", "", name)
    name = name.replace("-", "_")
    return [part for part in re.split(r"[^a-z0-9]+", name) if part]

def score_model_filename(
    model_path: Path,
    prompt_words: Iterable[str],
    fuzzy_cutoff: float = 0.82,
) -> float:
    filename_words = clean_model_filename(model_path)
    prompt_list = [word for word in prompt_words if word]
    if not filename_words or not prompt_list:
        return 0.0
    score = 0.0
    remaining_prompt_words = set(prompt_list)
    for filename_word in filename_words:
        if filename_word in remaining_prompt_words:
            score += 1.0
            remaining_prompt_words.remove(filename_word)
            continue
        close_match = difflib.get_close_matches(
            filename_word, list(remaining_prompt_words), n=1, cutoff=fuzzy_cutoff
        )
        if close_match:
            score += 0.5
            remaining_prompt_words.discard(close_match[0])
    return score

def select_best_model_path(
    models_folder: Path,
    prompt: str,
    fuzzy_cutoff: float = 0.82,
) -> tuple[Path, list[tuple[Path, float]]]:
    model_paths = list_model_paths(models_folder)
    if not model_paths:
        raise FileNotFoundError(f"No model files found in: {models_folder}")
    prompt_words = preprocess_prompt(prompt)
    if not prompt_words:
        raise ValueError("Prompt must contain at least one word or number.")
    ranked = sorted(
        (
            (path, score_model_filename(path, prompt_words, fuzzy_cutoff=fuzzy_cutoff))
            for path in model_paths
        ),
        key=lambda item: (item[1], item[0].name.lower()),
        reverse=True,
    )
    return ranked[0][0], ranked

def _detect_category_prompt(prompt_words: Iterable[str]) -> str | None:
    meaningful_words = [word for word in prompt_words if word not in GENERIC_PROMPT_WORDS]
    if not meaningful_words:
        return None
    for canonical_name, aliases in CATEGORY_ALIASES.items():
        if all(word in aliases for word in meaningful_words):
            return canonical_name
    return None

def _tokenize_label(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", value.lower()) if token}

def _expand_match_words(words: Iterable[str]) -> set[str]:
    expanded: set[str] = set()
    for word in words:
        if not word:
            continue
        expanded.add(word)
        if len(word) > 3 and word.endswith("s"):
            expanded.add(word[:-1])
        else:
            expanded.add(f"{word}s")
    return expanded

def _iter_category_folders(models_folder: Path) -> list[Path]:
    if not models_folder.exists() or not models_folder.is_dir():
        return []
    return sorted((path for path in models_folder.rglob("*") if path.is_dir()), key=lambda path: (len(path.parts), str(path).lower()))

def _find_category_folder(models_folder: Path, canonical_name: str) -> Path | None:
    if not models_folder.exists() or not models_folder.is_dir():
        return None
    allowed_names = {canonical_name, *CATEGORY_ALIASES.get(canonical_name, set())}
    allowed_names = _expand_match_words({name.lower() for name in allowed_names})
    for child in _iter_category_folders(models_folder):
        folder_words = _expand_match_words(_tokenize_label(child.name))
        if folder_words & allowed_names:
            return child
    return None

def _find_prompt_folder(models_folder: Path, prompt_words: Iterable[str]) -> Path | None:
    meaningful_words = [word for word in prompt_words if word not in GENERIC_PROMPT_WORDS]
    if not meaningful_words:
        return None
    prompt_variants = _expand_match_words(meaningful_words)
    best_match: tuple[int, int, int, Path] | None = None
    for folder in _iter_category_folders(models_folder):
        relative_words = _tokenize_label(" ".join(folder.relative_to(models_folder).parts))
        if not relative_words:
            continue
        folder_variants = _expand_match_words(relative_words)
        matched_words = sum(1 for word in meaningful_words if word in folder_variants)
        if matched_words != len(meaningful_words):
            continue
        exact_bonus = sum(1 for word in meaningful_words if word in relative_words)
        variant_bonus = sum(1 for word in prompt_variants if word in folder_variants)
        candidate = (matched_words, exact_bonus, variant_bonus + len(folder.parts), folder)
        if best_match is None or candidate > best_match:
            best_match = candidate
    return None if best_match is None else best_match[3]

def select_model_path_for_prompt(
    models_folder: Path,
    prompt: str,
    fuzzy_cutoff: float = 0.82,
) -> tuple[Path, str]:
    prompt_words = preprocess_prompt(prompt)
    if not prompt_words:
        raise ValueError("Prompt must contain at least one word or number.")
    prompt_folder = _find_prompt_folder(models_folder, prompt_words)
    if prompt_folder is not None:
        prompt_models = list_model_paths(prompt_folder)
        if not prompt_models:
            raise FileNotFoundError(f"No model files found in: {prompt_folder}")
        chosen_model = random.choice(prompt_models)
        return chosen_model, f"Random {prompt_folder.relative_to(models_folder)} model"
    category_name = _detect_category_prompt(prompt_words)
    if category_name:
        category_folder = _find_category_folder(models_folder, category_name)
        if category_folder is None:
            raise FileNotFoundError(f"No folder found for category '{category_name}' in: {models_folder}")
        category_models = list_model_paths(category_folder)
        if not category_models:
            raise FileNotFoundError(f"No model files found in: {category_folder}")
        chosen_model = random.choice(category_models)
        return chosen_model, f"Random {category_folder.name} model"
    best_model_path, ranked_models = select_best_model_path(models_folder, prompt, fuzzy_cutoff=fuzzy_cutoff)
    best_score = ranked_models[0][1]
    return best_model_path, f"Best filename match (score {best_score:.2f})"

def normalize_apvd_image(image: Image.Image, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    elif image.mode in {"RGBA", "LA"}:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (*background, 255))
        bg.alpha_composite(image)
        image = bg.convert("RGB")
    return image

def _pil_to_tensor(image: Image.Image, target_size: tuple[int, int]) -> torch.Tensor:
    rgb = normalize_apvd_image(image).resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

def load_training_images_from_paths(
    paths: Iterable[Path],
    device: torch.device | None = None,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"Invalid target size: {target_size}")
    paths_list = list(paths)
    total = len(paths_list)
    tensors: list[torch.Tensor] = []
    bad_files = 0
    for idx, path in enumerate(paths_list):
        try:
            with Image.open(path) as img:
                tensors.append(_pil_to_tensor(img, target_size=target_size))
            if progress_callback is not None and total > 0:
                progress_callback(len(tensors), total)
        except Exception:
            bad_files += 1
    if not tensors:
        raise ValueError("No valid images could be loaded from the selected paths.")
    batch = torch.stack(tensors, dim=0)
    if device is not None:
        batch = batch.to(device)
    if bad_files:
        print(f"Warning: skipped {bad_files} unreadable image file(s).")
    return batch

def _estimate_video_sample_count(path: Path, frame_stride: int) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        return 0
    s = max(1, frame_stride)
    return max(0, (n + s - 1) // s)

def load_training_images_from_videos(
    video_paths: Iterable[Path],
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    frame_stride: int = 30,
    max_frames: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"Invalid target size: {target_size}")
    paths_list = [Path(p) for p in video_paths]
    if not paths_list:
        raise ValueError("No video paths provided.")
    stride = max(1, int(frame_stride))
    tensors: list[torch.Tensor] = []
    total_est = sum(_estimate_video_sample_count(p, stride) for p in paths_list)
    if max_frames is not None:
        cap_m = max(0, int(max_frames))
        if total_est > 0:
            total_est = min(total_est, cap_m)
        else:
            total_est = cap_m if cap_m > 0 else 1
    elif total_est <= 0:
        total_est = 1

    for vpath in paths_list:
        if max_frames is not None and len(tensors) >= max_frames:
            break
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            continue
        local_i = 0
        while True:
            if max_frames is not None and len(tensors) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            if local_i % stride != 0:
                local_i += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(_pil_to_tensor(pil, target_size=target_size))
            if progress_callback is not None:
                progress_callback(len(tensors), max(total_est, len(tensors)))
            local_i += 1
        cap.release()

    if not tensors:
        raise ValueError("No frames could be read from the selected video(s).")
    batch = torch.stack(tensors, dim=0)
    return batch

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert BCHW/CHW tensor to a PIL RGB image.

    RGB tensors are expected in [0, 1]. Wavelet tensors with 12 channels are
    reconstructed back to RGB before clamping for display.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}")

    chw = torch.nan_to_num(
        tensor.detach().float().cpu(),
        nan=0.0,
        posinf=4.0,
        neginf=-4.0,
    )
    if chw.size(0) == 12:
        chw = wavelet_to_rgb(chw)
    elif chw.size(0) > 3:
        chw = chw[:3]

    chw = torch.nan_to_num(chw, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    arr = (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

# --- Wavelet Utilities ---

# --- Wavelet Utilities ---

# --- Wavelet Utilities ---

def _haar_decompose_2d(x: torch.Tensor):
    if x.size(-2) % 2 != 0 or x.size(-1) % 2 != 0:
        pad_h = x.size(-2) % 2
        pad_w = x.size(-1) % 2
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    x_even = x[..., ::2, :]
    x_odd = x[..., 1::2, :]
    L = (x_even + x_odd) * inv_sqrt2
    H = (x_even - x_odd) * inv_sqrt2

    LL = (L[..., :, ::2] + L[..., :, 1::2]) * inv_sqrt2
    LH = (L[..., :, ::2] - L[..., :, 1::2]) * inv_sqrt2
    HL = (H[..., :, ::2] + H[..., :, 1::2]) * inv_sqrt2
    HH = (H[..., :, ::2] - H[..., :, 1::2]) * inv_sqrt2
    return LL, LH, HL, HH


def _haar_reconstruct_2d(LL, LH, HL, HH):
    if not (LL.shape == LH.shape == HL.shape == HH.shape):
        raise ValueError("Wavelet bands must all have the same shape.")

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    half_h, half_w = LL.size(-2), LL.size(-1)
    full_h, full_w = half_h * 2, half_w * 2

    L = LL.new_empty(*LL.shape[:-1], full_w)
    L[..., :, ::2] = (LL + LH) * inv_sqrt2
    L[..., :, 1::2] = (LL - LH) * inv_sqrt2

    H = LL.new_empty(*LL.shape[:-1], full_w)
    H[..., :, ::2] = (HL + HH) * inv_sqrt2
    H[..., :, 1::2] = (HL - HH) * inv_sqrt2

    x = LL.new_empty(*LL.shape[:-2], full_h, full_w)
    x[..., ::2, :] = (L + H) * inv_sqrt2
    x[..., 1::2, :] = (L - H) * inv_sqrt2
    return x


def rgb_to_wavelet(tensor: torch.Tensor) -> torch.Tensor:
    is_3d = tensor.ndim == 3
    if is_3d:
        tensor = tensor.unsqueeze(0)
    LL, LH, HL, HH = _haar_decompose_2d(tensor)
    wavelet_tensor = torch.cat([LL, LH, HL, HH], dim=1)
    if is_3d:
        wavelet_tensor = wavelet_tensor.squeeze(0)
    return wavelet_tensor


def wavelet_to_rgb(tensor: torch.Tensor) -> torch.Tensor:
    is_3d = tensor.ndim == 3
    if is_3d:
        tensor = tensor.unsqueeze(0)
    LL = tensor[:, 0:3, :, :]
    LH = tensor[:, 3:6, :, :]
    HL = tensor[:, 6:9, :, :]
    HH = tensor[:, 9:12, :, :]
    rgb_tensor = _haar_reconstruct_2d(LL, LH, HL, HH)
    if is_3d:
        rgb_tensor = rgb_tensor.squeeze(0)
    return rgb_tensor