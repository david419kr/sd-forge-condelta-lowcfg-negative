from __future__ import annotations

import logging
import math
from functools import wraps
from typing import Any

import gradio as gr
import torch

from modules import devices, processing, prompt_parser, scripts, shared


TITLE = "ConDelta low-CFG negative"

DEFAULT_THRESHOLD = 1.0
DEFAULT_STRENGTH = 0.6

SETTINGS_ATTR = "_condelta_low_cfg_negative_settings"
BASE_APPLIED_ATTR = "_condelta_low_cfg_negative_base_marker"
HR_APPLIED_ATTR = "_condelta_low_cfg_negative_hr_marker"
PATCH_ATTR = "_condelta_low_cfg_negative_original"

DELTA_KEYS = ("crossattn", "cross_attn", "vector", "pooled_output")

logger = logging.getLogger(__name__)


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default

    if math.isnan(value) or math.isinf(value):
        value = default

    return min(max(value, minimum), maximum)


def _as_bool(value: Any, default: bool = False) -> bool:
    return default if value is None else bool(value)


def _get_settings(p) -> dict[str, Any]:
    settings = getattr(p, SETTINGS_ATTR, None)
    if not isinstance(settings, dict):
        settings = {}

    return {
        "threshold": _as_float(settings.get("threshold"), DEFAULT_THRESHOLD, 1.0, 24.0),
        "strength": _as_float(settings.get("strength"), DEFAULT_STRENGTH, 0.0, 1.0),
        "also_native": _as_bool(settings.get("also_native"), False),
    }


def _listify_prompts(prompts: Any) -> list[str]:
    if prompts is None:
        return []

    if isinstance(prompts, str):
        return [prompts]

    try:
        return ["" if x is None else str(x) for x in prompts]
    except TypeError:
        return [str(prompts)]


def _has_negative_text(prompts: list[str]) -> bool:
    return any(prompt.strip() for prompt in prompts)


def _is_cfg_one(value: float) -> bool:
    return math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1e-6)


def _max_schedule_step_from_multicond(cond: Any) -> int | None:
    if not isinstance(cond, prompt_parser.MulticondLearnedConditioning):
        return None

    max_step = None
    for composable_prompts in cond.batch:
        for composable_prompt in composable_prompts:
            for schedule in composable_prompt.schedules:
                step = int(schedule.end_at_step)
                max_step = step if max_step is None else max(max_step, step)

    return max_step


def _pass_dimensions(p, is_hr: bool) -> tuple[int | None, int | None]:
    if is_hr:
        return getattr(p, "hr_upscale_to_x", None), getattr(p, "hr_upscale_to_y", None)

    return getattr(p, "width", None), getattr(p, "height", None)


def _pass_distilled_cfg(p, is_hr: bool) -> float | None:
    if is_hr:
        return getattr(p, "hr_distilled_cfg", None)

    return getattr(p, "distilled_cfg_scale", None)


def _pass_steps(p, cond: Any, is_hr: bool) -> tuple[int, int | None]:
    if is_hr:
        hires_steps = _max_schedule_step_from_multicond(cond)
        if hires_steps is None:
            hires_steps = int(getattr(p, "hr_second_pass_steps", 0) or getattr(p, "steps", 1) or 1)

        base_steps = int(getattr(p, "firstpass_steps", 0) or getattr(p, "steps", hires_steps) or hires_steps)
        return max(base_steps, 1), max(int(hires_steps), 1)

    steps = _max_schedule_step_from_multicond(cond)
    if steps is None:
        steps = int(getattr(p, "firstpass_steps", 0) or getattr(p, "steps", 1) or 1)

    return max(int(steps), 1), None


def _model_uses_sdxl_zero_negative(p) -> bool:
    return bool(getattr(getattr(p, "sd_model", None), "is_sdxl", False))


def _encode_prompt_schedules(
    p,
    prompts: list[str],
    steps: int,
    hires_steps: int | None,
    width: int | None,
    height: int | None,
    distilled_cfg_scale: float | None,
    is_negative_prompt: bool,
):
    conditioning = prompt_parser.SdConditioning(
        prompts,
        width=width,
        height=height,
        is_negative_prompt=is_negative_prompt,
        distilled_cfg_scale=distilled_cfg_scale,
    )

    with devices.autocast():
        sd_model = getattr(p, "sd_model", None) or shared.sd_model
        if hasattr(sd_model, "set_clip_skip"):
            sd_model.set_clip_skip(int(shared.opts.CLIP_stop_at_last_layers))
        return prompt_parser.get_learned_conditioning(sd_model, conditioning, int(steps), hires_steps)


def _schedule_at(schedules, end_at_step: int):
    selected = schedules[-1]
    for schedule in schedules:
        if end_at_step <= schedule.end_at_step:
            selected = schedule
            break
    return selected.cond


def _align_tensor_to_target(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    if not torch.is_tensor(source) or not torch.is_tensor(target):
        return None

    tensor = source.to(device=target.device, dtype=target.dtype)

    while tensor.ndim > target.ndim and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    while tensor.ndim < target.ndim and target.shape[0] == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != target.ndim:
        return None

    if tensor.shape == target.shape:
        return tensor

    if tensor.ndim == 0:
        return None

    align_dim = tensor.ndim - 2 if tensor.ndim >= 2 else tensor.ndim - 1

    for dim, (source_size, target_size) in enumerate(zip(tensor.shape, target.shape)):
        if dim == align_dim:
            continue
        if source_size != target_size:
            return None

    source_size = tensor.shape[align_dim]
    target_size = target.shape[align_dim]

    if source_size > target_size:
        return tensor.narrow(align_dim, 0, target_size)

    if source_size < target_size:
        pad_shape = list(tensor.shape)
        pad_shape[align_dim] = target_size - source_size
        padding = tensor.new_zeros(pad_shape)
        return torch.cat([tensor, padding], dim=align_dim)

    return tensor


def _apply_tensor_delta(base: torch.Tensor, negative: torch.Tensor, blank: torch.Tensor, strength: float) -> torch.Tensor:
    negative_aligned = _align_tensor_to_target(negative, base)
    if negative_aligned is None:
        return base

    blank_aligned = _align_tensor_to_target(blank, negative_aligned)
    if blank_aligned is None:
        return base

    return base - (negative_aligned - blank_aligned) * strength


def _apply_conditioning_delta(base: Any, negative: Any, blank: Any, strength: float):
    if torch.is_tensor(base) and torch.is_tensor(negative) and torch.is_tensor(blank):
        return _apply_tensor_delta(base, negative, blank, strength)

    if isinstance(base, dict) and isinstance(negative, dict) and isinstance(blank, dict):
        result = dict(base)

        for key in DELTA_KEYS:
            if key not in base or key not in negative or key not in blank:
                continue

            if not (torch.is_tensor(base[key]) and torch.is_tensor(negative[key]) and torch.is_tensor(blank[key])):
                continue

            result[key] = _apply_tensor_delta(base[key], negative[key], blank[key], strength)

        return result

    return base


def _merge_delta_schedule(base_schedules, negative_schedules, blank_schedules, strength: float):
    endpoints = sorted(
        {
            int(schedule.end_at_step)
            for schedules in (base_schedules, negative_schedules, blank_schedules)
            for schedule in schedules
        }
    )

    return [
        prompt_parser.ScheduledPromptConditioning(
            end_at_step,
            _apply_conditioning_delta(
                _schedule_at(base_schedules, end_at_step),
                _schedule_at(negative_schedules, end_at_step),
                _schedule_at(blank_schedules, end_at_step),
                strength,
            ),
        )
        for end_at_step in endpoints
    ]


def _apply_delta_to_multicond(
    cond: prompt_parser.MulticondLearnedConditioning,
    negative_schedules,
    blank_schedules,
    strength: float,
) -> prompt_parser.MulticondLearnedConditioning:
    batch = []

    for index, composable_prompts in enumerate(cond.batch):
        negative_schedule = negative_schedules[min(index, len(negative_schedules) - 1)]
        blank_schedule = blank_schedules[min(index, len(blank_schedules) - 1)]

        new_composable_prompts = []
        for composable_prompt in composable_prompts:
            schedules = _merge_delta_schedule(composable_prompt.schedules, negative_schedule, blank_schedule, strength)
            new_composable_prompts.append(
                prompt_parser.ComposableScheduledPromptConditioning(
                    schedules=schedules,
                    weight=composable_prompt.weight,
                )
            )

        batch.append(new_composable_prompts)

    return prompt_parser.MulticondLearnedConditioning(cond.shape, batch)


def _marker(cond: Any, uc: Any, cfg: float, settings: dict[str, Any], prompts: list[str]) -> tuple[Any, ...]:
    return (
        id(cond),
        id(uc),
        round(float(cfg), 8),
        round(float(settings["threshold"]), 8),
        round(float(settings["strength"]), 8),
        bool(settings["also_native"]),
        tuple(prompts),
    )


def _record_generation_params(p, cfg: float, settings: dict[str, Any], is_hr: bool, kept_native: bool) -> None:
    prefix = "Hires " if is_hr else ""
    mode = "Native negative + ConDelta" if kept_native else "ConDelta only"

    p.extra_generation_params[f"{prefix}ConDelta low-CFG negative"] = mode
    p.extra_generation_params[f"{prefix}ConDelta CFG threshold"] = settings["threshold"]
    p.extra_generation_params[f"{prefix}ConDelta strength"] = settings["strength"]

    if cfg > 1.0:
        p.extra_generation_params[f"{prefix}ConDelta native negative above CFG 1"] = bool(kept_native)


def _apply_condelta_pass(p, is_hr: bool) -> None:
    settings = _get_settings(p)
    threshold = float(settings["threshold"])
    strength = float(settings["strength"])
    also_native = bool(settings["also_native"])

    cond_attr = "hr_c" if is_hr else "c"
    uc_attr = "hr_uc" if is_hr else "uc"
    prompts_attr = "hr_negative_prompts" if is_hr else "negative_prompts"
    cfg_attr = "hr_cfg" if is_hr else "cfg_scale"
    marker_attr = HR_APPLIED_ATTR if is_hr else BASE_APPLIED_ATTR

    cfg = _as_float(getattr(p, cfg_attr, DEFAULT_THRESHOLD), DEFAULT_THRESHOLD, 0.0, 1000.0)
    negative_prompts = _listify_prompts(getattr(p, prompts_attr, None))

    if cfg > threshold or not _has_negative_text(negative_prompts):
        return

    cond = getattr(p, cond_attr, None)
    if not isinstance(cond, prompt_parser.MulticondLearnedConditioning):
        logger.debug("Skipping ConDelta: unsupported %s conditioning type %s", cond_attr, type(cond).__name__)
        return

    uc = getattr(p, uc_attr, None)
    current_marker = _marker(cond, uc, cfg, settings, negative_prompts)
    if getattr(p, marker_attr, None) == current_marker:
        return

    steps, hires_steps = _pass_steps(p, cond, is_hr)
    width, height = _pass_dimensions(p, is_hr)
    distilled_cfg_scale = _pass_distilled_cfg(p, is_hr)
    blank_prompts = [""] * len(negative_prompts)

    negative_schedules = _encode_prompt_schedules(
        p,
        negative_prompts,
        steps,
        hires_steps,
        width,
        height,
        distilled_cfg_scale,
        is_negative_prompt=True,
    )

    blank_delta_schedules = _encode_prompt_schedules(
        p,
        blank_prompts,
        steps,
        hires_steps,
        width,
        height,
        distilled_cfg_scale,
        is_negative_prompt=not _model_uses_sdxl_zero_negative(p),
    )

    new_cond = _apply_delta_to_multicond(cond, negative_schedules, blank_delta_schedules, strength)
    setattr(p, cond_attr, new_cond)

    kept_native = False
    if _is_cfg_one(cfg):
        setattr(p, uc_attr, None)
    elif also_native:
        kept_native = True
    else:
        blank_uc_schedules = _encode_prompt_schedules(
            p,
            blank_prompts,
            steps,
            hires_steps,
            width,
            height,
            distilled_cfg_scale,
            is_negative_prompt=True,
        )
        setattr(p, uc_attr, blank_uc_schedules)

    _record_generation_params(p, cfg, settings, is_hr, kept_native)
    setattr(p, marker_attr, _marker(new_cond, getattr(p, uc_attr, None), cfg, settings, negative_prompts))


def _patch_processing_methods() -> None:
    setup_conds = processing.StableDiffusionProcessing.setup_conds
    if not getattr(setup_conds, PATCH_ATTR, None):

        @wraps(setup_conds)
        def setup_conds_wrapper(self, *args, **kwargs):
            result = setup_conds(self, *args, **kwargs)
            try:
                _apply_condelta_pass(self, is_hr=False)
            except Exception:
                logger.exception("ConDelta low-CFG negative failed during base conditioning")
            return result

        setattr(setup_conds_wrapper, PATCH_ATTR, setup_conds)
        processing.StableDiffusionProcessing.setup_conds = setup_conds_wrapper

    calculate_hr_conds = processing.StableDiffusionProcessingTxt2Img.calculate_hr_conds
    if not getattr(calculate_hr_conds, PATCH_ATTR, None):

        @wraps(calculate_hr_conds)
        def calculate_hr_conds_wrapper(self, *args, **kwargs):
            result = calculate_hr_conds(self, *args, **kwargs)
            try:
                _apply_condelta_pass(self, is_hr=True)
            except Exception:
                logger.exception("ConDelta low-CFG negative failed during hires conditioning")
            return result

        setattr(calculate_hr_conds_wrapper, PATCH_ATTR, calculate_hr_conds)
        processing.StableDiffusionProcessingTxt2Img.calculate_hr_conds = calculate_hr_conds_wrapper


class Script(scripts.Script):
    def title(self):
        return TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        with gr.Accordion(label=TITLE, open=False, elem_id=f"{tab}_condelta_low_cfg_negative"):
            with gr.Row():
                threshold = gr.Slider(
                    minimum=1.0,
                    maximum=24.0,
                    step=0.5,
                    value=DEFAULT_THRESHOLD,
                    label="Activation CFG threshold",
                    elem_id=f"{tab}_condelta_activation_cfg_threshold",
                    scale=1,
                )

                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=DEFAULT_STRENGTH,
                    label="ConDelta strength",
                    elem_id=f"{tab}_condelta_strength",
                    scale=1,
                )

            also_native = gr.Checkbox(
                value=False,
                label="Also use native negative prompt above CFG 1.0",
                elem_id=f"{tab}_condelta_also_native_negative",
            )

        self.infotext_fields = [
            (threshold, "ConDelta CFG threshold"),
            (strength, "ConDelta strength"),
            (also_native, "ConDelta native negative above CFG 1"),
        ]
        self.paste_field_names = [name for _, name in self.infotext_fields]

        return [threshold, strength, also_native]

    def process(self, p, threshold=DEFAULT_THRESHOLD, strength=DEFAULT_STRENGTH, also_native=False):
        setattr(
            p,
            SETTINGS_ATTR,
            {
                "threshold": _as_float(threshold, DEFAULT_THRESHOLD, 1.0, 24.0),
                "strength": _as_float(strength, DEFAULT_STRENGTH, 0.0, 1.0),
                "also_native": _as_bool(also_native, False),
            },
        )


_patch_processing_methods()
