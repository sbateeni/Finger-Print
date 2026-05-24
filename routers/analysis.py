import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from config import (
    APP_VERSION,
    MATCH_SCORE_THRESHOLDS,
    MIN_MINUTIAE_RECOMMENDED,
    SOFTWARE_NAME,
    DEFAULT_BORDER_MARGIN,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_MIN_CONTRAST,
    DEFAULT_MIN_ANGLE_DIFF,
)
from utils.web_utils import _render
from services.analysis_service import (
    analysis_event_generator,
    process_form_analysis,
    run_matching_pipeline,
    run_auto_sweep,
    _build_visual_ctx,
    _sanitize_match_for_json,
    _apply_deep_sweep_to_transforms,
    resolve_analysis_mode,
    is_deep_analysis,
)
from utils.web_utils import resolve_ui_lang
from utils.sse_utils import _sse_line
from services.analysis_queue import get_analysis_queue, schedule_web_telegram_notify

logger = logging.getLogger(__name__)

router = APIRouter()


def _form_int(form, key: str, default: int) -> int:
    v = form.get(key)
    if v is None or v == "":
        return default
    return int(v)


def _form_bool(form, key: str, default: bool = False) -> bool:
    v = form.get(key)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "on", "yes")


@router.post("/analyze-stream")
async def analyze_stream(request: Request):
    form = await request.form()
    orig = form.get("original")
    part = form.get("partial")
    if orig is None or part is None:
        async def empty_err():
            yield _sse_line({"type": "fatal", "message": "يرفع ملفين."})

        return StreamingResponse(empty_err(), media_type="text/event-stream")

    o_raw = await orig.read()
    p_raw = await part.read()

    border_margin = _form_int(form, "border_margin", DEFAULT_BORDER_MARGIN)
    min_distance = _form_int(form, "min_distance", DEFAULT_MIN_DISTANCE)
    min_contrast = _form_int(form, "min_contrast", DEFAULT_MIN_CONTRAST)
    min_angle_diff = _form_int(form, "min_angle_diff", DEFAULT_MIN_ANGLE_DIFF)
    denoise_method = str(form.get("denoise_method") or "fastNlMeans")
    fast_denoise_h = _form_int(form, "fast_denoise_h", 10)
    gauss_ksize = _form_int(form, "gauss_ksize", 3)
    original_zoom = _form_int(form, "original_zoom", 100)
    partial_zoom = _form_int(form, "partial_zoom", 100)
    partial_shift_x = _form_int(form, "partial_shift_x", 0)
    partial_shift_y = _form_int(form, "partial_shift_y", 0)
    apply_preview_scale = _form_bool(form, "apply_preview_scale", True)
    auto_scale_normalization = _form_bool(form, "auto_scale_normalization", True)
    operator_name = str(form.get("operator_name") or "")
    case_reference = str(form.get("case_reference") or "")
    analysis_mode = resolve_analysis_mode(str(form.get("analysis_mode") or ""))
    auto_align_sweep = (
        is_deep_analysis(analysis_mode)
        and "auto_align_sweep" in form
        and _form_bool(form, "auto_align_sweep", False)
    )
    report_lang = resolve_ui_lang(
        str(form.get("report_lang") or request.query_params.get("lang") or "ar")
    )
    ref_grid_divisions = _form_int(form, "ref_grid_divisions", 1)
    ref_grid_cell = _form_int(form, "ref_grid_cell", 0)
    ref_grid_cells = str(form.get("ref_grid_cells") or "")
    ref_region = str(form.get("ref_region") or "0,0,1,1")
    partial_grid_divisions = _form_int(form, "partial_grid_divisions", 1)
    partial_grid_cells = str(form.get("partial_grid_cells") or "")
    partial_region = str(form.get("partial_region") or "0,0,1,1")

    async def queued_stream() -> AsyncIterator[bytes]:
        q = get_analysis_queue()
        ahead = await q.run_exclusive(label="web-stream")
        if ahead > 0:
            yield _sse_line(
                {
                    "type": "log",
                    "message": f"⏳ في انتظار دورك — {ahead} طلب(ات) أمامك في الطابور…",
                }
            )
        try:
            gen = analysis_event_generator(
                o_raw,
                p_raw,
                border_margin,
                min_distance,
                min_contrast,
                min_angle_diff,
                denoise_method,
                fast_denoise_h,
                gauss_ksize,
                original_zoom,
                partial_zoom,
                partial_shift_x,
                partial_shift_y,
                apply_preview_scale,
                auto_scale_normalization,
                operator_name,
                case_reference,
                analysis_mode,
                report_lang,
                auto_align_sweep,
                ref_grid_divisions,
                ref_grid_cell,
                ref_grid_cells,
                ref_region,
                partial_grid_divisions,
                partial_grid_cells,
                partial_region,
            )
            same_file_warning = False
            for chunk in gen:
                if chunk:
                    try:
                        line = chunk.decode("utf-8").strip()
                        if line.startswith("data:"):
                            payload = json.loads(line[5:].strip())
                            if payload.get("type") == "hashes":
                                same_file_warning = bool(payload.get("same_file_warning"))
                            elif payload.get("type") == "done":
                                schedule_web_telegram_notify(
                                    payload.get("match") or {},
                                    payload.get("report_download"),
                                    same_file_warning=same_file_warning,
                                    forensic_quality_warning=bool(
                                        payload.get("forensic_quality_warning")
                                    ),
                                    operator_name=operator_name,
                                    case_reference=case_reference,
                                    report_lang=payload.get("report_lang") or report_lang,
                                )
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                yield chunk
                await asyncio.sleep(0)
        finally:
            q.release_exclusive()

    return StreamingResponse(
        queued_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/analyze-sweep")
async def analyze_sweep(request: Request):
    form = await request.form()
    orig = form.get("original")
    part = form.get("partial")
    if orig is None or part is None:
        return JSONResponse({"ok": False, "message": "يرجى رفع الصورتين أولاً."}, status_code=400)

    o_raw = await orig.read()
    p_raw = await part.read()
    if not o_raw or not p_raw:
        return JSONResponse({"ok": False, "message": "فشل قراءة الملفات."}, status_code=400)

    border_margin = _form_int(form, "border_margin", DEFAULT_BORDER_MARGIN)
    min_distance = _form_int(form, "min_distance", DEFAULT_MIN_DISTANCE)
    min_contrast = _form_int(form, "min_contrast", DEFAULT_MIN_CONTRAST)
    min_angle_diff = _form_int(form, "min_angle_diff", DEFAULT_MIN_ANGLE_DIFF)
    denoise_method = str(form.get("denoise_method") or "fastNlMeans")
    fast_denoise_h = _form_int(form, "fast_denoise_h", 10)
    gauss_ksize = _form_int(form, "gauss_ksize", 3)
    original_zoom = _form_int(form, "original_zoom", 100)
    partial_zoom = _form_int(form, "partial_zoom", 100)
    partial_shift_x = _form_int(form, "partial_shift_x", 0)
    partial_shift_y = _form_int(form, "partial_shift_y", 0)
    apply_preview_scale = _form_bool(form, "apply_preview_scale", True)
    auto_scale_normalization = _form_bool(form, "auto_scale_normalization", True)
    sweep_mode = str(form.get("sweep_mode") or "quick")

    try:
        result = run_auto_sweep(
            o_raw,
            p_raw,
            border_margin,
            min_distance,
            min_contrast,
            min_angle_diff,
            denoise_method,
            fast_denoise_h,
            gauss_ksize,
            original_zoom,
            partial_zoom,
            partial_shift_x,
            partial_shift_y,
            apply_preview_scale,
            auto_scale_normalization,
            sweep_mode,
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 422)
    except Exception as e:
        logger.exception(e)
        return JSONResponse({"ok": False, "message": f"Auto-sweep failed: {e}"}, status_code=500)


@router.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    original: UploadFile = File(...),
    partial: UploadFile = File(...),
    border_margin: int = Form(DEFAULT_BORDER_MARGIN),
    min_distance: int = Form(DEFAULT_MIN_DISTANCE),
    min_contrast: int = Form(DEFAULT_MIN_CONTRAST),
    min_angle_diff: int = Form(DEFAULT_MIN_ANGLE_DIFF),
    denoise_method: str = Form("fastNlMeans"),
    fast_denoise_h: int = Form(10),
    gauss_ksize: int = Form(3),
    original_zoom: int = Form(100),
    partial_zoom: int = Form(100),
    partial_shift_x: int = Form(0),
    partial_shift_y: int = Form(0),
    apply_preview_scale: bool = Form(True),
    auto_scale_normalization: bool = Form(True),
    operator_name: str = Form(""),
    case_reference: str = Form(""),
    analysis_mode: str = Form("deep"),
    report_lang: str = Form(""),
    auto_align_sweep: str = Form("0"),
    ref_grid_divisions: int = Form(1),
    ref_grid_cell: int = Form(0),
    ref_grid_cells: str = Form(""),
    ref_region: str = Form("0,0,1,1"),
    partial_grid_divisions: int = Form(1),
    partial_grid_cells: str = Form(""),
    partial_region: str = Form("0,0,1,1"),
):
    ui_lang = resolve_ui_lang(report_lang or request.query_params.get("lang"))
    analysis_mode = resolve_analysis_mode(analysis_mode)
    do_auto_align = is_deep_analysis(analysis_mode) and _form_bool(
        {"auto_align_sweep": auto_align_sweep}, "auto_align_sweep", False
    )

    try:
        o_raw = await original.read()
        p_raw = await partial.read()
    except Exception as e:
        logger.exception(e)
        raise HTTPException(400, "قراءة الملفات فشلت") from e

    if not o_raw or not p_raw:
        return _render(
            request,
            {
                "results": None,
                "error": "يرفع ملفين صالحين للبصمة الأصلية والجزئية.",
                "match": None,
                "report_download": None,
                "thresholds": MATCH_SCORE_THRESHOLDS,
                "software_name": SOFTWARE_NAME,
                "app_version": APP_VERSION,
                "min_minutiae_recommended": MIN_MINUTIAE_RECOMMENDED,
                "operator_name": "",
                "case_reference": "",
            },
        )

    eff_zoom, eff_shift_x, eff_shift_y = partial_zoom, partial_shift_x, partial_shift_y
    if do_auto_align:
        eff_zoom, eff_shift_x, eff_shift_y, _sweep = _apply_deep_sweep_to_transforms(
            o_raw,
            p_raw,
            border_margin,
            min_distance,
            min_contrast,
            min_angle_diff,
            denoise_method,
            fast_denoise_h,
            gauss_ksize,
            original_zoom,
            partial_zoom,
            partial_shift_x,
            partial_shift_y,
            apply_preview_scale,
            auto_scale_normalization,
            analysis_mode,
            auto_align_sweep=True,
            ref_grid_divisions=ref_grid_divisions,
            ref_grid_cell=ref_grid_cell,
            ref_grid_cells=ref_grid_cells,
            ref_region=ref_region,
            partial_grid_divisions=partial_grid_divisions,
            partial_grid_cells=partial_grid_cells,
            partial_region=partial_region,
        )
        partial_zoom, partial_shift_x, partial_shift_y = eff_zoom, eff_shift_x, eff_shift_y

    try:
        same_file, sha_o, sha_p, ro, rp, dm = process_form_analysis(
            o_raw,
            p_raw,
            border_margin,
            min_distance,
            min_contrast,
            min_angle_diff,
            denoise_method,
            fast_denoise_h,
            gauss_ksize,
            original_zoom,
            partial_zoom,
            partial_shift_x,
            partial_shift_y,
            apply_preview_scale,
            auto_scale_normalization,
            operator_name,
            case_reference,
            ref_grid_divisions=ref_grid_divisions,
            ref_grid_cell=ref_grid_cell,
            ref_grid_cells=ref_grid_cells,
            ref_region=ref_region,
            partial_grid_divisions=partial_grid_divisions,
            partial_grid_cells=partial_grid_cells,
            partial_region=partial_region,
            report_lang=ui_lang,
        )
    except Exception as e:
        logger.exception(e)
        return _render(
            request,
            {
                "results": None,
                "error": f"تعذر فك ترميز الصورة: {e}",
                "match": None,
                "report_download": None,
                "thresholds": MATCH_SCORE_THRESHOLDS,
                "software_name": SOFTWARE_NAME,
                "app_version": APP_VERSION,
                "min_minutiae_recommended": MIN_MINUTIAE_RECOMMENDED,
                "operator_name": "",
                "case_reference": "",
            },
        )

    ctx = {
        "request": request,
        "results": None,
        "error": None,
        "match": None,
        "report_download": None,
        "same_file_warning": same_file,
        "thresholds": MATCH_SCORE_THRESHOLDS,
        "form": {
            "border_margin": border_margin,
            "min_distance": min_distance,
            "min_contrast": min_contrast,
            "min_angle_diff": min_angle_diff,
            "denoise_method": dm,
            "fast_denoise_h": fast_denoise_h,
            "gauss_ksize": gauss_ksize,
            "original_zoom": original_zoom,
            "partial_zoom": partial_zoom,
            "partial_shift_x": partial_shift_x,
            "partial_shift_y": partial_shift_y,
            "apply_preview_scale": apply_preview_scale,
            "auto_scale_normalization": auto_scale_normalization,
            "auto_scale_factor_applied": round(float(ro.get("auto_scale_factor_applied", 1.0)), 4),
            "analysis_mode": analysis_mode,
            "auto_align_sweep": do_auto_align,
            "ref_grid_divisions": ref_grid_divisions,
            "ref_grid_cell": ref_grid_cell,
            "ref_region": ref_region,
            "partial_grid_divisions": partial_grid_divisions,
            "partial_region": partial_region,
            "report_lang": ui_lang,
        },
        "software_name": SOFTWARE_NAME,
        "app_version": APP_VERSION,
        "min_minutiae_recommended": MIN_MINUTIAE_RECOMMENDED,
        "sha256_original": sha_o,
        "sha256_partial": sha_p,
        "operator_name": operator_name,
        "case_reference": case_reference,
    }

    if ro.get("error"):
        ctx["error"] = "الأصلية: " + ro["error"]
        return _render(request, ctx)
    if rp.get("error"):
        ctx["error"] = "الجزئية: " + rp["error"]
        return _render(request, ctx)

    mo, mp = ro.get("minutiae") or [], rp.get("minutiae") or []
    if not mo or not mp:
        ctx["error"] = "لا توجد نقاط دقيقة كافية في إحدى الصورتين."
        ctx["results"] = _build_visual_ctx(ro, rp, None, None)
        ctx["forensic_quality_warning"] = True
        return _render(request, ctx)

    match_result, matches_vis, report_rel, audit, forensic_quality_warning = run_matching_pipeline(
        ro, rp, sha_o, sha_p, dm, ctx["form"], operator_name, case_reference,
        border_margin, min_distance, min_contrast, min_angle_diff,
        fast_denoise_h, gauss_ksize
    )

    ctx["forensic_quality_warning"] = forensic_quality_warning
    ctx["results"] = _build_visual_ctx(ro, rp, match_result, matches_vis)
    ctx["match"] = match_result
    ctx["report_download"] = report_rel
    ctx["audit"] = audit
    schedule_web_telegram_notify(
        match_result,
        report_rel,
        same_file_warning=same_file,
        forensic_quality_warning=forensic_quality_warning,
        operator_name=operator_name,
        case_reference=case_reference,
        report_lang=(audit or {}).get("report_lang") or ui_lang,
    )
    return _render(request, ctx)
