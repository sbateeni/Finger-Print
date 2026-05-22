"""
Global analysis queue: one fingerprint pair analysis at a time.
Telegram and web share the same worker (single process — embedded bot in server).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Optional

from services.pair_analysis import (
    analyze_fingerprint_pair,
    format_match_summary_ar,
    telegram_deep_analysis_enabled,
    telegram_sweep_mode,
)
from utils.telegram_notify import (
    deliver_result_sync,
    notify_chat_ids,
    notify_on_web_enabled,
    send_text_sync,
)

logger = logging.getLogger(__name__)

SendFn = Callable[[int, str], Awaitable[None]]


@dataclass
class PairJob:
    ref_bytes: bytes
    qry_bytes: bytes
    kwargs: dict[str, Any]
    source: str
    chat_id: Optional[int] = None
    notify_telegram: bool = True
    source_label: str = ""
    future: Optional[asyncio.Future] = None


class AnalysisQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[PairJob] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._send: Optional[SendFn] = None

    def set_telegram_sender(self, send_fn: SendFn) -> None:
        self._send = send_fn

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker_loop(), name="analysis-queue")
        logger.info("Analysis queue worker started")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    def queue_position(self) -> int:
        """Number of jobs waiting (not including the one currently running)."""
        return self._queue.qsize()

    async def _notify_chat(self, chat_id: int, text: str) -> None:
        if self._send:
            try:
                await self._send(chat_id, text)
                return
            except Exception as e:
                logger.warning("bot send failed, fallback to HTTP: %s", e)
        await asyncio.to_thread(send_text_sync, chat_id, text)

    async def enqueue_pair(
        self,
        ref_bytes: bytes,
        qry_bytes: bytes,
        *,
        source: str,
        chat_id: Optional[int] = None,
        notify_telegram: bool = True,
        source_label: str = "",
        **analyze_kwargs: Any,
    ) -> tuple[PairJob, int]:
        """
        Enqueue pair analysis. Returns (job, position).
        position 0 = next to run; position > 0 = waiters ahead in queue.
        """
        loop = asyncio.get_running_loop()
        job = PairJob(
            ref_bytes=ref_bytes,
            qry_bytes=qry_bytes,
            kwargs=analyze_kwargs,
            source=source,
            chat_id=chat_id,
            notify_telegram=notify_telegram,
            source_label=source_label or source,
            future=loop.create_future(),
        )
        ahead = self._queue.qsize() + (1 if self._lock.locked() else 0)
        await self._queue.put(job)
        if ahead > 0 and chat_id and notify_telegram:
            await self._notify_chat(
                chat_id,
                f"⏳ *طلبك في قائمة الانتظار*\nموقعك: *{ahead + 1}*\nسيتم إرسال النتيجة هنا عند انتهاء الدور.",
            )
        return job, ahead

    async def wait_result(self, job: PairJob) -> dict[str, Any]:
        if job.future is None:
            raise RuntimeError("job has no future")
        return await job.future

    async def run_exclusive(self, label: str = "web") -> int:
        """
        Acquire the analysis lock (blocks until turn).
        Returns how many jobs were ahead when waiting started.
        """
        ahead = self._queue.qsize() + (1 if self._lock.locked() else 0)
        await self._lock.acquire()
        return ahead

    def release_exclusive(self) -> None:
        if self._lock.locked():
            self._lock.release()

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                async with self._lock:
                    run_kwargs = dict(job.kwargs)
                    if job.source == "telegram" and telegram_deep_analysis_enabled():
                        run_kwargs.setdefault("auto_sweep_before", True)
                        run_kwargs.setdefault("sweep_mode", telegram_sweep_mode())
                        if job.chat_id and job.notify_telegram:
                            mode = run_kwargs.get("sweep_mode", "wide")
                            await self._notify_chat(
                                job.chat_id,
                                "🔬 *تحليل عميق* — نفس محرك الكمبيوتر:\n"
                                f"1) بحث محاذاة تلقائي ({mode})\n"
                                "2) استخراج نقاط + MCC + ORB + تقرير PDF…\n"
                                "_قد يستغرق عدة دقائق._",
                            )
                    elif job.chat_id and job.notify_telegram:
                        await self._notify_chat(job.chat_id, "🔬 *بدء تحليل طلبك الآن…*")
                    result = await asyncio.to_thread(
                        analyze_fingerprint_pair,
                        job.ref_bytes,
                        job.qry_bytes,
                        **run_kwargs,
                    )
                    await self._deliver(job, result)
                    if job.future is not None and not job.future.done():
                        job.future.set_result(result)
            except Exception as e:
                logger.exception("analysis queue job failed")
                err = {"ok": False, "error": str(e)}
                if job.future is not None and not job.future.done():
                    job.future.set_result(err)
                if job.chat_id and job.notify_telegram:
                    await self._notify_chat(job.chat_id, f"❌ خطأ أثناء التحليل: {e}")
            finally:
                self._queue.task_done()

    async def _deliver(self, job: PairJob, result: dict[str, Any]) -> None:
        if job.source == "web_notify_only":
            label = job.source_label or "الواجهة"
            for cid in notify_chat_ids():
                await asyncio.to_thread(
                    deliver_result_sync, cid, result, source_label=label
                )
            return

        if job.chat_id and job.notify_telegram:
            label = job.source_label or job.source
            if self._send:
                summary = format_match_summary_ar(result)
                if label:
                    summary = f"📡 *{label}*\n\n{summary}"
                await self._send(job.chat_id, summary)
                from pathlib import Path
                from utils.telegram_notify import send_document_sync

                pdf_p = result.get("report_pdf")
                html_p = result.get("report_html")
                doc = None
                if pdf_p and Path(pdf_p).is_file():
                    doc = Path(pdf_p)
                elif html_p and Path(html_p).is_file():
                    doc = Path(html_p)
                if doc:
                    await asyncio.to_thread(
                        send_document_sync, job.chat_id, doc, caption="تقرير"
                    )
            else:
                await asyncio.to_thread(
                    deliver_result_sync,
                    job.chat_id,
                    result,
                    source_label=label,
                )

        elif job.source == "web" and notify_on_web_enabled():
            label = job.source_label or "الواجهة"
            for cid in notify_chat_ids():
                await asyncio.to_thread(
                    deliver_result_sync, cid, result, source_label=label
                )


_queue: Optional[AnalysisQueue] = None


def get_analysis_queue() -> AnalysisQueue:
    global _queue
    if _queue is None:
        _queue = AnalysisQueue()
    return _queue


async def start_analysis_queue() -> None:
    await get_analysis_queue().start()


async def stop_analysis_queue() -> None:
    await get_analysis_queue().stop()


def build_web_result_payload(
    match: dict[str, Any],
    report_rel: Optional[str],
    *,
    same_file_warning: bool = False,
    forensic_quality_warning: bool = False,
) -> dict[str, Any]:
    from pathlib import Path

    from config import OUTPUT_DIR
    from services.analysis_service import _ensure_pdf_from_html

    report_html: Optional[Path] = None
    report_pdf: Optional[Path] = None
    if report_rel:
        report_html = (Path(OUTPUT_DIR) / report_rel.replace("\\", "/")).resolve()
        if report_html.exists():
            try:
                report_pdf = _ensure_pdf_from_html(report_html)
            except Exception:
                report_pdf = None
    return {
        "ok": True,
        "error": None,
        "same_file_warning": same_file_warning,
        "match": match,
        "report_rel": report_rel,
        "report_html": str(report_html) if report_html and report_html.exists() else None,
        "report_pdf": str(report_pdf) if report_pdf and report_pdf.exists() else None,
        "forensic_quality_warning": forensic_quality_warning,
    }


def schedule_web_telegram_notify(
    match: dict[str, Any],
    report_rel: Optional[str],
    *,
    same_file_warning: bool = False,
    forensic_quality_warning: bool = False,
    operator_name: str = "",
    case_reference: str = "",
) -> None:
    """Fire-and-forget Telegram notification after web analysis (same event loop)."""
    if not notify_on_web_enabled():
        return
    payload = build_web_result_payload(
        match,
        report_rel,
        same_file_warning=same_file_warning,
        forensic_quality_warning=forensic_quality_warning,
    )
    label = "الواجهة"
    if operator_name or case_reference:
        label = f"الواجهة — {operator_name or case_reference}".strip(" —")

    async def _send() -> None:
        q = get_analysis_queue()
        job = PairJob(
            ref_bytes=b"",
            qry_bytes=b"",
            kwargs={},
            source="web_notify_only",
            notify_telegram=True,
            source_label=label,
        )
        await q._deliver(job, payload)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send())
    except RuntimeError:
        from utils.telegram_notify import notify_result_to_all_chats

        notify_result_to_all_chats(payload, source_label=label)
