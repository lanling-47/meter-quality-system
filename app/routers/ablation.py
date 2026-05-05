"""消融实验路由"""
import json
import asyncio
from fastapi import APIRouter, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from database import get_db

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/ablation", response_class=HTMLResponse)
async def ablation_page(request: Request, db: Session = Depends(get_db)):
    from app.services.ablation_service import load_ablation_results, FEATURE_GROUPS, MODEL_REGISTRY
    cached = load_ablation_results()
    return templates.TemplateResponse(request, "ablation.html", {
        "page": "ablation",
        "has_results": cached is not None,
        "cached_results": json.dumps(cached, ensure_ascii=False) if cached else "null",
        "feature_groups": json.dumps({k: {"name": v["name"], "desc": v["desc"],
                                           "n_features": len(v["features"]), "color": v["color"]}
                                       for k, v in FEATURE_GROUPS.items()}, ensure_ascii=False),
        "model_registry": json.dumps({k: {"name": v["name"], "name_en": v["name_en"], "color": v["color"]}
                                       for k, v in MODEL_REGISTRY.items()}, ensure_ascii=False),
    })


@router.get("/ablation/results")
async def ablation_results(request: Request):
    """获取缓存的消融实验结果"""
    from app.services.ablation_service import load_ablation_results
    results = load_ablation_results()
    if results is None:
        return JSONResponse({"error": "尚未运行消融实验，请先点击「运行实验」"}, status_code=404)
    return JSONResponse(results)


@router.get("/ablation/run-stream")
async def ablation_run_stream(request: Request, db: Session = Depends(get_db)):
    """
    SSE 流式运行消融实验
    前端通过 EventSource 监听进度
    """
    from app.services.ablation_service import run_ablation_experiment
    from app.services.predict_service import _emit as global_emit

    progress_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    def emit_fn(event: str, data: dict):
        msg = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        try:
            progress_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    async def run_in_background():
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: run_ablation_experiment(db, emit_fn=emit_fn)
            )
        except Exception as e:
            emit_fn("error", {"message": str(e)})
        finally:
            emit_fn("stream_end", {})

    async def event_generator():
        asyncio.create_task(run_in_background())
        while True:
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=60)
                parsed = json.loads(msg)
                yield f"data: {msg}\n\n"
                if parsed.get("event") in ("ablation_done", "error", "stream_end"):
                    break
            except asyncio.TimeoutError:
                yield "data: {\"event\":\"heartbeat\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ablation/run")
async def ablation_run_sync(request: Request, db: Session = Depends(get_db)):
    """同步运行消融实验（备用接口，前端等待）"""
    from app.services.ablation_service import run_ablation_experiment
    try:
        results = run_ablation_experiment(db)
        return JSONResponse({"success": True, "results": results})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
