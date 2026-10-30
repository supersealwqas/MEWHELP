"""审核API的严格人工输入、分页详情、状态冲突和恢复入口测试。"""

import importlib
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.api.chat import get_chat_service
from app.models.flywheel import ReviewQueue
from tests.flywheel.test_review_publish import review_runtime


async def test_review_api_validation_status_and_cursor(tmp_path, monkeypatch):
    """不存在/冲突/输入错误须分别404/409/422。参数：资源；返回：None；异常：断言失败。"""
    assert importlib.util.find_spec("app.api.flywheel"), "缺少审核API"
    async with review_runtime(tmp_path, monkeypatch) as env:
        app = FastAPI()
        app.include_router(importlib.import_module("app.api.flywheel").router)
        app.dependency_overrides[get_chat_service] = lambda: SimpleNamespace(review_publisher=env.publisher)
        async with env.factory.begin() as session:
            session.add(ReviewQueue(id=9007199254740993, normalized_question="第二条"))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            base = "/api/flywheel/reviews"
            assert (await client.get(base + "/999")).status_code == 404
            for suffix in ("/999/retry", "/999/reject", "/999/approve"):
                assert (await client.post(base + suffix, json={"answer": "人工答复"} if suffix.endswith("approve") else None)).status_code == 404
            for payload in ({}, {"answer": 3}, {"answer": " "}, {"answer": ""}, {"answer": "字" * 1600},
                            {"answer": "人工答复", "ai_suggested_answer": "注入"}, {"answer": "人工答复", "normalized_question": "替换"}):
                assert (await client.post(base + "/1/approve", json=payload)).status_code == 422
            for query in ("?status=错误", "?after_id=-1", "?limit=0", "?limit=201"):
                assert (await client.get(base + query)).status_code == 422
            first = (await client.get(base + "?status=待审&limit=1")).json()
            assert first["items"][0]["id"] == "1" and first["next_after_id"] == "1"
            second = (await client.get(base + "?after_id=1&limit=1")).json()
            assert second["items"][0]["id"] == "9007199254740993" and second["next_after_id"] is None
            assert (await client.post(base + "/1/retry")).status_code == 409
            rejected = await client.post(base + "/1/reject")
            assert rejected.status_code == 200 and rejected.json()["review_status"] == "驳回"
            assert (await client.post(base + "/1/reject")).json() == rejected.json()
            assert (await client.post(base + "/1/approve", json={"answer": "核准"})).status_code == 409
            assert (await client.get(base + "?status=驳回")).json()["items"][0]["id"] == "1"


async def test_review_api_retry_and_explicit_processing(tmp_path, monkeypatch):
    """发布失败明确保留pending，retry和process调用现有服务。参数：资源；返回：None；异常：断言失败。"""
    assert importlib.util.find_spec("app.api.flywheel"), "缺少审核API"
    async with review_runtime(tmp_path, monkeypatch) as env:
        calls = []
        async def process_pending(limit=50):
            """记录人工触发。参数：上限；返回：计数；异常：无。"""
            calls.append(limit)
            return {"processed": 2, "merged": 1, "failed": 0}
        app = FastAPI()
        app.include_router(importlib.import_module("app.api.flywheel").router)
        app.dependency_overrides[get_chat_service] = lambda: SimpleNamespace(review_publisher=env.publisher,
            flywheel=SimpleNamespace(process_pending=process_pending))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            env.index.fail_after_upsert = True
            failed = await client.post("/api/flywheel/reviews/1/approve", json={"answer": "人工答案"})
            assert failed.status_code == 503 and failed.json()["detail"]["publication_status"] == "pending"
            detail = (await client.get("/api/flywheel/reviews/1")).json()
            assert detail["approved_answer"] == "人工答案" and detail["review_status"] == "待审"
            assert (await client.post("/api/flywheel/reviews/1/reject")).status_code == 409
            assert (await client.post("/api/flywheel/reviews/1/approve", json={"answer": "改写答案"})).status_code == 409
            env.index.fail_after_upsert = False
            retried = await client.post("/api/flywheel/reviews/1/retry")
            assert retried.status_code == 200 and retried.json()["publication_status"] == "published"
            assert (await client.post("/api/flywheel/process")).json() == {"processed": 2, "merged": 1, "failed": 0}
            assert calls == [50]


async def test_tokenizer_failure_does_not_claim_pending_publication(tmp_path, monkeypatch):
    """冻结前依赖故障不可伪报已冻结pending。参数：资源；返回：None；异常：断言失败。"""
    from app.knowledge.embedding import EmbeddingUnavailable
    async with review_runtime(tmp_path, monkeypatch) as env:
        def failed(value):
            """模拟真实分词器不可用。参数：文本；返回：不返回；异常：EmbeddingUnavailable。"""
            raise EmbeddingUnavailable("分词器不可用")
        env.encoder.count_tokens = failed
        app = FastAPI()
        app.include_router(importlib.import_module("app.api.flywheel").router)
        app.dependency_overrides[get_chat_service] = lambda: SimpleNamespace(review_publisher=env.publisher)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/flywheel/reviews/1/approve", json={"answer": "人工正文"})
            assert response.status_code == 503
            assert response.json()["detail"].get("publication_status") != "pending"
            assert (await env.repo.review_detail(env.review_id))["publication_status"] == "not_started"


async def test_release_failure_reports_persisted_publication_status(tmp_path, monkeypatch):
    """审核已提交后释放异常不可把已发布状态伪报pending。参数：资源；返回：None；异常：断言失败。"""
    from contextlib import asynccontextmanager
    async with review_runtime(tmp_path, monkeypatch) as env:
        original = env.store.writer
        @asynccontextmanager
        async def writer():
            """发布已完成后模拟释放确认失败。参数：无；返回：原写入器；异常：RuntimeError。"""
            async with original() as active:
                yield active
            raise RuntimeError("release confirmation failed")
        env.store.writer = writer
        app = FastAPI()
        app.include_router(importlib.import_module("app.api.flywheel").router)
        app.dependency_overrides[get_chat_service] = lambda: SimpleNamespace(review_publisher=env.publisher)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            result = await client.post("/api/flywheel/reviews/1/approve", json={"answer": "人工答案"})
            assert result.status_code == 503
            assert (await env.repo.review_detail(env.review_id))["publication_status"] == "published"
            assert result.json()["detail"]["publication_status"] == "published"


async def test_review_api_exposes_raw_and_proven_retrieval_state(tmp_path, monkeypatch):
    """HTTP同时保留原始NULL与可解释展示状态，不把旧缺失当未检索。参数：资源；返回：None；异常：断言失败。"""
    from app.models.rag_quality import LowConfidenceQuestion
    from app.repositories.chat import ChatRepository
    from tests.flywheel.test_pool import start
    async with review_runtime(tmp_path, monkeypatch) as env:
        chat = ChatRepository(env.factory)
        await chat.finish_turn(await start(chat), "无需检索的回答")
        receipt = await chat.record_unresolved_feedback("s", "u", "first")
        async with env.factory.begin() as session:
            current = await session.get(LowConfidenceQuestion, int(receipt.pool_id))
            current.matched_review_id = env.review_id
            session.add(LowConfidenceQuestion(raw_question="旧原件", source="retrieval_low_conf", matched_review_id=env.review_id))
        app = FastAPI()
        app.include_router(importlib.import_module("app.api.flywheel").router)
        app.dependency_overrides[get_chat_service] = lambda: SimpleNamespace(review_publisher=env.publisher)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            result = await client.get("/api/flywheel/reviews/1")
            assert result.status_code == 200
            current, legacy = result.json()["originals"]
            assert current["retrieved_chunks"] is legacy["retrieved_chunks"] is None
            assert current["retrieval_status"] == "not_retrieved" and current["retrieval_snapshot"] is None
            assert legacy["retrieval_status"] == legacy["retrieval_snapshot"]["status"] == "legacy_unavailable"
