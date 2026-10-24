"""完成元数据发布、引用接口隔离和数据库快照的回归测试。"""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from app.schemas.chat import ChatRequest
from tests.chat_fakes import make_chat_service


async def test_completed_greeting_has_metadata_after_commit():
    """普通问候也须先完成保存再发布可反馈标识。参数：无；返回：None；异常：断言错误。"""
    service, repository, _, _ = make_chat_service(chunks=["您好"])
    events = []
    async for event in service.stream_turn(ChatRequest(session_id="greeting", message="你好")):
        events.append(event)
        if event.event == "answer_metadata":
            assert repository.turns[event.data["turn_id"]].turn_status == "completed"
    assert [event.event for event in events][-2:] == ["answer_metadata", "done"]
    assert events[-2].data == {"session_id": "greeting", "turn_id": repository.contexts[0].turn_id,
                               "status": "completed", "citations": []}


async def test_failed_commit_has_no_metadata_or_done():
    """保存失败的片段不能提供反馈或引用入口。参数：无；返回：None；异常：断言错误。"""
    service, repository, _, _ = make_chat_service(chunks=["您好"])
    repository.fail_finish = True
    events = [event async for event in service.stream_turn(ChatRequest(session_id="failed", message="你好"))]
    assert events[-1].event == "error"
    assert not any(event.event in {"answer_metadata", "done"} for event in events)


async def test_all_model_evidence_is_persisted_before_metadata():
    """只引用第一条时也保存模型看到的完整证据快照。参数：无；返回：None；异常：断言错误。"""
    from app.knowledge.answering import KnowledgeAnswer
    from tests.rag.test_quality_answer import _service
    service, repository, _ = _service(KnowledgeAnswer(useful=True, reason="", answer="答复[1]", citation_numbers=[1]))
    events = [event async for event in service.stream_turn(ChatRequest(session_id="cited", message="政策"))]
    assert events[-2].event == "answer_metadata"
    metadata = events[-2].data
    assert metadata["citations"][0]["answer"]
    assert repository.metadata[metadata["turn_id"]] == metadata


def test_citation_routes_use_saved_scope_and_uniform_404():
    """接口只取已完成同用户会话快照，所有不可读情况同样404。参数：无；返回：None；异常：断言错误。"""
    from app.main import create_app

    class Repository:
        """替代数据库边界，不接受浏览器提供的路径。"""

        async def get_citation(self, session_id, user_id, turn_id, n):
            """返回唯一允许的快照。参数：归属及引用编号；返回：dict或None；异常：无。"""
            if (session_id, user_id, turn_id, n) == ("s", "owner", "t", 1):
                return {"n": 1, "answer": "当时完整原文", "source_key": "qa:20", "section_path": None}
            return None

    @asynccontextmanager
    async def runtime():
        """注入真实路由的数据库边界。参数：无；返回：服务；异常：无。"""
        service, _, _, _ = make_chat_service()
        service.chat = Repository()
        yield service

    with TestClient(create_app(runtime)) as client:
        result = client.get("/api/chat/s/turns/t/citations/1?user_id=owner")
        assert result.status_code == 200 and result.json()["answer"] == "当时完整原文"
        for path in ["s/turns/t/citations/2?user_id=owner", "other/turns/t/citations/1?user_id=owner",
                     "s/turns/t/citations/1?user_id=other", "s/turns/unknown/citations/1?user_id=owner",
                     "s/turns/t/citations/1/source?user_id=owner"]:
            response = client.get("/api/chat/" + path)
            assert response.status_code == 404 and response.json() == {"detail": "引用不存在或不可访问"}


def test_ch03_live_parser_accepts_metadata_without_mixing_answer():
    """旧章真实验证兼容新完成事件，同时仍核对正文和轮次。参数：无；返回：None；异常：断言错误。"""
    import json
    from tests.ch03_live_runtime import inspect_exchange, parse_frame
    from tests.knowledge.test_live_runtime import _exchange
    metadata = {"session_id": "s", "turn_id": "turn1", "status": "completed", "citations": []}
    event = parse_frame(["event: answer_metadata", "data: " + json.dumps(metadata)], 0.2)
    observed = _exchange()
    observed["events"].insert(-1, event)
    result = inspect_exchange({"query": "邮费", "expect_empty": False}, observed)
    assert result["failures"] == [] and result["answer"] == "答案"
