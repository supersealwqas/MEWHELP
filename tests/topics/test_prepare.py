"""验证主题准备入口对合法长原问的隔离、来源回配与成功批次恢复。"""

import json
import socket
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk

from app.schemas.chat import ChatRequest
from app.topics.cli import prepare_dataset
from app.topics.data import validate_dataset
from app.topics.labelling import TopicLabeller
from app.topics.pool import derive_pool


class ReplyModel:
    """只替代外部模型流，保留真实清洗、工具解析及序号校验。"""

    def __init__(self, *, fail_on_call=None, disputed=False):
        """设置受控响应。参数：fail_on_call为故障请求序号或None，disputed为争议标记；返回：无；异常：无。"""
        self.requests = []
        self.fail_on_call = fail_on_call
        self.disputed = disputed

    def bind_tools(self, tools, **kwargs):
        """保存真实工具名称。参数：tools工具定义、kwargs绑定选项；返回：自身；异常：非法定义KeyError。"""
        self.tool_name = tools[0]["function"]["name"]
        return self

    async def astream(self, messages):
        """回显请求或模拟中断。参数：messages真实模型消息；返回：工具分片流；异常：RuntimeError或JSON错误。"""
        payload = json.loads(messages[1].content)
        self.requests.append(payload)
        if len(self.requests) == self.fail_on_call:
            raise RuntimeError("模拟第二批网络中断")
        rows = [dict(index=item["index"], text=item["text"], labels=["尺码"],
                     family=f"fixture-{item['index']}", needs_review=self.disputed,
                     review_reason="尺码边界有争议" if self.disputed else "")
                for item in reversed(payload["items"])]
        yield AIMessageChunk(content="", tool_calls=[dict(name=self.tool_name, id="fixture-call", args={"items": rows})],
                             response_metadata={"finish_reason": "tool_calls"})


def _block_network(monkeypatch):
    """禁止测试触碰真实网络。参数：monkeypatch为pytest替换器；返回：无；异常：连接尝试AssertionError。"""
    def deny(*args, **kwargs):
        """拒绝真实连接。参数：任意socket参数；返回：无；异常：AssertionError。"""
        raise AssertionError("主题准备回归禁止真实网络")

    # 在异步用例启动后替换，避免影响Windows事件循环初始化所用socketpair。
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


def _export_source(path, texts):
    """通过真实聊天校验和池派生建立虚构源。参数：path文件路径、texts原问；返回：派生记录；异常：校验或IO错误。"""
    records = []
    for index, text in enumerate(texts, 1):
        accepted = ChatRequest(session_id="prepare-regression", message=text).message
        records.append(SimpleNamespace(id=index, conversation_id=index, source="self_check", raw_question=accepted,
                                       matched_review_id=None, created_at=None))
    exported = derive_pool(records, [], [], [])
    path.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
    return exported


def _read_parts(output):
    """读取三份真实数据制品。参数：output制品目录；返回：合并记录列表；异常：IO或JSON错误。"""
    return [json.loads(line) for split in ("train", "validation", "test")
            for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def _assert_origin(row, original):
    """核对来源及完整脱敏正文。参数：row制品行、original派生源行；返回：无；异常：断言失败。"""
    assert row["text"] == row["original_text"] == original["text"]
    assert row["pool_ids"] == [original["question_id"]]
    assert row["source"] == original["source"]
    assert row["group_id"] == original["group_id"]
    assert row["provenance"] == original["provenance"]
    assert row["history_uncertain"] == original["history_uncertain"]
    assert row["pool_review_status"] == original["review_status"]


@pytest.mark.asyncio
async def test_prepare_isolates_long_questions_and_reindexes_short_inputs(tmp_path, monkeypatch):
    """长题不能阻断短题或误配模型序号，1500字符仍可外发。

    参数：tmp_path为隔离目录，monkeypatch为网络阻断器。返回：None。
    异常：断言失败、实际准备校验或IO错误。
    """
    _block_network(monkeypatch)
    long_text = "尺码怎么选？" + "我在说明这件商品的尺寸。" * 150
    assert len(long_text) == 1806
    source, output = tmp_path / "pool.json", tmp_path / "dataset"
    originals = _export_source(source, [long_text, "鞋子的尺码怎么选？", "尺" * 1501, "问" * 1500])
    source_before = source.read_bytes()
    model = ReplyModel()
    manifest = await prepare_dataset(source, output, per_label=0, multi=0, max_calls=1,
                                     labeller=TopicLabeller(model, report_dir=tmp_path / "raw", max_attempts=1))
    assert source.read_bytes() == source_before
    assert model.requests == [{"task": "label", "items": [
        {"index": 0, "text": "鞋子的尺码怎么选？"}, {"index": 1, "text": "问" * 1500}]}]
    rows = {row["pool_ids"][0]: row for row in _read_parts(output)}
    assert set(rows) == {"2", "4"}
    _assert_origin(rows["2"], originals[1])
    _assert_origin(rows["4"], originals[3])
    pending = json.loads((tmp_path / "dataset.resume/pending.json").read_text(encoding="utf-8"))
    assert len(pending) == 2
    for row, original in zip(pending, [originals[0], originals[2]], strict=True):
        _assert_origin(row, original)
        assert row["labels"] == [] and row["needs_review"] is True
        assert "1500" in row["review_reason"]
        assert row["annotation_status"] != "model_prelabel"
    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    isolated = [row for row in review["items"] if row["split"] == "pending_not_in_dataset"]
    assert [row["pool_ids"] for row in isolated] == [["1"], ["3"]]
    assert isolated[0]["text"] == long_text
    assert manifest["metadata"]["pending_count"] == manifest["metadata"]["overlong_count"] == 2
    assert validate_dataset(output)["total"] == 2


@pytest.mark.asyncio
async def test_prepare_all_long_questions_still_writes_review_without_model(tmp_path, monkeypatch):
    """全长输入也能产出完整待复核包，且不构建模型、不消耗请求预算。

    参数：tmp_path为隔离目录，monkeypatch为外部边界替换器。返回：None。
    异常：断言失败、实际准备校验或IO错误。
    """
    _block_network(monkeypatch)

    def reject_model(*args, **kwargs):
        """拒绝不必要的模型构建。参数：任意模型参数；返回：无；异常：AssertionError。"""
        raise AssertionError("全长输入不应构建外部模型")

    monkeypatch.setattr("app.core.llm.get_chat_model", reject_model)
    source, output = tmp_path / "pool.json", tmp_path / "dataset"
    originals = _export_source(source, [f"尺码{i}：" + "问" * 1800 for i in range(33)])
    manifest = await prepare_dataset(source, output, per_label=0, multi=0, max_calls=1)
    assert validate_dataset(output)["total"] == 0
    assert _read_parts(output) == []
    assert manifest["metadata"]["logical_requests"] == 0
    assert manifest["metadata"]["pending_count"] == manifest["metadata"]["overlong_count"] == 33
    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert len(review["items"]) == 33
    for row, original in zip(review["items"], originals, strict=True):
        _assert_origin(row, original)
        assert row["split"] == "pending_not_in_dataset" and row["labels"] == []
        assert row["human_decision"] is None


@pytest.mark.asyncio
async def test_prepare_resumes_filtered_batches_without_source_index_drift(tmp_path, monkeypatch):
    """跨批隔离长题后恢复仍复用成功原件，短题和待复核来源不重复、不漂移。

    参数：tmp_path为隔离目录，monkeypatch为网络阻断器。返回：None。
    异常：故意模拟中断RuntimeError；其余断言失败或准备错误。
    """
    _block_network(monkeypatch)
    texts = [f"第{i}款鞋子的尺码怎么选？" for i in range(35)]
    for index in (0, 12, 34):
        texts.insert(index, f"长题{index}：" + "商品尺寸说明。" * 230)
    source, output = tmp_path / "pool.json", tmp_path / "dataset"
    originals = _export_source(source, texts)
    model = ReplyModel(fail_on_call=2)
    with pytest.raises(RuntimeError, match="模拟第二批网络中断"):
        await prepare_dataset(source, output, per_label=0, multi=0, max_calls=2,
                              labeller=TopicLabeller(model, report_dir=tmp_path / "raw", max_attempts=1))
    assert not output.exists()
    checkpoint = tmp_path / "dataset.resume/batch-000.json"
    checkpoint_before = checkpoint.read_bytes()
    assert len(model.requests[0]["items"]) == 32
    restarted = ReplyModel()
    manifest = await prepare_dataset(source, output, per_label=0, multi=0, max_calls=2,
                                     labeller=TopicLabeller(restarted, report_dir=tmp_path / "raw", max_attempts=1))
    assert checkpoint.read_bytes() == checkpoint_before
    assert len(restarted.requests) == 1
    assert [item["index"] for item in restarted.requests[0]["items"]] == [0, 1, 2]
    rows = _read_parts(output)
    assert len(rows) == len({row["id"] for row in rows}) == 35
    expected = {original["question_id"]: original for original in originals if not original["text"].startswith("长题")}
    assert {row["pool_ids"][0] for row in rows} == set(expected)
    for row in rows:
        _assert_origin(row, expected[row["pool_ids"][0]])
    pending = json.loads((tmp_path / "dataset.resume/pending.json").read_text(encoding="utf-8"))
    assert len(pending) == len({row["id"] for row in pending}) == 3
    assert {row["pool_ids"][0] for row in pending} == {"1", "13", "35"}
    assert manifest["metadata"]["pending_count"] == manifest["metadata"]["overlong_count"] == 3
    assert validate_dataset(output)["total"] == 35


@pytest.mark.asyncio
async def test_prepare_keeps_nonempty_disputed_labels_in_experimental_dataset(tmp_path, monkeypatch):
    """合法非空争议预标保留现有实验准入规则，并继续提供抽审记录。

    参数：tmp_path为隔离目录，monkeypatch为网络阻断器。返回：None。
    异常：断言失败、实际准备校验或IO错误。
    """
    _block_network(monkeypatch)
    source, output = tmp_path / "pool.json", tmp_path / "dataset"
    _export_source(source, ["鞋子的尺码怎么选？"])
    manifest = await prepare_dataset(source, output, per_label=0, multi=0,
                                     labeller=TopicLabeller(ReplyModel(disputed=True), report_dir=tmp_path / "raw", max_attempts=1))
    rows = _read_parts(output)
    assert len(rows) == 1 and rows[0]["labels"] == ["尺码"] and rows[0]["needs_review"] is True
    assert manifest["metadata"]["pending_count"] == 0
    assert manifest["human_review_status"] == "pending"
    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert review["items"][0]["labels"] == ["尺码"]
    assert review["items"][0]["split"] != "pending_not_in_dataset"
