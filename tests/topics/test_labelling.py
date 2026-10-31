"""真实结构化边界上的模型输出回配与失败测试。"""

import pytest
from langchain_core.messages import AIMessageChunk
from app.topics.labelling import TopicLabeller


class Model:
    """仅替代付费网络流，保留实际工具解析。"""

    def __init__(self, rows):
        """保存上游结果。参数：rows工具条目；返回：无；异常：无。"""
        self.rows, self.calls, self.messages = rows, 0, []

    def bind_tools(self, tools, **kwargs):
        """捕获工具边界。参数：工具、绑定参数；返回：模型；异常：无。"""
        self.name = tools[0]["function"]["name"]
        return self

    async def astream(self, messages):
        """模拟外部流。参数：消息；返回：分片流；异常：无。"""
        self.calls += 1
        self.messages = messages
        yield AIMessageChunk(content="", tool_calls=[dict(name=self.name, id="call1", args={"items": self.rows})], response_metadata={"finish_reason": "tool_calls"})


@pytest.mark.asyncio
async def test_invalid_ids_and_labels_are_rejected(tmp_path):
    """错序号和非法标签均不能成为成功数据。参数：临时目录；返回：无；异常：断言失败。"""
    for row in (dict(index=2, text="尺寸多少", labels=["尺码"], family="one", needs_review=False, review_reason=""),
                dict(index=0, text="尺寸多少", labels=["瞎编"], family="one", needs_review=False, review_reason="")):
        model = Model([row])
        with pytest.raises(ValueError):
            await TopicLabeller(model, report_dir=tmp_path / str(row["index"])).label(["尺寸多少"])
        assert model.calls <= 2


@pytest.mark.asyncio
async def test_input_is_redacted_and_output_is_model_prelabel(tmp_path):
    """外发隐私消除，模型结果不冒充人工确认。参数：临时目录；返回：无；异常：断言失败。"""
    model = Model([dict(index=0, text="[电话]，XL不退，问尺寸", labels=["尺码"], family="one", needs_review=False, review_reason="")])
    result = await TopicLabeller(model, report_dir=tmp_path).label(["13812345678，XL不退，问尺寸"])
    assert result[0]["labels"] == ["尺码"] and result[0]["annotation_status"] == "model_prelabel"
    assert "13812345678" not in str(model.messages)
    assert list(tmp_path.glob("*.json"))


@pytest.mark.asyncio
async def test_correction_cannot_change_numbers_or_negation(tmp_path):
    """纠错不能删否定或改变业务数字。参数：临时目录；返回：无；异常：断言失败。"""
    model = Model([dict(index=0, text="XL要退，3件", labels=["退换货"], family="one", needs_review=False, review_reason="")])
    with pytest.raises(ValueError):
        await TopicLabeller(model, report_dir=tmp_path).label(["XL不退，2件"])


@pytest.mark.asyncio
@pytest.mark.parametrize("original,changed", [
    ("XL不退，L退", "XL退，L不退"),
    ("2件退货，3件不退", "3件退货，2件不退"),
    ("XL不退，L换", "XL不换，L退"),
])
async def test_correction_cannot_rebind_protected_business_meaning(tmp_path, original, changed):
    """相同保护词数或顺序也不能改变诉求绑定。参数：目录、原话、篡改文本；返回：无；异常：断言失败。"""
    model = Model([dict(index=0, text=changed, labels=["退换货"], family="one", needs_review=False, review_reason="")])
    with pytest.raises(ValueError):
        await TopicLabeller(model, report_dir=tmp_path).label([original])


@pytest.mark.asyncio
async def test_explicit_safe_typo_and_punctuation_corrections_remain_available(tmp_path):
    """限定错字和等价标点可以修正。参数：临时目录；返回：无；异常：断言失败。"""
    model = Model([dict(index=0, text="发票怎么补开？", labels=["发票"], family="one", needs_review=False, review_reason="")])
    output = await TopicLabeller(model, report_dir=tmp_path).label(["发漂怎么补开?"])
    assert output[0]["text"] == "发票怎么补开？" and output[0]["original_text"] == "发漂怎么补开?"


@pytest.mark.asyncio
async def test_ambiguous_text_is_pending_and_generated_pii_is_redacted(tmp_path):
    """含糊问题不假补其他，模型生成个人信息仍须脱敏。参数：临时目录；返回：无；异常：断言失败。"""
    unclear = Model([dict(index=0, text="？", labels=[], family="one", needs_review=True, review_reason="无诉求信息")])
    pending = await TopicLabeller(unclear, report_dir=tmp_path / "pending").label(["？"])
    assert pending[0]["labels"] == [] and pending[0]["needs_review"]
    generated = Model([dict(index=0, text="电话13812345678，运费多少钱", labels=["运费"], family="one", needs_review=False, review_reason="")])
    outputs = await TopicLabeller(generated, report_dir=tmp_path / "generated").generate(["运费"], 1, "a")
    assert "13812345678" not in outputs[0]["text"]


@pytest.mark.asyncio
async def test_oversize_inputs_never_reach_external_model(tmp_path):
    """限制外发输入量，避免无界请求。参数：临时目录；返回：无；异常：断言失败。"""
    model = Model([dict(index=0, text="尺寸多少", labels=["尺码"], family="one", needs_review=False, review_reason="")])
    with pytest.raises(ValueError):
        await TopicLabeller(model, report_dir=tmp_path).label(["问" * 1501])
    assert model.calls == 0


@pytest.mark.asyncio
async def test_invalid_raw_result_is_preserved(tmp_path):
    """输出超限也保留已收到的原始分片。参数：临时目录；返回：无；异常：断言失败。"""
    import json
    model = Model([dict(index=0, text="问" * 65000, labels=["尺码"], family="one", needs_review=False, review_reason="")])
    with pytest.raises(ValueError):
        await TopicLabeller(model, report_dir=tmp_path).label(["尺寸多少"])
    records = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("*.json")]
    assert records and all(record["status"] == "failed" and record["raw_chunks"] for record in records)


@pytest.mark.parametrize("original,private,business", [
    ("电话：010-88886666，我想退货", "010-88886666", "我想退货"),
    ("电话(010)88886666 转123，我想退货", "88886666", "我想退货"),
    ("电话+86 13812345678，XL不退，2026-11-01问M200型号", "13812345678", "XL不退，2026-11-01问M200型号"),
    ("我的订单 12345678901234567890 到哪了", "12345678901234567890", "到哪了"),
    ("姓名是张三，地址是北京市朝阳区示例路1号，请问能改地址吗", "张三", "请问能改地址吗"),
    ("联系人张三，收货地址是北京市朝阳区示例路1号，XL不退", "北京市朝阳区示例路1号", "XL不退"),
])
async def test_explicit_private_fields_never_reach_prelabel_payload(tmp_path, original, private, business):
    """拦截实际预标载荷验证脱敏边界。参数：tmp_path隔离目录、original虚构原话、private私密片段、business业务原意；返回：None；异常：断言失败。"""
    from app.topics.data import clean_question
    model = Model([dict(index=0, text=clean_question(original), labels=["物流"], family="one", needs_review=False, review_reason="")])
    await TopicLabeller(model, report_dir=tmp_path).label([original])
    payload = str(model.messages)
    assert private not in payload
    assert "张三" not in payload and "北京市朝阳区示例路1号" not in payload
    assert business in payload
