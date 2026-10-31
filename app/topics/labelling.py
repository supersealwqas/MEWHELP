"""复用严格单工具LLM边界，提供脱敏预标、模拟生成和本地原件留痕。"""

import asyncio
import hashlib
import json
import re
from collections import Counter
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.core.structured import bind_output_tool, collect_tool_output, output_tool, parse_tool_output
from app.topics.data import clean_question
from app.topics.taxonomy import load_taxonomy, validate_labels


class TopicItem(BaseModel):
    """单题模型预标：临时序号、纠错正文、家族及争议说明。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    index: int = Field(ge=0, description="本次输入临时序号，必须准确回配")
    text: str = Field(min_length=1, max_length=1500, description="脱敏纠错后的问题，保留否定、数字和当前诉求")
    labels: list[str] = Field(max_length=17, description="权威标签名称；语义不足时为空并标待复核")
    family: str = Field(min_length=1, max_length=80, description="同一生成场景或其改写共享的临时家族名")
    needs_review: bool = Field(description="语义不足、纠错不确定或标签有争议时为真")
    review_reason: str = Field(max_length=300, description="待复核原因；无争议为空字符串")


class TopicOutput(BaseModel):
    """唯一结果工具的受限条目数组。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    items: list[TopicItem] = Field(min_length=1, max_length=64, description="按输入临时序号返回全部条目")


class TopicLabeller:
    """只承担模型预标/生成，不替代人工确认，也不接触业务主键。"""

    def __init__(self, model=None, *, report_dir: Path | None = None, max_attempts: int = 2):
        """初始化有界客户端。参数：model可注入、report_dir本地原件、max_attempts为1或2；返回：无；异常：ValueError。"""
        from app.core.llm import get_chat_model
        if max_attempts not in (1, 2):
            raise ValueError("预标最多尝试两次")
        self.taxonomy = load_taxonomy()
        self.model = model if model is not None else get_chat_model(streaming=True, temperature=.4, max_retries=0, max_tokens=12000, request_timeout=180)
        self.report_dir = Path(report_dir) if report_dir is not None else Path("reports/ch10/topic-labelling") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts

    async def _call(self, payload: dict, count: int, expected_labels: list[str] | None = None) -> list[dict]:
        """发起有界单工具请求并验证业务约束。参数：无主键payload、count数量、期望标签；返回：预标；异常：模型/校验错误。"""
        if not 1 <= count <= 64:
            raise ValueError("每次请求数量须为1至64")
        system = "你是电商主题数据预标员。只标当前原话确实表达的诉求；否定、撤回和纯引用不算。修归保修维修、退归退换货；运费管钱、物流管货。其他排他且不表示不确定。语义不足返回labels=[]、needs_review=true和原因，禁止猜上下文。只纠错格式，不添加新信息；保留否定、数量、型号。不要产生真实姓名、电话、邮箱、地址或订单号。生成时每个独立场景用不同family；同场景改写共享family；不引用或改写taxonomy正反例。主题权威表：" + json.dumps(self.taxonomy, ensure_ascii=False)
        messages = [SystemMessage(content=system), HumanMessage(content=json.dumps(payload, ensure_ascii=False))]
        bound = bind_output_tool(self.model, output_tool(TopicOutput, "submit_topics"))
        for attempt in range(self.max_attempts):
            record = {"request": payload, "attempt": attempt + 1, "usage": None, "model": getattr(self.model, "model_name", None)}
            path = self.report_dir / f"{uuid4().hex}-attempt{attempt + 1}.json"
            chunks = []

            async def recorded_stream(request_messages):
                """捕获失败前分片并关闭原始流。参数：消息；返回：原始流；异常：上游网络或取消。"""
                async with aclosing(bound.astream(request_messages)) as stream:
                    async for chunk in stream:
                        chunks.append(chunk.model_dump(mode="json"))
                        yield chunk

            try:
                gathered = await asyncio.wait_for(collect_tool_output(SimpleNamespace(astream=recorded_stream), messages, max_chars=60000), timeout=210)
                # 原始结果先保存，解析失败也不会丢失模型输出；从不序列化客户端或密钥。
                record.update(raw=gathered.model_dump(mode="json"), usage=gathered.usage_metadata)
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                output = parse_tool_output(gathered, TopicOutput, "submit_topics")
                indices = [item.index for item in output.items]
                if sorted(indices) != list(range(count)):
                    raise ValueError("模型序号遗漏、重复或越界")
                result = []
                for item in sorted(output.items, key=lambda item: item.index):
                    row = item.model_dump()
                    row["text"] = clean_question(row["text"])
                    if payload["task"] == "label":
                        original = payload["items"][item.index]["text"]
                        protected = r"[A-Za-z]+(?:[-_]?[A-Za-z0-9]+)*|\d+(?:\.\d+)?|不|没|未|无|勿|别"
                        if Counter(re.findall(protected, original)) != Counter(re.findall(protected, row["text"])):
                            raise ValueError("模型纠错改变了否定、数字、型号或尺码")
                    if not item.labels:
                        if not item.needs_review or not item.review_reason.strip() or expected_labels is not None:
                            raise ValueError("空标签必须明确进入待复核，生成题不能空标")
                    else:
                        row["labels"] = validate_labels(item.labels, self.taxonomy)
                    if expected_labels is not None and row["labels"] != expected_labels:
                        raise ValueError("模拟结果没有表达请求的全部主题")
                    row["annotation_status"] = "model_prelabel"
                    result.append(row)
                record["status"] = "success"
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                return result
            except Exception as error:
                # 只记异常类型，网络错误字符串可能携带端点/认证信息，不能原样落盘。
                record.update(status="failed", error_type=type(error).__name__)
                record["raw_chunks"] = chunks
                usages = [chunk.get("usage_metadata") for chunk in chunks if chunk.get("usage_metadata")]
                if usages:
                    record["usage"] = usages[-1]
                path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                if attempt + 1 == self.max_attempts:
                    raise
        raise RuntimeError("预标请求未完成")

    async def label(self, texts: list[str]) -> list[dict]:
        """预标已脱敏问题。参数：texts一至64条原话；返回：纠错、标签和待复核；异常：模型或ValueError。"""
        if not isinstance(texts, list) or not 1 <= len(texts) <= 64:
            raise ValueError("预标输入须为1至64条文本数组")
        cleaned = [clean_question(text) for text in texts]
        if any(len(text) > 1500 for text in cleaned):
            raise ValueError("预标输入单题不能超过1500字符")
        result = await self._call({"task": "label", "items": [{"index": i, "text": text} for i, text in enumerate(cleaned)]}, len(cleaned))
        for row in result:
            row["original_text"] = cleaned[row["index"]]
        return result

    async def generate(self, labels: list[str], count: int, seed_tag: str) -> list[dict]:
        """生成独立模拟问题。参数：labels诉求组合、count一至64、seed_tag场景批次；返回：模拟预标；异常：模型/校验错误。"""
        labels = validate_labels(labels, self.taxonomy)
        if not isinstance(seed_tag, str) or not seed_tag:
            raise ValueError("生成必须提供独立批次标记")
        result = await self._call({"task": "generate", "labels": labels, "count": count, "seed_tag": seed_tag,
                                   "instructions": "生成不同商品、时间、口语、否定近邻及售后场景的模拟问题；每条必须表达指定全部标签，序号从0开始。不用固定验收例句作种子。"}, count, labels)
        for row in result:
            row["group_id"] = "synthetic:" + hashlib.sha256((seed_tag + ":" + row["family"]).encode()).hexdigest()[:24]
            row["source"] = "synthetic"
        return result
