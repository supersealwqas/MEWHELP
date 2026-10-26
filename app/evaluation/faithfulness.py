"""按实际证据全集逐事实核对答案，并把编造个案写入长期台账。

模型裁判独立于生成，不读取预期分数或把 ground-truth 当本轮证据；拒答没有
可核对事实时指标为空而不是1。只有合法裁判确认 unsupported 才写台账，
网络/格式/引用错误不能伪造编造结论或抹掉历史记录。
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings
from app.core.llm import get_chat_model
from app.core.structured import StructuredOutputError, bind_output_tool, collect_tool_output, output_tool, parse_tool_output
from app.knowledge.answering import EvidenceCitation
from app.knowledge.schemas import KnowledgeUnavailable
from app.repositories.faith_cases import FaithCaseDraft, FaithCaseRepository


class FaithfulnessError(KnowledgeUnavailable):
    """裁判模型、格式或证据编号非法，不能把故障当作无编造/满分。"""


class FactJudgment(BaseModel):
    """答案中单个可核对事实的裁判结论，保留具体理由和证据引用。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    claim: str = Field(min_length=1, max_length=2000, description="答案里需要核对的完整事实句")
    supported: bool = Field(description="该事实能否从本轮实际证据获得支持，严格布尔")
    reason: str = Field(min_length=1, max_length=2000, description="支持或不支持的具体理由")
    evidence_numbers: list[Annotated[int, Field(gt=0, strict=True)]] = Field(description="裁判依据的本轮证据编号；不支持时可为空")


class FaithfulnessDecision(BaseModel):
    """严格裁判输出，不由单个总分代替逐条事实核对。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    facts: list[FactJudgment] = Field(max_length=64, description="答案可核对事实及判定，无事实可为空但非拒答会报错")


class FaithfulnessResult(BaseModel):
    """忠实度逐事实指标与台账写入状态，拒答不记虚假满分。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    faithfulness: float | None = Field(default=None, ge=0, le=1, description="支持事实数/总事实，拒答无事实为None")
    supported_count: int = Field(ge=0, description="有证据支持事实数")
    total_count: int = Field(ge=0, description="全部可核对事实数")
    recorded: bool = Field(description="是否已确认写入编造台账")
    decision: FaithfulnessDecision = Field(description="本轮完整逐事实裁判")


_JUDGE_SYSTEM = """你是独立的答案忠实度裁判。只使用下面实际喂给生成模型的证据全集判断，不能使用常识补齐。
把答案拆成可核对事实，每条写claim、supported、reason、evidence_numbers；supported为JSON布尔。
只调用一次submit_faithfulness，参数为 {"facts":[{"claim":"完整事实句","supported":true,"reason":"具体证据依据","evidence_numbers":[1]}]}。
保证退款到账、未知型号能力、遗漏否定条件导致的相反结论等若无证据必须supported=false；证据外的话术不是新依据。
只引用输入已有n。没有事实的拒答facts=[]；真实有事实答案不得省略全部事实。禁止自由文本、Markdown围栏、其他工具调用或额外字段。
"""


class FaithfulnessJudge:
    """通过强制控制工具提交逐事实结论的有界裁判，保留全集证据，失败明确上抛。"""

    def __init__(self, client_factory: Callable[..., Any] = get_chat_model, *, model_name: str | None = None) -> None:
        """保存裁判工厂。参数：client_factory 项目模型工厂、model_name 可选真实模型名；返回：None；异常：无。"""
        self.factory = client_factory
        self.model_name = model_name or settings.chat_model

    async def judge(self, query: str, answer: str, citations: list[EvidenceCitation]) -> FaithfulnessDecision:
        """把本轮问答及全部证据传给独立模型并严格回收逐事实结论。

        参数：query 当前评估原问；answer 已生成完整答案；citations 实际模型证据全集，不含ground-truth或策略名称。
        返回：FaithfulnessDecision，尚需服务端核验引用后统计/写表。
        异常：FaithfulnessError，输入/输出超限或控制调用非法；上游网络异常；取消关闭流。
        """
        payload = json.dumps({"query": query, "answer": answer, "citations": [item.model_dump() for item in citations]}, ensure_ascii=False)
        name = "submit_faithfulness"
        tool = output_tool(FaithfulnessDecision, name)
        # 沿用65000字符上限，并计入新增的工具定义和固定提示，证据全集不能被截短。
        if len(payload) + len(_JUDGE_SYSTEM) + len(json.dumps([tool], ensure_ascii=False)) > 65000:
            raise FaithfulnessError("裁判输入超出完整证据预算，不能截掉证据")
        model = self.factory(streaming=True, temperature=0.0, model=self.model_name, max_tokens=4096)
        messages = [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=payload)]
        try:
            gathered = await collect_tool_output(bind_output_tool(model, tool), messages, max_chars=20000)
            return parse_tool_output(gathered, FaithfulnessDecision, name)
        except StructuredOutputError as error:
            # 协议失败不能成为空事实或满分，业务层只接收合法的逐事实结论。
            raise FaithfulnessError("裁判未提交合法结构化结果") from error


class FaithfulnessService:
    """统计逐事实忠实度，并在确认存在编造时原子更新该题台账。"""

    def __init__(self, judge: FaithfulnessJudge, repository: FaithCaseRepository) -> None:
        """注入独立裁判和仓储。参数：judge 模型裁判、repository 长期台账；返回：None；异常：无。"""
        self.judge = judge
        self.repository = repository

    async def evaluate(self, *, eval_id: str, bucket: str, query: str, strategy: str,
                       answer: str, citations: list[EvidenceCitation], refused: bool = False) -> FaithfulnessResult:
        """核对本轮完整答案，只有 unsupported 结论才写台账。

        参数：eval_id/bucket 为题号/类型；query 原问；strategy 真实检索策略；answer 完整答案；
            citations 模型可见证据全集；refused 显式拒答则不调用裁判、不记忠实度满分。
        返回：FaithfulnessResult，统计、完整判定及是否写表。
        异常：FaithfulnessError，引用/事实非法；上游网络/仓储异常直接传播，不伪造指标。
        """
        if refused:
            return FaithfulnessResult(faithfulness=None, supported_count=0, total_count=0,
                                      recorded=False, decision=FaithfulnessDecision(facts=[]))
        decision = await self.judge.judge(query, answer, citations)
        if not decision.facts:
            raise FaithfulnessError("非拒答答案的裁判没有提供可核对事实")
        numbers = {item.n for item in citations}
        for fact in decision.facts:
            if (not set(fact.evidence_numbers).issubset(numbers)
                    or (fact.supported and not fact.evidence_numbers)):
                raise FaithfulnessError("裁判支持结论缺少有效本轮证据编号")
        unsupported = [fact for fact in decision.facts if not fact.supported]
        if unsupported:
            reason = "\n".join(f"{fact.claim}：{fact.reason}" for fact in unsupported)
            draft = FaithCaseDraft(eval_id=eval_id, bucket=bucket, query=query, strategy=strategy, answer=answer,
                                   reason=reason, citations=citations, judge_model=self.judge.model_name)
            await self.repository.record(draft)
        count = len(decision.facts)
        supported = count - len(unsupported)
        return FaithfulnessResult(faithfulness=supported / count, supported_count=supported, total_count=count,
                                  recorded=bool(unsupported), decision=decision)
