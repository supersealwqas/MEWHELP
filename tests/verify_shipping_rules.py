"""只读真实验证邮费问法与未知问题，复用生产理解、检索、判闸和回答。

通过命令行显式运行；不进入普通 pytest、不写聊天或知识库。评分参数可冻结回放，
协议差异完整记录，因此该脚本本身不代表策略已发布或重新校准。
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from app.config import Settings
from app.evaluation.production import production_runtime
from app.knowledge.confidence import EvidencePolicy
from app.knowledge.pipeline import production_protocols


CASES = (
    {"id": "shipping-rules", "query": "邮费规则是什么", "refused": False, "required": ["99", "6", "12"]},
    {"id": "shipping-colloquial", "query": "邮费怎么收", "refused": False, "required": ["99", "6", "12"]},
    {"id": "shipping-standard", "query": "运费怎么算", "refused": False, "required": ["99", "6", "12"]},
    {"id": "free-shipping", "query": "包邮条件是什么", "refused": False, "required": ["99", "偏远"]},
    {"id": "shipping-installments", "query": "运费能分期吗", "refused": True},
    {"id": "unknown-shipping-model", "query": "MH-W99的运费补贴规则是什么", "refused": True},
    {"id": "unknown-product", "query": "MH-LP300的容量和适用体重是多少", "refused": True},
    {"id": "battery-assumption", "query": "MH-CAM1不插电时自带电池能撑几小时", "refused": True},
    {"id": "realtime-stock", "query": "生日礼盒今天剩多少库存", "refused": True},
    {"id": "invoice-conditions", "query": "收货20天了，企业账户想开电子专票，需要哪些资料，通过后多久能开出？",
     "refused": False, "required": ["纳税人识别号", "开户行", "对公账号", "工作日"],
     "required_query_terms": ["20"]},
    {"id": "return-shipping", "query": "无理由退货邮费谁出", "refused": False, "required": ["买家"]},
)

# 仅供验收判定的预期事实，不传给模型；金额须紧邻对应规则，不能靠子串凑齐。
AMOUNT_FACTS = {
    "99": r"(?<!未)(?:满|达到)99(?:\.0+)?元[^。；]{0,16}包邮|包邮[^。；]{0,20}(?<!未)满99(?:\.0+)?元",
    "6": r"未满[^。；]{0,30}收(?:取)?6(?:\.0+)?元(?:基础)?运费|基础运费(?:为|是|：|:)?6(?:\.0+)?元",
    "12": r"偏远[^。；]{0,60}(?:加收|另加|附加运费(?:为|是|：|:)?)12(?:\.0+)?元",
}


def inspect_result(case: dict, result: dict) -> list[str]:
    """按原问答案与完整引用判定，不把检索相关或HTTP成功当作回答成功。

    参数：case仅供本地判定的预期；result生产运行时结果。
    返回：失败原因列表，空列表表示符合预期。
    异常：KeyError，运行时未提供必需结果字段。
    """
    failures = []
    if result["refused"] != case["refused"]:
        failures.append("refusal_mismatch")
    for value in case.get("required_query_terms", []):
        if value not in result["standard_query"]:
            failures.append("missing_query_condition:" + value)
    if not case["refused"]:
        if not result["citations"]:
            failures.append("missing_citations")
        # 空白和强调标记不改变事实；保留句号和分号，防止跨段拼出不存在的金额规则。
        answer = re.sub(r"[\s*]", "", result["answer"])
        # 相关补充不等于错误；如提到无理由退货，承担方必须与已核实政策一致。
        if "无理由退货" in answer and not re.search(
                r"无理由退货[^。；]{0,20}(?:买家|消费者|顾客)[^。；]{0,6}(?:承担|支付)|"
                r"(?:买家|消费者|顾客)[^。；]{0,10}(?:承担|支付)[^。；]{0,10}无理由退货", answer):
            failures.append("incorrect_return_shipping_payer")
        for value in case.get("required", []):
            pattern = AMOUNT_FACTS.get(value)
            if not (re.search(pattern, answer) if pattern else value in answer):
                failures.append("missing_answer_fact:" + value)
    return failures


async def verify(args) -> int:
    """逐题调用真实生产边界，保存完整证据后返回可自动判定的退出码。

    参数：args含output_dir、policy和可选case；目录必须尚不存在。
    返回：0表示全部通过，1表示误答、拒答不符或运行故障。
    异常：文件或依赖初始化错误透传；单题错误记录后继续。
    """
    config = Settings()
    policy = EvidencePolicy.model_validate_json(args.policy.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocols = production_protocols(config)
    manifest = {"mode": "frozen-parameter-regression", "policy": policy.model_dump(mode="json"),
                "protocols": protocols, "protocols_match": protocols == policy.protocols}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    async with production_runtime(config, target="application-readonly") as (runtime, _, _, _):
        # 显式冻结参数做候选回放；不修改制品，也不把新协议冒充旧协议。
        runtime.frozen_policy = policy
        with (args.output_dir / "results.jsonl").open("x", encoding="utf-8") as stream:
            for case in CASES:
                if args.case and case["id"] != args.case:
                    continue
                row = {"id": case["id"], "query": case["query"]}
                try:
                    # 期望答案和预期拒答标记绝不传给模型或检索器。
                    result = await runtime.execute(case["query"], {}, retrieval_only=False)
                    row.update({key: result[key] for key in (
                        "standard_query", "queries", "route", "answer", "refused", "gate", "budget")})
                    row["citations"] = [item.model_dump(mode="json") for item in result["citations"]]
                    row["selected"] = [item.model_dump(mode="json") for item in result["selected"]]
                    row["ranked"] = [item.model_dump(mode="json") for item in result["ranked"]]
                    row["failures"] = inspect_result(case, result)
                except Exception as error:
                    row.update(error_type=type(error).__name__, failures=["runtime_error"])
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps({key: row.get(key) for key in (
                    "id", "standard_query", "answer", "refused", "failures", "error_type")}, ensure_ascii=False), flush=True)
    summary = {"total": len(rows), "passed": sum(not row["failures"] for row in rows),
               "failed_ids": [row["id"] for row in rows if row["failures"]]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return int(bool(summary["failed_ids"]))


def main() -> int:
    """解析显式真实验收参数，避免普通单测意外调用模型。

    参数：无，读取命令行；返回：进程退出码。
    异常：参数、文件和运行时初始化错误透传。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("data/knowledge/evidence-policy.json"))
    parser.add_argument("--case", choices=[case["id"] for case in CASES])
    return asyncio.run(verify(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
