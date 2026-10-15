"""真实上游大模型端到端在线验证与标注样本评测脚本。

本脚本直接调用当前应用配置（.env）中的真实上游大语言模型（如 deepseek-v4.1-flash），
通过 FastAPI ASGI 客户端全链路发起真实 HTTP/SSE 请求，执行两大核心业务验证：
1. 售后信息结构化提取验证：对 5 条具有代表性的标注售后文本执行抽取（POST /api/extract），
   严格校验订单号提取、无单号输出 None、诉求类别归纳（退款/换货/维修/投诉/其他）及期望解决方案。
2. 客服多轮会话流式交互与记忆继承验证：对 SSE 流式对话端点（POST /api/chat）发起连续 2 轮对话，
   核验增量 Token 流式传输、标准协议帧（[DONE]）完结以及第二轮对首轮上下文（"猫粮"）的准确保留。

运行方式：
    uv run python tests/verify_ch01_live.py
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# 将工程根目录添加至 sys.path，支持通过 python tests/verify_ch01_live.py 独立直接运行
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app


# 标注售后测试样本数据集契约
LABELED_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "完整订单号 + 退款诉求",
        "input": "我上周买的猫爬架,订单号 MH20260701123,到货就散架了,我要退款",
        "expected_order_id": "MH20260701123",
        "expected_request_type": "退款",
    },
    {
        "id": 2,
        "name": "无订单号 + 换货诉求 (边界用例)",
        "input": "买的猫砂盆尺寸太大了放不下，我想换个小一号的型号，包装还没拆",
        "expected_order_id": None,
        "expected_request_type": "换货",
    },
    {
        "id": 3,
        "name": "包含订单号 + 返厂维修诉求",
        "input": "之前买的自动喂食器不出粮了，电机咔咔响，订单号 SN88392019，还在一年质保期内，麻烦帮我安排返厂维修",
        "expected_order_id": "SN88392019",
        "expected_request_type": "维修",
    },
    {
        "id": 4,
        "name": "无订单号 + 服务态度投诉诉求",
        "input": "你们家物流太慢了！说好三天到结果拖了一星期，客服态度还极其恶劣，我要投诉你们工号8012的客服！",
        "expected_order_id": None,
        "expected_request_type": "投诉",
    },
    {
        "id": 5,
        "name": "口语化复杂场景 + 咨询发票与会员积分",
        "input": "我想咨询一下你们家会员积分怎么兑换商品，另外上个月买的东西能开增值税专用发票吗？",
        "expected_order_id": None,
        "expected_request_type": "其他",
    },
]


def get_current_time_str() -> str:
    """获取当前时间的格式化字符串，用于控制台标准日志输出。

    返回值:
        str: 格式为 YYYY-MM-DD HH:MM:SS 的时间字符串。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_step(message: str) -> None:
    """向控制台输出带有时间戳与标准格式的业务步骤日志。

    参数:
        message (str): 需要打印的日志内容。

    返回值:
        None: 无返回值。
    """
    print(f"[{get_current_time_str()}] {message}")


async def verify_extract_samples(client: AsyncClient) -> bool:
    """验证 5 条标注售后样本对 POST /api/extract 接口的真实抽取效果。

    依次向接口发送标注文本，解析响应体字段，核验订单号（存在或为 None）、
    诉求分类枚举值（退款/换货/维修/投诉/其他）以及期望方案的有效性。

    参数:
        client (AsyncClient): 已绑定 FastAPI ASGI Transport 的异步 HTTP 客户端。

    返回值:
        bool: 若 5 条样本全部符合标注预期则返回 True，若有任一条不匹配则返回 False。

    异常:
        Exception: 捕获网络调用或 JSON 解析过程中的异常并记录日志。
    """
    log_step("======================================================================")
    log_step("【Part 1】开始验证 5 条标注售后样本结构化提取 (POST /api/extract)")
    log_step("======================================================================")

    all_passed = True

    for sample in LABELED_SAMPLES:
        sample_id = sample["id"]
        sample_name = sample["name"]
        text_input = sample["input"]
        expected_oid = sample["expected_order_id"]
        expected_type = sample["expected_request_type"]

        log_step(f"--> [样本 {sample_id}] 测试用例: {sample_name}")
        log_step(f"    输入文本: \"{text_input}\"")

        try:
            # 向提取接口发送 POST 请求
            response = await client.post("/api/extract", json={"text": text_input}, timeout=60.0)

            if response.status_code != 200:
                log_step(f"    [X] 失败: 接口响应非 200 状态码 (status_code={response.status_code}, body={response.text})")
                all_passed = False
                continue

            data = response.json()
            actual_oid = data.get("order_id")
            actual_type = data.get("request_type")
            solution = data.get("expected_solution")

            # 校验订单号与诉求分类
            oid_matches = (actual_oid == expected_oid)
            type_matches = (actual_type == expected_type)
            solution_valid = bool(solution and solution.strip())

            if oid_matches and type_matches and solution_valid:
                log_step(
                    f"    [✓] 通过: order_id={actual_oid!r} (预期 {expected_oid!r}), "
                    f"request_type={actual_type!r} (预期 {expected_type!r}), "
                    f"expected_solution=\"{solution}\""
                )
            else:
                log_step("    [X] 校验不匹配:")
                if not oid_matches:
                    log_step(f"        - 订单号不匹配: 实际 {actual_oid!r} != 预期 {expected_oid!r}")
                if not type_matches:
                    log_step(f"        - 诉求类型不匹配: 实际 {actual_type!r} != 预期 {expected_type!r}")
                if not solution_valid:
                    log_step(f"        - 期望方案无效或为空: 实际 {solution!r}")
                all_passed = False

        except Exception as exc:
            log_step(f"    [X] 调用抛出异常: {type(exc).__name__}: {str(exc)}")
            all_passed = False

    return all_passed


async def verify_chat_stream_and_context(client: AsyncClient, session_id: str) -> bool:
    """验证基于 SSE 协议的多轮客服对话 (POST /api/chat) 流式输出与上下文继承能力。

    分两轮连续与客服进行问答：
    - 轮次 1: 提问「你们家猫粮发什么快递？」，验证 SSE 逐 chunk 推送并以 [DONE] 正常完结。
    - 轮次 2: 提问「我刚才问的是什么东西的快递？」，验证流式输出且回答中必须包含「猫粮」关键词，
      以证明上下文跨轮次准确记忆。

    参数:
        client (AsyncClient): 已绑定 FastAPI ASGI Transport 的异步 HTTP 客户端。
        session_id (str): 用于串联本组多轮对话的唯一会话标识符。

    返回值:
        bool: 当两轮对话的流式传输、结束标志及第二轮上下文断言均满足时返回 True，否则返回 False。

    异常:
        Exception: 捕获流式消费过程中的网络或解码异常。
    """
    log_step("======================================================================")
    log_step("【Part 2】开始验证多轮客服对话 SSE 流式输出与上下文记忆 (POST /api/chat)")
    log_step("======================================================================")

    # 轮次 1: 首轮提问
    round1_msg = "你们家猫粮发什么快递？"
    log_step(f"--> [轮次 1] 发送提问: \"{round1_msg}\" (session_id={session_id})")

    r1_chunks: List[str] = []
    r1_got_done = False

    try:
        async with client.stream(
            "POST",
            "/api/chat",
            json={"session_id": session_id, "message": round1_msg},
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                log_step(f"    [X] 失败: 轮次 1 响应状态码为 {response.status_code}")
                return False

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload_str = line[6:].strip()
                    if payload_str == "[DONE]":
                        r1_got_done = True
                        break
                    try:
                        chunk_dict = json.loads(payload_str)
                        delta = chunk_dict.get("delta", "")
                        r1_chunks.append(delta)
                    except json.JSONDecodeError:
                        pass

        r1_full_reply = "".join(r1_chunks)
        log_step(f"    [✓] 轮次 1 收到完整回复: \"{r1_full_reply}\"")
        log_step(f"    [✓] 轮次 1 收到流式结束标识 [DONE]: {r1_got_done}")

        if not r1_got_done or not r1_full_reply:
            log_step("    [X] 轮次 1 失败: 未收到 [DONE] 标识或回复为空")
            return False

    except Exception as exc:
        log_step(f"    [X] 轮次 1 发生异常: {type(exc).__name__}: {str(exc)}")
        return False

    # 轮次 2: 验证首轮上下文记忆
    round2_msg = "我刚才问的是什么东西的快递？"
    log_step(f"--> [轮次 2] 发送追问: \"{round2_msg}\" (session_id={session_id})")

    r2_chunks: List[str] = []
    r2_got_done = False

    try:
        async with client.stream(
            "POST",
            "/api/chat",
            json={"session_id": session_id, "message": round2_msg},
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                log_step(f"    [X] 失败: 轮次 2 响应状态码为 {response.status_code}")
                return False

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    payload_str = line[6:].strip()
                    if payload_str == "[DONE]":
                        r2_got_done = True
                        break
                    try:
                        chunk_dict = json.loads(payload_str)
                        delta = chunk_dict.get("delta", "")
                        r2_chunks.append(delta)
                    except json.JSONDecodeError:
                        pass

        r2_full_reply = "".join(r2_chunks)
        log_step(f"    [✓] 轮次 2 收到完整回复: \"{r2_full_reply}\"")
        log_step(f"    [✓] 轮次 2 收到流式结束标识 [DONE]: {r2_got_done}")

        # 核心上下文断言：必须准确命中「猫粮」
        has_context_keyword = "猫粮" in r2_full_reply
        log_step(f"    [✓] 轮次 2 上下文关键词「猫粮」命中检测: {has_context_keyword}")

        if not r2_got_done:
            log_step("    [X] 轮次 2 失败: 未收到 [DONE] 标识")
            return False

        if not has_context_keyword:
            log_step("    [X] 轮次 2 失败: 回复中未找到上下文关键实体「猫粮」")
            return False

    except Exception as exc:
        log_step(f"    [X] 轮次 2 发生异常: {type(exc).__name__}: {str(exc)}")
        return False

    return True


async def main() -> int:
    """真实模型端到端评测主执行入口。

    初始化 ASGI 测试客户端，依次执行 5 条售后样本抽取评测和 2 轮上下文流式评测，
    最后打印格式化验收报告摘要。

    返回值:
        int: 若全部验证通过返回 0，若有任一项失败返回 1。
    """
    print("\n" + "=" * 70)
    print("      MewHelp Chapter 01 真实大模型在线端到端能力验证评测")
    print("=" * 70)
    print(f"上游模型端点: {settings.chat_base_url}")
    print(f"当前评测模型: {settings.chat_model}")
    print(f"上下文 Token 预算: {settings.chat_token_budget}")
    print("=" * 70 + "\n")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. 验证 5 条售后样本提取
        extract_ok = await verify_extract_samples(client)

        # 2. 验证多轮流式对话上下文
        session_id = "live-verify-session"
        chat_ok = await verify_chat_stream_and_context(client, session_id=session_id)

    # 输出最终验收报告摘要
    print("\n" + "=" * 70)
    print("                      全链路端到端验收总结报告")
    print("=" * 70)
    print(f"1. 售后信息结构化提取 (POST /api/extract): {'[PASS 全部通过]' if extract_ok else '[FAIL 未通过]'}")
    print(f"2. 多轮对话流式输出与上下文记忆 (POST /api/chat): {'[PASS 全部通过]' if chat_ok else '[FAIL 未通过]'}")
    print("=" * 70)

    if extract_ok and chat_ok:
        print("[✓✓✓] 所有核心业务链路在真实上游大模型评测中 100% 通过验证！\n")
        return 0
    else:
        print("[XXX] 真实大模型在线验证存在未达标项，请根据上述日志排查！\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
