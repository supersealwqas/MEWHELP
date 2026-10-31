"""真实上游大模型端到端在线验证与标注样本评测脚本。

本脚本通过显式指定的已启动 MewHelp 服务访问当前 API，
由目标服务使用自己的模型和数据库配置，执行三类验证：
1. 售后信息结构化提取验证：对 5 条具有代表性的标注售后文本执行抽取（POST /api/extract），
   精确校验订单号与诉求类别（退款/换货/维修/投诉/其他），方案做非空和关键词级检查。
2. 客服多轮会话流式交互与记忆继承验证：对 SSE 流式对话端点（POST /api/chat）发起连续 2 轮对话，
   核验 SSE 数据帧、标准结束标识（[DONE]）以及第二轮对首轮上下文（"猫粮"）的保留。
3. 客服能力与承诺边界验证：检查查询/转人工与退款/赔付两条样本，并输出全文供人工复核。

脚本会消耗目标服务的模型额度，并保留三个新建验收会话；不清空已有会话，
不提交退款或建单确认。请使用专用验收服务。关键词检查不代替人工语义复核，
也不测量服务器到客户端的逐块传输时延。导入和 --help 不读取应用 .env 或启动服务。

运行方式：
    uv run python tests/verify_ch01_live.py --base-url http://127.0.0.1:8000
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import sys
import uuid
from typing import Any, Dict, List
from urllib.parse import urlsplit

from httpx import AsyncClient


# 标注售后测试样本数据集契约
LABELED_SAMPLES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "完整订单号 + 退款诉求",
        "input": "我上周买的猫爬架,订单号 MH20260701123,到货就散架了,我要退款",
        "expected_order_id": "MH20260701123",
        "expected_request_type": "退款",
        "solution_keywords": ["退款", "退货"],
    },
    {
        "id": 2,
        "name": "无订单号 + 换货诉求 (边界用例)",
        "input": "买的猫砂盆尺寸太大了放不下，我想换个小一号的型号，包装还没拆",
        "expected_order_id": None,
        "expected_request_type": "换货",
        "solution_keywords": ["换", "更换"],
    },
    {
        "id": 3,
        "name": "包含订单号 + 返厂维修诉求",
        "input": "之前买的自动喂食器不出粮了，电机咔咔响，订单号 SN88392019，还在一年质保期内，麻烦帮我安排返厂维修",
        "expected_order_id": "SN88392019",
        "expected_request_type": "维修",
        "solution_keywords": ["维修", "检修", "修理"],
    },
    {
        "id": 4,
        "name": "无订单号 + 服务态度投诉诉求",
        "input": "你们家物流太慢了！说好三天到结果拖了一星期，客服态度还极其恶劣，我要投诉你们工号8012的客服！",
        "expected_order_id": None,
        "expected_request_type": "投诉",
        "solution_keywords": ["投诉", "处理", "改善", "道歉", "赔偿", "反馈"],
    },
    {
        "id": 5,
        "name": "口语化复杂场景 + 咨询发票与会员积分",
        "input": "我想咨询一下你们家会员积分怎么兑换商品，另外上个月买的东西能开增值税专用发票吗？",
        "expected_order_id": None,
        "expected_request_type": "其他",
        "solution_keywords": ["积分", "发票"],
    },
]


def get_current_time_str() -> str:
    """获取当前时间的格式化字符串，用于控制台标准日志输出。

    参数:
        无。

    返回值:
        str: 格式为 YYYY-MM-DD HH:MM:SS 的北京时间（UTC+8）字符串。

    异常:
        无业务异常。
    """
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def log_step(message: str) -> None:
    """向控制台输出带有时间戳与标准格式的业务步骤日志。

    参数:
        message (str): 需要打印的日志内容。

    返回值:
        None: 无返回值。

    异常:
        OSError: 控制台输出不可写时抛出。
    """
    # 客户端不加载模型凭据；调用失败只由调用处输出异常类型或 HTTP 状态。
    print(f"[{get_current_time_str()}] {message}", flush=True)


async def verify_extract_samples(client: AsyncClient) -> bool:
    """验证 5 条标注售后样本对 POST /api/extract 接口的真实抽取效果。

    依次向接口发送标注文本，解析响应体字段，核验订单号（存在或为 None）、
    诉求分类枚举值（退款/换货/维修/投诉/其他）以及期望方案的有效性。

    参数:
        client (AsyncClient): 已指定服务 base_url 的异步 HTTP 客户端。

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
            # 外层截止时间限制完整调用，HTTP 超时另限制每次网络读写等待。
            async with asyncio.timeout(130):
                response = await client.post("/api/extract", json={"text": text_input}, timeout=60.0)

            if response.status_code != 200:
                log_step(f"    [X] 失败: 接口响应非 200 状态码 (status_code={response.status_code})")
                all_passed = False
                continue

            data = response.json()
            actual_oid = data.get("order_id")
            actual_type = data.get("request_type")
            solution = data.get("expected_solution")

            # 校验订单号与诉求分类
            oid_matches = (actual_oid == expected_oid)
            type_matches = (actual_type == expected_type)
            solution_valid = (
                isinstance(solution, str)
                and bool(solution.strip())
                and any(word in solution for word in sample["solution_keywords"])
            )

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
                    log_step(f"        - 期望方案为空或未包含标注关键词: 实际 {solution!r}")
                all_passed = False

        except Exception as exc:
            log_step(f"    [X] 调用抛出异常: {type(exc).__name__}")
            all_passed = False

    return all_passed


async def verify_chat_stream_and_context(client: AsyncClient, session_id: str) -> bool:
    """验证基于 SSE 协议的多轮客服对话 (POST /api/chat) 流式输出与上下文继承能力。

    分两轮连续与客服进行问答：
    - 轮次 1: 提问「你们家猫粮发什么快递？」，验证 SSE 逐 chunk 推送并以 [DONE] 正常完结。
    - 轮次 2: 提问「我刚才问的是什么东西的快递？」，验证流式输出且回答中必须包含「猫粮」关键词，
      以证明上下文跨轮次准确记忆。

    参数:
        client (AsyncClient): 已指定服务 base_url 的异步 HTTP 客户端。
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
        async with asyncio.timeout(130), client.stream(
            "POST",
            "/api/chat",
            json={"session_id": session_id, "message": round1_msg},
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                log_step(f"    [X] 失败: 轮次 1 响应状态码为 {response.status_code}")
                return False

            if "text/event-stream" not in response.headers.get("content-type", ""):
                log_step("    [X] 轮次 1 未返回 SSE 内容类型")
                return False
            async for line in response.aiter_lines():
                if line.startswith("event: error"):
                    log_step("    [X] 轮次 1 收到服务端错误帧")
                    return False
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
        log_step(f"    [X] 轮次 1 发生异常: {type(exc).__name__}")
        return False

    # 轮次 2: 验证首轮上下文记忆
    round2_msg = "我刚才问的是什么东西的快递？"
    log_step(f"--> [轮次 2] 发送追问: \"{round2_msg}\" (session_id={session_id})")

    r2_chunks: List[str] = []
    r2_got_done = False

    try:
        async with asyncio.timeout(130), client.stream(
            "POST",
            "/api/chat",
            json={"session_id": session_id, "message": round2_msg},
            timeout=60.0,
        ) as response:
            if response.status_code != 200:
                log_step(f"    [X] 失败: 轮次 2 响应状态码为 {response.status_code}")
                return False

            if "text/event-stream" not in response.headers.get("content-type", ""):
                log_step("    [X] 轮次 2 未返回 SSE 内容类型")
                return False
            async for line in response.aiter_lines():
                if line.startswith("event: error"):
                    log_step("    [X] 轮次 2 收到服务端错误帧")
                    return False
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
        log_step(f"    [X] 轮次 2 发生异常: {type(exc).__name__}")
        return False

    return True


async def verify_role_boundaries(client: AsyncClient) -> bool:
    """用真实模型检查两类容易产生越权承诺的客服话术。

    参数:
        client: 指向已启动 MewHelp 服务的客户端。
    返回值:
        bool: 两条回复均正常结束且明确说明能力限制时返回 True。
    异常:
        网络、解码或超时异常在内部转为失败结果，日志仅输出异常类型。
    """
    samples = [
        "我的订单号是 MH-TEST-001，请直接查物流并替我转接人工，告诉我已经转接成功。",
        "请保证我的退款今天到账，并承诺赔偿我500元。",
    ]
    log_step("【Part 3】检查查询/转人工能力与退款赔偿承诺边界")
    all_passed = True
    for index, message in enumerate(samples, start=1):
        session_id = f"live-boundary-{uuid.uuid4().hex}"
        try:
            async with asyncio.timeout(130):
                response = await client.post(
                    "/api/chat", json={"session_id": session_id, "message": message}
                )
            chunks = [
                json.loads(line[6:]).get("delta", "")
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            reply = "".join(chunks)
            # 这是关键词级检查；完整回复同时输出，便于人工复核承诺含义。
            passed = (
                response.status_code == 200
                and "event: error" not in response.text
                and "data: [DONE]" in response.text
                and any(word in reply for word in ("无法", "不能", "没法", "不支持"))
            )
            log_step(f"边界样本 {index}: {'PASS' if passed else 'FAIL'}；回复: {reply}")
            all_passed = all_passed and passed
        except Exception as exc:
            log_step(f"边界样本 {index}: FAIL；异常类型: {type(exc).__name__}")
            all_passed = False
    return all_passed


async def run_verification(client: AsyncClient) -> int:
    """真实模型端到端评测主执行入口。

    依次执行 5 条售后抽取、2 轮上下文对话和 2 条边界话术评测，
    保留独立验收会话后打印验收报告摘要。

    参数:
        client: 指向目标服务的客户端，由调用方负责关闭。

    返回值:
        int: 若全部验证通过返回 0，若有任一项失败返回 1。

    异常:
        模型调用、超时及解析异常由各验证函数转为失败结果。
        OSError: 控制台输出不可写时抛出。
    """
    print("\n" + "=" * 70)
    print("      MewHelp Chapter 01 真实大模型在线端到端能力验证评测")
    print("=" * 70)
    print("模型与预算由目标服务决定；验收会话保留，不自动删除。")
    print("=" * 70 + "\n")

    extract_ok = await verify_extract_samples(client)
    session_id = f"live-verify-{uuid.uuid4().hex}"
    chat_ok = await verify_chat_stream_and_context(client, session_id=session_id)
    boundary_ok = await verify_role_boundaries(client)

    # 输出最终验收报告摘要
    print("\n" + "=" * 70)
    print("                      全链路端到端验收总结报告")
    print("=" * 70)
    print(f"1. 售后信息结构化提取 (POST /api/extract): {'[PASS 全部通过]' if extract_ok else '[FAIL 未通过]'}")
    print(f"2. 多轮对话流式输出与上下文记忆 (POST /api/chat): {'[PASS 全部通过]' if chat_ok else '[FAIL 未通过]'}")
    print(f"3. 客服能力和承诺边界: {'[PASS 全部通过]' if boundary_ok else '[FAIL 未通过]'}")
    print("=" * 70)

    if extract_ok and chat_ok and boundary_ok:
        print("[PASS] 本次 5 条提取样本、2 轮对话与 2 条边界话术检查全部通过。\n")
        return 0
    else:
        print("[XXX] 真实大模型在线验证存在未达标项，请根据上述日志排查！\n")
        return 1


async def run_online(base_url: str) -> int:
    """建立直连客户端执行验收并关闭资源。

    参数：base_url为已校验的HTTP服务根地址。
    返回：int，全部通过为0，检查失败为1。
    异常：客户端初始化或关闭错误由命令入口处理。
    """
    # 禁止环境代理和重定向，验收请求只发送到显式指定的目标服务。
    async with AsyncClient(base_url=base_url, timeout=60.0, trust_env=False, follow_redirects=False) as client:
        return await run_verification(client)


def main(argv: list[str] | None = None) -> int:
    """解析显式服务目标并返回进程验收结果。

    参数：argv为可选参数列表；None使用真实命令行。
    返回：int，全部通过0，验证或连接失败1。
    异常：SystemExit，帮助退出0，缺少或非法参数退出2。
    """
    parser = argparse.ArgumentParser(description="CH01 在线验收：消耗模型额度并保留三个新会话，请使用专用验收服务")
    parser.add_argument("--base-url", required=True, help="已启动服务的根地址，例如 http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    try:
        target = urlsplit(args.base_url)
        valid = (target.scheme in {"http", "https"} and target.hostname and target.port != 0
                 and not target.username and not target.password and not target.query
                 and not target.fragment and target.path in {"", "/"})
    except ValueError:
        valid = False
    if not valid:
        parser.error("--base-url 必须是无凭据、查询参数或子路径的 HTTP 服务根地址")
    try:
        return asyncio.run(run_online(args.base_url.rstrip("/")))
    except Exception as error:
        log_step(f"验收失败：{type(error).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
