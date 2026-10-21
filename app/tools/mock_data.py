"""集中管理有限演示订单的归属、订单快照与按物流单号读取的物流快照。"""

import random
from datetime import datetime, timedelta, timezone


# 未知订单和非本人订单使用同一结果，避免通过错误判断订单是否存在。
NOT_OWNED = {"error": "没有找到您的这笔订单", "code": "order_not_owned"}
_ORDER_OWNERS = {
    "1001": "demo-user", "1002": "demo-user", "1003": "demo-user",
    "1004": "demo-user", "1005": "demo-user", "1006": "demo-user",
    "1007": "demo-user", "A-1001": "demo-user", "2003": "demo-other-user",
}
# 固定省市不消耗随机数，避免新增配送信息改变既有金额、状态或时间。
_ORDER_DESTINATIONS = {
    "1001": "浙江省宁波市", "1002": "新疆维吾尔自治区乌鲁木齐市",
    "1003": "江苏省南京市", "1004": "广东省广州市", "1005": "四川省成都市",
    "1006": "湖北省武汉市", "1007": "福建省厦门市", "A-1001": "安徽省合肥市",
    "2003": "山东省青岛市",
}


def owns_order(user_id: str, order_id: str | None) -> bool:
    """判断已知演示订单是否属于当前用户。

    参数:
        user_id: 由运行环境注入的非空演示用户标识。
        order_id: 已去除首尾空白的订单号；物流号无映射时可以为 None。
    返回值:
        bool: 订单存在且属于该用户时为 True，否则为 False。
    异常:
        无主动抛出的业务异常；不访问网络或数据库。
    """
    return order_id in _ORDER_OWNERS and _ORDER_OWNERS[order_id] == user_id


def _mock_order_snapshot(order_id: str) -> tuple[dict, dict]:
    """按订单号生成一致的订单与物流演示快照。

    参数:
        order_id: 数据模块内部使用的非空订单号，最多 64 字符。
    返回值:
        tuple[dict, dict]: 新生成的订单摘要和物流详情，共用发货状态及物流单号。
    异常:
        无主动抛出的业务异常；不访问网络或数据库。
    """
    # 字符串种子不依赖进程 hash；局部发生器不读取或改变全局随机数状态。
    rng = random.Random(f"order:{order_id}")
    quantity, price = rng.randint(1, 3), rng.choice([39.9, 89.0, 129.0])
    # 同时选择订单状态与物流节点，避免未发货订单出现签收轨迹。
    order_status, logistics_status, node = rng.choice([
        ("已发货", "运输中", "包裹已到达杭州转运中心"),
        ("已发货", "派送中", "包裹由目的地配送站安排派送"),
        ("已签收", "已签收", "包裹已在收货地址签收"),
        ("待发货", "待发货", "订单尚未发货，暂无物流轨迹"),
    ])
    # 未发货时不虚构承运方或物流单号；新单号使用业务格式，数据来源由工具 tags 留痕。
    tracking_no = f"SF{rng.randint(100000000000, 999999999999)}" if order_status != "待发货" else None
    # 固定演示时间基准；保留原抽样顺序，再由节点时间推导创建时间。
    updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=rng.randrange(90 * 24 * 60))
    order = {"order_id": order_id, "product": "猫粮", "quantity": quantity,
             "unit_price": price, "total_amount": round(quantity * price, 2),
             "currency": "CNY", "status": order_status, "tracking_no": tracking_no,
             # 待发货时省市表示计划配送起终点，不代表已发生发货。
             "shipping_origin": "浙江省杭州市", "shipping_destination": _ORDER_DESTINATIONS[order_id],
             "created_at": (updated_at - timedelta(days=2)).isoformat()}
    logistics = {"order_id": order_id, "carrier": "顺丰速运" if tracking_no else None,
                 "tracking_no": tracking_no, "status": logistics_status, "latest_node": node,
                 "updated_at": updated_at.isoformat()}
    return order, logistics


def order_snapshot(order_id: str) -> dict:
    """读取已通过归属校验的演示订单快照。

    参数:
        order_id: 有限演示数据中的订单号；调用方须先通过 owns_order 校验。
    返回值:
        dict: 独立的订单摘要，包含商品、金额、状态、创建时间、物流单号及配送起终点省市；
            待发货订单的配送省市表示计划发出地与目的地。
    异常:
        KeyError: 订单号不在演示数据中；不访问网络或数据库。
    """
    if order_id not in _ORDER_OWNERS:
        raise KeyError(order_id)
    return _mock_order_snapshot(order_id)[0]


def order_id_for_tracking_no(tracking_no: str) -> str | None:
    """将已发货订单的物流单号映射到演示订单号。

    参数:
        tracking_no: 已去除首尾空白的非空物流单号，最多 64 字符；兼容历史 MOCK 前缀。
    返回值:
        str | None: 匹配的演示订单号；未知单号或未发货订单没有映射。
    异常:
        无主动抛出的业务异常；不访问网络或数据库。
    """
    # 历史会话保留的旧单号仅转换前缀，随后仍查有限订单映射并由调用方校验归属。
    if tracking_no.startswith("MOCK"):
        tracking_no = f"SF{tracking_no[4:]}"
    for order_id in _ORDER_OWNERS:
        order, _ = _mock_order_snapshot(order_id)
        # None 仅表示未发货，不是可以访问物流的单号。
        if order["tracking_no"] is not None and order["tracking_no"] == tracking_no:
            return order_id
    return None


def logistics_snapshot(tracking_no: str) -> dict:
    """读取已通过订单归属校验的物流快照。

    参数:
        tracking_no: 有限演示数据中的物流单号；调用方须先校验对应订单归属。
    返回值:
        dict: 独立的物流详情，包含订单号、承运方、物流单号和最新节点。
    异常:
        KeyError: 物流单号不在演示数据中；不访问网络或数据库。
    """
    order_id = order_id_for_tracking_no(tracking_no)
    if order_id is None:
        raise KeyError(tracking_no)
    return _mock_order_snapshot(order_id)[1]
