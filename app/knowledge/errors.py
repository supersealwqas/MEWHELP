"""知识库存储层安全异常。

所有异常只携带安全说明，不包含 SQL、凭据或完整聊天内容；上层据此区分
并发冲突、锁丢失、快照变化与依赖不可用，不能把它们误报成正常空结果。
"""


class KnowledgeStoreUnavailable(RuntimeError):
    """MySQL 存储或写锁依赖不可用。"""


class ConcurrentKnowledgeWrite(KnowledgeStoreUnavailable):
    """另一个写入者持有知识库写锁或批次来源冲突。"""


class LeaseLost(KnowledgeStoreUnavailable):
    """当前物理连接已不再持有知识库写锁。"""


class SnapshotChanged(KnowledgeStoreUnavailable):
    """发布依据的数据库快照已经变化。"""
