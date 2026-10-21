"""知识关系纯计算：稳定ref组件、来源关联与持久隔离组约束传播。"""

from __future__ import annotations

from collections.abc import Iterable

from app.knowledge.schemas import ChunkSnapshot, RelationDecision, StagedQA


def relation_components(
    staged: list[StagedQA], chunks: list[ChunkSnapshot],
    decisions: Iterable[RelationDecision],
) -> list[list[str]]:
    """先传播全部关系与历史来源约束，再返回完整组件。

    参数：staged为全部暂存审计；chunks为QA块；decisions仅QA之间的关系。
    返回：稳定ref排序的组件列表；same/conflict/uncertain连接，互补与无关不连接。
    异常：KeyError，决定引用不存在的QA节点时抛出。
    """
    refs = [f"staged:{row.id}" for row in staged] + [f"chunk:{row.id}" for row in chunks]
    parent = {ref: ref for ref in refs}

    def find(ref: str) -> str:
        """寻找并压缩根。参数：ref已知节点；返回：根ref；异常：KeyError。"""
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    def union(left: str, right: str) -> None:
        """合并节点。参数：left/right已知ref；返回：None；异常：KeyError。"""
        first, second = find(left), find(right)
        if first != second:
            parent[second] = first

    chunk_ids = {row.id for row in chunks}
    prior_groups: dict[str, str] = {}
    for row in staged:
        ref = f"staged:{row.id}"
        if row.knowledge_id in chunk_ids:
            union(ref, f"chunk:{row.knowledge_id}")
        if row.conflict_group:
            previous = prior_groups.setdefault(row.conflict_group, ref)
            union(previous, ref)
    for decision in decisions:
        if decision.relation in ("same", "conflict", "uncertain"):
            union(decision.left_ref, decision.right_ref)
    components: dict[str, list[str]] = {}
    for ref in refs:
        components.setdefault(find(ref), []).append(ref)
    return sorted((sorted(members) for members in components.values()), key=lambda members: members[0])
