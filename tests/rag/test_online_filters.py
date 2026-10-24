"""线上预过滤和短条件保护：从真实 API/工具贯通至索引边界。"""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from app.knowledge.retrieval import HybridKnowledgeSearch
from app.knowledge.search import KnowledgeSearchUnavailable
from app.knowledge.text import content_fingerprint
from app.repositories.support import SupportRepository
from app.schemas.tools import ToolContext
from app.tools.registry import build_registry
from tests.chat_fakes import make_chat_service
from tests.knowledge.fakes import FakeEncoder, MemoryKnowledgeStore, make_settings
from tests.rag.test_retrieval import HybridBoundary, RerankBoundary, _chunk, _search


class FilterIndex(HybridBoundary):
    """模拟支持主键预过滤的外部索引，支持故障注入。"""

    def __init__(self, chunks, *, ignore_filter=False, empty=False):
        """参数：chunks 权威块、ignore_filter/empty 故障开关；返回：None；异常：无。"""
        super().__init__(chunks)
        self.filters = []
        self.ignore_filter, self.empty = ignore_filter, empty

    async def hybrid_search(self, vector, text, *, dense_limit, sparse_limit, limit, filter_ids=None):
        """参数：召回输入与白名单；返回：筛选命中；异常：无。"""
        self.filters.append(filter_ids)
        hits = await super().hybrid_search(vector, text, dense_limit=dense_limit,
                                            sparse_limit=sparse_limit, limit=limit)
        if self.empty:
            return []
        return [hit for hit in hits if self.ignore_filter or filter_ids is None or hit.id in filter_ids]


def filtered_search(**index_options):
    """组装两品类真实检索。参数：索引故障开关；返回：服务/索引；异常：校验错误。"""
    chunks = []
    for identifier, category in [(1, "饮水机"), (2, "猫砂盆")]:
        row = _chunk(identifier)
        text = row.text.model_copy(update={"category": category})
        chunks.append(row.model_copy(update={"text": text, "content_hash": content_fingerprint(text)}))
    index = FilterIndex(chunks, **index_options)
    search = HybridKnowledgeSearch(MemoryKnowledgeStore(snapshots=chunks), FakeEncoder(),
                                   index, RerankBoundary(), make_settings())
    return search, index


def test_search_api_applies_category_before_both_recalls():
    """删掉线上参数贯通会返回别的品类。参数：无；返回：None；异常：断言失败。"""
    from app.main import create_app

    search, index = filtered_search()
    service, _, _, _ = make_chat_service()
    service.executor.support = SupportRepository(None, search)

    @asynccontextmanager
    async def runtime():
        """提供隔离服务。参数：无；返回：服务；异常：无。"""
        yield service

    with TestClient(create_app(runtime_factory=runtime)) as client:
        response = client.get("/kb/search", params={"query": "如何保养", "category": "饮水机"})
        assert response.status_code == 200
        assert [row["id"] for row in response.json()["hits"]] == [1]
        assert client.get("/kb/search", params={"query": "保养", "category": ""}).status_code == 422
    assert index.filters == [[1]]
    assert index.calls[0][1:] == (50, 50, 50)


async def test_empty_filter_scope_is_successful_empty():
    """空范围禁止放宽为全库查询。参数：无；返回：None；异常：断言失败。"""
    search, index = filtered_search()
    assert await search.query_faq("如何保养", filters={"category": "不存在"}) == []
    assert index.calls == []


@pytest.mark.parametrize("options", [{"ignore_filter": True}, {"empty": True}])
async def test_index_returning_outside_filter_fails_closed(options):
    """越界或范围内坏索引不能伪装知识不足。参数：故障开关；返回：None；异常：断言失败。"""
    search, _ = filtered_search(**options)
    with pytest.raises(KnowledgeSearchUnavailable):
        await search.query_faq("如何保养", filters={"category": "饮水机"})


@pytest.mark.parametrize("name,args", [("query_faq", {"keyword": "如何保养"}),
                                     ("query_refund_policy", {"query": "如何退款"})])
async def test_tool_filters_do_not_expose_identity(name, args):
    """真实工具不可丢筛选或公开身份配置。参数：工具/参数；返回：None；异常：断言失败。"""
    search, index = filtered_search()
    tool = build_registry().get(name)
    schema = tool.tool_call_schema.model_json_schema()["properties"]
    assert not {"user_id", "context", "config"} & schema.keys()
    context = ToolContext(session_id="filters", user_id="demo-user", turn_id="filters", user_message="如何退款")
    result = await tool.ainvoke({**args, "category": "饮水机", "content_type": "policy"},
                               config={"configurable": {"support_repository": SupportRepository(None, search),
                                                         "context": context}})
    assert [hit["id"] for hit in result["hits"]] == [1]
    assert all(ids == [1] for ids in index.filters)


@pytest.mark.parametrize("query,condition", [("新疆呢，配送收费怎么算，会员免邮有什么限制", "新疆呢"),
                                           ("不退款，申请条件是什么，寄回运费谁承担", "不退款")])
async def test_short_region_and_negation_survive_clause_split(query, condition):
    """短片段不能在拆句时丢失。参数：原问/短约束；返回：None；异常：断言失败。"""
    search, encoder, index, reranker = _search([_chunk(1)])
    await search.retrieve(query)
    assert len(index.calls) == 2
    assert all(condition in text for text in encoder.inputs)
    assert all(condition in call[0] for call in index.calls)
    assert all(condition in call[0] for call in reranker.calls)
