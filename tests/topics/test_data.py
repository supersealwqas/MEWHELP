"""主题数据的隐私、分组隔离与清单校验测试。"""

import json
from copy import deepcopy

import pytest

from app.topics.taxonomy import load_taxonomy, validate_labels
from app.topics.data import clean_question, split_examples, augment_training, validate_dataset, write_dataset


def sample(identifier, text, group, source="synthetic", parent=None):
    """构造无隐私样本。参数：标识、正文、家族、来源、父样本；返回：字典；异常：无。"""
    return dict(id=identifier, text=text, labels=["尺码"], group_id=group, source=source,
                pool_ids=[], parent_id=parent, annotation_status="model_prelabel")


def test_labels_are_canonical_and_other_is_exclusive():
    """拒绝未知、重复、空和混用标签。参数：无；返回：无；异常：断言失败。"""
    taxonomy = load_taxonomy()
    assert validate_labels(["尺码", "退换货"], taxonomy) == ["退换货", "尺码"]
    for labels in ([], ["尺码", "尺码"], ["未知"], ["其他", "尺码"]):
        with pytest.raises(ValueError):
            validate_labels(labels, taxonomy)


def test_group_and_paraphrase_never_cross_splits():
    """防止跨来源文本、父子和连通家族泄漏。参数：无；返回：无；异常：断言失败。"""
    rows = [sample(str(i), f"请问鞋子的{i}号有多长", f"g{i}") for i in range(60)]
    rows += [sample("a", rows[0]["text"], "ga", "application_demo"),
             sample("b", "尺码请帮我确认", "gb", parent="a"),
             sample("c", "这双鞋的尺寸", "g1", parent="b")]
    parts = split_examples(rows, load_taxonomy())
    assert parts == split_examples(deepcopy(rows), load_taxonomy())
    owners = {r["id"]: split for split, group in parts.items() for r in group}
    assert len({owners[i] for i in ("0", "1", "a", "b", "c")}) == 1
    assert all(parts.values())


def test_only_train_is_augmented():
    """增强保留否定数字，留出样本保持原样。参数：无；返回：无；异常：断言失败。"""
    parts = split_examples([sample(str(i), f"请问XL不退，{i}号尺寸多少", str(i)) for i in range(40)], load_taxonomy())
    before = deepcopy(parts)
    parts["train"] = augment_training(parts["train"], load_taxonomy())
    assert parts["validation"] == before["validation"] and parts["test"] == before["test"]
    derived = [r for r in parts["train"] if r["source"] == "augmentation"]
    assert derived and all("XL不退" in r["text"] and r["parent_id"] is not None for r in derived)


def test_redaction_keeps_size_and_negation():
    """脱敏明确个人信息而保留业务数字。参数：无；返回：无；异常：断言失败。"""
    text = clean_question("姓名：张三，手机号13812345678，邮箱a@example.com；XL不退，型号MH-W99，3天，订单号1234567890123456789")
    assert "13812345678" not in text and "a@example.com" not in text and "张三" not in text
    assert "1234567890123456789" not in text
    assert "XL不退" in text and "MH-W99" in text and "3天" in text


@pytest.mark.parametrize("question", [
    "怎么联系人工客服处理账号申诉", "怎么联系人工客服帮我查会员积分？",
    "我用积分兑换的商品，下单后发现收件人名字写错了能改吗？",
    "买的电池发票抬头能写个人姓名加身份证号吗？",
    "订单号能改一下收件人电话吗？还有发票抬头刚才忘记选单位了。",
    "不取消订单，能不能直接更改收件人姓名", "联系人是谁？", "姓名是什么？",
    "地址是否可以修改？", "订单 2026-11-01 能到吗？型号M01012345678，XL不退",
    "收件人能改吗？", "收件人不变，只修改地址", "收件人没变", "联系人未定",
    "收件人填错了，怎么修改？", "地址是必填项吗？", "姓名是 什么意思？", "地址是 必填项吗？",
    "地址是否正确？", "姓名是必填的吗？", "收件人可修改吗？",
    "地址是否有效？", "姓名是必填的么？", "收件人该换吗？", "收件人换掉了", "收件人改过了",
])
def test_explicit_field_questions_keep_business_meaning(question):
    """查询字段或人工客服不等于提供姓名，日期型号也不能被误删。参数：question无个人资料的业务问题；返回：None；异常：断言失败。"""
    assert clean_question(question) == question


def test_manifest_detects_tampering_and_split_leakage(tmp_path):
    """制品摘要和家族隔离不能被跳过。参数：tmp_path临时目录；返回：无；异常：断言失败。"""
    output = tmp_path / "dataset"
    rows = [sample(str(i), f"请问{i}码鞋的尺寸", str(i)) for i in range(40)]
    write_dataset(output, split_examples(rows, load_taxonomy()), load_taxonomy(), seed=42)
    assert validate_dataset(output)["total"] == 40
    with (output / "test.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(rows[0], ensure_ascii=False) + "\n")
    with pytest.raises(ValueError):
        validate_dataset(output)


def test_prepare_uses_saved_success_without_repaying(tmp_path):
    """失败恢复复用已成功批次，且拒绝覆盖制品。参数：临时目录；返回：无；异常：断言失败。"""
    from app.topics.cli import prepare_dataset
    import asyncio

    class Labeller:
        """替代付费模型，仅验证恢复流水线。"""

        def __init__(self):
            """初始化失败开关。参数：无；返回：无；异常：无。"""
            self.calls = 0

        async def label(self, texts):
            """返回规范预标。参数：texts；返回：预标列表；异常：无。"""
            return [dict(index=i, text=t, original_text=t, labels=["尺码"], needs_review=False) for i, t in enumerate(texts)]

        async def generate(self, labels, count, seed_tag):
            """模拟一次临时失败。参数：标签、数量、批次；返回：模拟；异常：RuntimeError。"""
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("temporary")
            return [dict(index=i, text=f"请问场景{seed_tag}问题{i}", labels=labels, group_id=f"g:{seed_tag}:{i}", needs_review=False) for i in range(count)]

    source = tmp_path / "source.json"
    source.write_text(json.dumps([dict(question_id="1", text="尺寸多少", source="application_demo", group_id="legacy-pool:1", history_uncertain=True, provenance={})]), encoding="utf-8")
    output = tmp_path / "prepared"
    model = Labeller()
    with pytest.raises(RuntimeError):
        asyncio.run(prepare_dataset(source, output, per_label=1, multi=0, labeller=model))
    assert not output.exists()
    asyncio.run(prepare_dataset(source, output, per_label=1, multi=0, labeller=model))
    assert model.calls == 18
    assert validate_dataset(output)["human_review_status"] == "pending"
    with pytest.raises(FileExistsError):
        asyncio.run(prepare_dataset(source, output, per_label=1, multi=0, labeller=model))


def test_multilabel_and_source_strata_have_holdout_coverage():
    """足量独立家族保持各标签和来源留出覆盖。参数：无；返回：无；异常：断言失败。"""
    from app.topics.taxonomy import label_names
    taxonomy = load_taxonomy()
    rows = []
    for label in label_names(taxonomy):
        for i in range(40):
            row = sample(f"{label}:{i}", f"场景{label}{i}的独立提问", f"{label}:{i}", "application_demo" if i < 10 else "synthetic")
            row["labels"] = [label]
            rows.append(row)
    parts = split_examples(rows, taxonomy)
    for split in parts.values():
        assert {label for row in split for label in row["labels"]} == set(label_names(taxonomy))
        assert {row["source"] for row in split} == {"application_demo", "synthetic"}
    assert 500 <= len(parts["train"]) <= 585


def test_manifest_counts_cannot_hide_label_or_source_support(tmp_path):
    """清单计数必须和实际样本一致。参数：临时目录；返回：无；异常：断言失败。"""
    output = tmp_path / "support"
    write_dataset(output, split_examples([sample(str(i), f"第{i}码尺寸", str(i)) for i in range(40)], load_taxonomy()), load_taxonomy(), seed=42)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["counts"]["train"]["labels"] = {}
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        validate_dataset(output)


def test_augmentation_skips_holdout_collision_and_keeps_holdout_unchanged(tmp_path):
    """自然留出问法与训练增强碰撞时跳过派生。参数：临时目录；返回：无；异常：断言失败。"""
    rows = [sample(str(i), f"尺寸问题{i}", str(i)) for i in range(40)]
    rows[0]["text"] = "请问XL尺寸多少"
    rows[16]["text"] = "想问一下XL尺寸多少"
    parts = split_examples(rows, load_taxonomy(), seed=42)
    before = deepcopy(parts)
    assert "0" in {row["id"] for row in parts["train"]}
    assert "16" in {row["id"] for row in parts["validation"]}
    reserved = {row["text"] for split in ("validation", "test") for row in parts[split]}
    parts["train"] = augment_training(parts["train"], load_taxonomy(), reserved_texts=reserved)
    assert parts["validation"] == before["validation"] and parts["test"] == before["test"]
    assert "想问一下XL尺寸多少" not in {row["text"] for row in parts["train"]}
    write_dataset(tmp_path / "no-collision", parts, load_taxonomy(), seed=42)


def test_failed_final_validation_preserves_diagnostics_and_allows_retry(tmp_path, monkeypatch):
    """末尾校验失败不占最终目录，原命令可恢复。参数：目录、替换器；返回：无；异常：断言失败。"""
    from app.topics import data as topic_data
    output = tmp_path / "recoverable"
    parts = split_examples([sample(str(i), f"尺寸问题{i}", str(i)) for i in range(40)], load_taxonomy())
    real_validate = topic_data.validate_dataset

    def reject_once(directory):
        """模拟实际制品检查失败。参数：directory临时制品；返回：无；异常：ValueError。"""
        raise ValueError("diagnostic failure")

    monkeypatch.setattr(topic_data, "validate_dataset", reject_once)
    with pytest.raises(ValueError):
        write_dataset(output, parts, load_taxonomy(), seed=42)
    assert not output.exists()
    diagnostics = list(tmp_path.glob(".recoverable.building-*/failure.json"))
    assert diagnostics and json.loads(diagnostics[0].read_text(encoding="utf-8"))["error_type"] == "ValueError"
    monkeypatch.setattr(topic_data, "validate_dataset", real_validate)
    write_dataset(output, parts, load_taxonomy(), seed=42)
    assert validate_dataset(output)["total"] == 40


def test_prepare_recovers_after_final_validation_without_new_model_calls(tmp_path, monkeypatch):
    """制品校验失败后CLI重跑复用全部付费批次。参数：目录、替换器；返回：无；异常：断言失败。"""
    import asyncio
    from app.topics import data as topic_data
    from app.topics.cli import prepare_dataset

    class LabelOnly:
        """替代付费网络，统计已成功预标批次。"""

        def __init__(self):
            """初始化计数。参数：无；返回：无；异常：无。"""
            self.calls = 0

        async def label(self, texts):
            """返回模型边界结果。参数：文本；返回：模型预标列表；异常：无。"""
            self.calls += 1
            return [dict(index=i, text=t, labels=["尺码"], needs_review=False) for i, t in enumerate(texts)]

    source = tmp_path / "pool.json"
    source.write_text(json.dumps([dict(question_id="1", text="请问XL尺寸多少", source="application_demo",
                                     group_id="legacy-pool:1", history_uncertain=True, provenance={})]), encoding="utf-8")
    output = tmp_path / "cli-recoverable"
    model = LabelOnly()
    real_validate = topic_data.validate_dataset

    def reject(directory):
        """模拟末尾检查失败。参数：临时目录；返回：无；异常：ValueError。"""
        raise ValueError("validation failure")

    monkeypatch.setattr(topic_data, "validate_dataset", reject)
    with pytest.raises(ValueError):
        asyncio.run(prepare_dataset(source, output, per_label=0, multi=0, labeller=model))
    assert not output.exists() and model.calls == 1
    monkeypatch.setattr(topic_data, "validate_dataset", real_validate)
    asyncio.run(prepare_dataset(source, output, per_label=0, multi=0, labeller=model))
    assert model.calls == 1 and validate_dataset(output)["human_review_status"] == "pending"
