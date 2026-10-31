"""按权威17类计算多标签指标、验证集阈值和排他兜底解码。"""

import math

from app.topics.taxonomy import label_names, validate_labels


def _probabilities(values, taxonomy: dict) -> list[float]:
    """校验一行模型概率。参数：values17维概率、taxonomy主题表；返回：浮点列表；异常：ValueError。"""
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("模型概率格式非法") from error
    if len(result) != len(label_names(taxonomy)) or len(result) != 17:
        raise ValueError("模型概率必须为17维")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in result):
        raise ValueError("模型概率必须为有限的0至1数值")
    return result


def decode_labels(probabilities, threshold: float, taxonomy: dict) -> list[str]:
    """解码多标签并保持未确定为空。参数：17维概率、0至1阈值、主题表；返回：标签数组；异常：ValueError。"""
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("阈值非法")
    probabilities = _probabilities(probabilities, taxonomy)
    names = label_names(taxonomy)
    labels = [name for name, value in zip(names[:-1], probabilities[:-1]) if value >= threshold]
    # 其他是排他业务类；全不通过表示未确定，绝不强制取最大概率。
    return labels or ([names[-1]] if probabilities[-1] >= threshold else [])


def compute_metrics(expected: list[list[str]], predicted: list[list[str]], taxonomy: dict) -> dict:
    """计算独立分母的多标签指标。参数：等长非空金标/预测数组、主题表；返回：指标字典；异常：ValueError。"""
    if len(expected) != len(predicted) or not expected:
        raise ValueError("指标必须使用等长非空样本")
    for labels in expected + predicted:
        if labels != []:
            validate_labels(labels, taxonomy)
    per_label, unsupported, supported_f1 = {}, [], []
    total_tp = total_fp = total_fn = 0
    for name in label_names(taxonomy):
        tp = sum(name in actual and name in guess for actual, guess in zip(expected, predicted))
        fp = sum(name not in actual and name in guess for actual, guess in zip(expected, predicted))
        fn = sum(name in actual and name not in guess for actual, guess in zip(expected, predicted))
        tn = len(expected) - tp - fp - fn
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if support else None
        per_label[name] = dict(precision=precision, recall=recall, f1=f1, support=support,
                               confusion_matrix=[[tn, fp], [fn, tp]])
        if support:
            supported_f1.append(f1)
        else:
            unsupported.append(name)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    denominator = 2 * total_tp + total_fp + total_fn
    return dict(samples=len(expected), per_label=per_label, unsupported=unsupported,
                micro_f1=2 * total_tp / denominator if denominator else 0.0,
                macro_f1=sum(supported_f1) / len(supported_f1) if supported_f1 else None,
                exact_match=sum(set(a) == set(b) for a, b in zip(expected, predicted)) / len(expected))


def select_threshold(probabilities, targets, taxonomy: dict) -> float:
    """仅用调用方传入的验证集选择全局阈值。参数：概率矩阵、0/1金标矩阵、主题表；返回：最优阈值；异常：ValueError。"""
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("验证集必须等长非空")
    probabilities = [_probabilities(row, taxonomy) for row in probabilities]
    targets = [_probabilities(row, taxonomy) for row in targets]
    if any(value not in (0.0, 1.0) for row in targets for value in row):
        raise ValueError("金标必须为0/1多热编码")
    names = label_names(taxonomy)
    expected = [[name for name, value in zip(names, row) if value] for row in targets]
    best_threshold, best_score = .5, None
    # 用整数生成候选，避免浮点累加改变边界概率与稳定平局规则。
    for candidate in range(10, 91, 5):
        threshold = candidate / 100
        report = compute_metrics(expected, [decode_labels(row, threshold, taxonomy) for row in probabilities], taxonomy)
        score = (report["macro_f1"] if report["macro_f1"] is not None else -1,
                 report["exact_match"], -abs(threshold - .5))
        if best_score is None or score > best_score:
            best_threshold, best_score = threshold, score
    return best_threshold
