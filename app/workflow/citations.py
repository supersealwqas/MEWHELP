"""正文引用的增量过滤：仅发布本轮编号，保留普通方括号、代码和 Markdown 链接。

这里只校验引用编号属于本轮来源，不判断引用内容是否支持正文中的具体结论。
"""

import re

from app.knowledge.answering import _INLINE_CITATION


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int | None:
    """查找完整括号片段，兼容嵌套、转义和链接标题。

    参数：text 待处理正文；start 左括号位置；opening、closing 成对字符。
    返回：闭合字符之后的位置；未闭合返回 None。异常：无。
    """
    depth, escaped, quote = 0, False, ""
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote:
            if char == quote:
                quote = ""
        elif opening == "(" and char in "\"'" and text[index - 1].isspace():
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


class CitationStreamFilter:
    """跨模型分片维护括号和代码边界；仅暂存尚未能判定的标记，不缓存整份回答。"""

    def __init__(self, numbers: set[int], tool_names: set[str]) -> None:
        """建立本轮独占过滤器。参数：numbers 当前证据编号、tool_names 已知内部工具名；返回：None；异常：无。"""
        self.numbers, self.tool_names = frozenset(str(number) for number in numbers), frozenset(tool_names)
        self._pending, self._line, self._code = "", "", ""
        self._fenced = False

    def _consume(self, size: int, replacement: str | None = None) -> str:
        """消费已判定片段并保留原始行位置。参数：size 字符数、replacement 可选替代文本；返回：可发布正文；异常：无。"""
        raw, self._pending = self._pending[:size], self._pending[size:]
        self._line = raw.rsplit("\n", 1)[-1] if "\n" in raw else self._line + raw
        return raw if replacement is None else replacement

    def _is_marker(self, text: str) -> bool:
        """识别本项目角标语法。参数：text 完整方括号片段；返回：是否编号或已知工具标记；异常：无。"""
        return bool(_INLINE_CITATION.fullmatch(text)) or text[1:-1].strip() in self.tool_names

    def _filter_marker(self, text: str) -> str:
        """保留合法编号并清理伪引用。参数：text 完整方括号片段；返回：过滤后的文本；异常：无。"""
        match = _INLINE_CITATION.fullmatch(text)
        if match:
            parts = re.split(r"\s*[,，]\s*", match[1])
            # 字符串比较避免超长伪编号触发整数转换限制；有效编号仅来自可信证据。
            kept = [part for part in parts if part in self.numbers]
            return text if len(kept) == len(parts) else f"[{', '.join(kept)}]" if kept else ""
        return "" if text[1:-1].strip() in self.tool_names else text

    def feed(self, text: str, *, final: bool = False) -> str:
        """过滤新增模型正文，普通文字及时返回。

        参数：text 任意大小的增量；final 仅在上游正常结束时为真，释放未闭合普通文本。
        返回：本次可安全发送并累计保存的正文；异常：无，不执行模型或业务操作。
        """
        self._pending += text
        output = []
        while self._pending:
            char = self._pending[0]
            if not self._code and self._line.startswith(("    ", "\t")):
                # 缩进代码的整行保持原样，不能删除数组下标等普通语法。
                end = self._pending.find("\n")
                output.append(self._consume(end + 1 if end >= 0 else len(self._pending)))
                continue
            if self._code and char != self._code[0]:
                end = self._pending.find(self._code[0])
                output.append(self._consume(end if end >= 0 else len(self._pending)))
                continue
            if char in "`~":
                size = len(self._pending) - len(self._pending.lstrip(char))
                if size == len(self._pending) and not final:
                    break  # 下一个分片仍可能延长反引号或围栏，暂不改变代码状态。
                marker = char * size
                at_line_start = re.fullmatch(r" {0,3}", self._line) is not None
                if self._code:
                    closes = marker == self._code and not self._fenced
                    if self._fenced and at_line_start and size >= len(self._code):
                        line_end = self._pending.find("\n", size)
                        if line_end < 0 and not final:
                            break
                        closes = not self._pending[size:line_end if line_end >= 0 else None].strip()
                    if closes:
                        self._code, self._fenced = "", False
                elif char == "`" or (size >= 3 and at_line_start):
                    self._code, self._fenced = marker, size >= 3 and at_line_start
                output.append(self._consume(size))
                continue
            if char == "\\":
                if len(self._pending) == 1 and not final:
                    break
                output.append(self._consume(2))
                continue
            if char == "[":
                end = _balanced_end(self._pending, 0, "[", "]")
                if end is None or (end == len(self._pending) and not final):
                    if final:
                        output.append(self._consume(len(self._pending)))
                    break
                label = self._pending[:end]
                following = self._pending[end:end + 1]
                if following == "(":
                    link_end = _balanced_end(self._pending, end, "(", ")")
                    if link_end is None and not final:
                        break
                    # 整个链接保留，地址或标题内部的方括号不是引用。
                    output.append(self._consume(link_end or len(self._pending)))
                    continue
                if following == "[":
                    reference_end = _balanced_end(self._pending, end, "[", "]")
                    if reference_end is None and not final:
                        break
                    if reference_end and not self._is_marker(self._pending[end:reference_end]):
                        output.append(self._consume(reference_end))
                        continue
                    # 连续 [1][2] 属于两个角标，不能误认为带文字标签的参考链接。
                if following == ":" and not self._line.strip():
                    line_end = self._pending.find("\n", end)
                    if line_end < 0 and not final:
                        break
                    output.append(self._consume(line_end + 1 if line_end >= 0 else len(self._pending)))
                    continue
                output.append(self._consume(end, self._filter_marker(label)))
                continue
            next_marker = re.search(r"[\\\[`~]", self._pending)
            output.append(self._consume(next_marker.start() if next_marker else len(self._pending)))
        return "".join(output)
