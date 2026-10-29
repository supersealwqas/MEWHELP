# CH08 浏览器验收记录

状态：**2026-10-29 八项真实页面验收全部通过**。由本人通过 Codex 内置浏览器操作，逐项看图，并由仍在运行的原验收实例核对真实 MySQL、原动作、请求、结果与审计。总验收九项通过，退出码 0；没有使用子智能体。

最终证据目录为 `.superpowers/sdd/2026-10-28-ch08-tools/live-final-4/`，数据库为 `mewhelp_test`。运行 ID 为 `f4813e4f34434f349050c393998227ae`，客服 PID 为 `38024`，注册表/图实例 ID 为 `2693818169584` / `2695315991024`。完整会话、原轮及工具调用 ID 见 [浏览器原始记录](../.superpowers/sdd/2026-10-28-ch08-tools/live-final-4/browser-evidence.json)，逐项真实库核对见 [浏览器验收结果](../.superpowers/sdd/2026-10-28-ch08-tools/live-final-4/browser_actions.json)。

| 操作与证据前缀 | 实际结果 | 原 action_id |
|---|---|---|
| 缺描述后补充 `missing_description` | 先追问，补充后保留原文及句号；工单 0→0 | `be152925-dc27-5ed9-b28a-98701f030096` |
| 确认 `confirm` | 工单 0→1，唯一成功结果及审计，页面显示实际票号 | `bf93563d-ce73-5f83-a7e7-f196aa9fe8e2` |
| 取消 `cancel` | 工单 0→0，唯一 USER_CANCELLED 结果及权限拒绝审计 | `918481c8-f39d-5c76-a736-6aa7ea2a9665` |
| 原生双击 `double_click` | 双击只发出一次确认请求；工单 0→1，唯一成功审计 | `b7d7fc67-8caf-5c07-9eba-2ef984fd0c7d` |
| 刷新 `refresh_preview` | 同动作、同描述回读；工单 0→0，未新增用户消息 | `62d7402e-482f-5cab-ac95-2b9e610288d3` |
| 断流恢复 `disconnect_resume` | 决定已保存时停止，核实后出现“继续处理”；两次请求沿用原动作，工单 0→1 | `6ae570ee-757e-57e1-9de1-78dc15b226c3` |
| 会话切换 `session_switch` | A 已写入后停止并切 B；迟到回执不出现在 B，A 工单 0→1 | `1439f162-5259-5ce6-a840-f0c454bdf530` |
| 窄屏 `narrow_screen` | 390×900 视口中描述正常换行；滚动后完整卡片与双按钮可见，工单 0→0 | `27bd8a3a-d1b3-5b35-959a-65e3194479d1` |

确认场景的实际工单号是 `Tfb27152a099aea8f54f08a8e447738a`。会话隔离检查使用 A 的实际工单号 `T45a14668e5e0737a0840622b7d9157e` 比对 B 的完整可见消息文本，没有只靠模糊票号正则判断。

## 原件与故障边界

- `browser-http/` 保存测试 ASGI 层实际接收和发送的请求、响应、状态及起止时间；没有修改请求原文。普通聊天省略 `user_id` 时按生产 `ChatRequest` 的默认值核验，显式其他用户仍被拒绝。
- `browser-snapshots/` 由原运行实例回读数据库产生，并以其内存中保存的哈希索引核验；前后快照严格按时间排序。浏览器没有自行构造数据库结果。
- `browser-artifacts/` 保留内置浏览器直接返回的 JPEG 原件和 UI 状态。另导出 PNG 供验收器读取，未裁剪、缩放或编辑；解码 RGB 像素逐字节核对相等。原件与导出哈希见 `screenshot-exports.json`。
- 断流测试只在“首次决定保存后、写处理器开始前”暂停，再点击页面的“停止回复”。迟到测试只让指定会话的原请求继续接收；生产模型、工具、工单事务和前端会话隔离逻辑仍实际执行。注入配置/时刻保存在同轮目录。
- 窄屏消息区需要滚动；通过 Tab 焦点滚动使完整描述及双按钮进入视口，未提交工单。证明截图为 `narrow_screen-visible.jpg`。验收结束已恢复正常视口，服务进程由验收器关闭。

## 回归与结论范围

全开关 Python 回归 **2252 passed、0 failed、0 skipped**；既有 Node 回归 **128 passed**。随后验收器默认用户问题修复取得真实反例，验收器/模型评估器 **80 passed**。原始日志分别为 `final-all3.log`、`final-node4.log` 和 `final-browser-identity-green2.log`。

早期失败与调试记录不作为最终通过证据；本报告仅引用 `live-final-4`。其中 live-final-3 终结于验收器默认用户误判，后续旧辅助函数还误覆盖过该失效目录的一张追问截图，已在开发记录注明。最终八项均在新的运行实例重新操作并保存独立证据。没有向生产库发布或迁移。

![内置浏览器窄屏完整工单预览](../.superpowers/sdd/2026-10-28-ch08-tools/live-final-4/browser-artifacts/narrow_screen-visible.jpg)
