-- =============================================================
-- ch07 · 会话上下文管理 · 建表 DDL（第一步，共两步）
-- 本文件增加摘要投影与覆盖边界；随后必须执行 ch07-layers.sql。
-- 摘要投影只拼接已提交的完整段，不将旧摘要反复压缩。
-- 已有部分字段时使用 app.db.migrate_ch07 增量检查，勿盲目重复 ALTER。
-- =============================================================

-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN summary             TEXT            NULL COMMENT '最近几段梗概拼成的投影,拼装时跟证据一起挂在用户那句之后' AFTER status,
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL COMMENT '摘要已覆盖到哪条消息,滑窗从其后接原文' AFTER summary;
