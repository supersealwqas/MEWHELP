-- CH06状态迁移参考；推荐使用幂等预览/应用入口 app.db.migrate_ch06。
-- 保留所有历史行，不清库。退款独立表在后续迁移中补充。
ALTER TABLE messages DROP CHECK ck_turn_status,
  ADD CONSTRAINT ck_turn_status
  CHECK (turn_status IN ('running','awaiting_input','completed','failed','cancelled'));
