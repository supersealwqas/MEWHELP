-- CH06状态迁移参考；推荐使用幂等预览/应用入口 app.db.migrate_ch06。
-- 保留所有历史行，不清库。退款独立表在后续迁移中补充。
ALTER TABLE messages DROP CHECK ck_turn_status,
  ADD CONSTRAINT ck_turn_status
  CHECK (turn_status IN ('running','awaiting_input','completed','failed','cancelled'));

CREATE TABLE IF NOT EXISTS refund_applications (
  refund_no VARCHAR(34) COLLATE utf8mb4_bin PRIMARY KEY,
  action_id VARCHAR(36) COLLATE utf8mb4_bin NOT NULL,
  conversation_id BIGINT UNSIGNED NOT NULL,
  source_turn_id VARCHAR(36) COLLATE utf8mb4_bin NOT NULL,
  user_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
  order_id VARCHAR(64) COLLATE utf8mb4_bin NOT NULL,
  reason_code VARCHAR(32) NOT NULL,
  original_question TEXT NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT '待处理',
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT uq_refund_action UNIQUE (action_id),
  CONSTRAINT fk_refunds_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id),
  CONSTRAINT ck_refund_pending CHECK (status = '待处理')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='本地退款申请，不执行真实退款';
