-- CH07 第二步：原文层边界和分段摘要，两份ch07 DDL都要应用。
-- id <= summary_upto 已摘要；summary_upto < id <= layer1_from 为半压层；其后为原文层。
-- 段落只追加，旧段不回炉；原始消息保留不搬动。
SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN layer1_from_msg_id BIGINT UNSIGNED NULL
    COMMENT '层1原文起点;此id之后原样,之前渲染成半压形态'
    AFTER summary_upto_msg_id;

CREATE TABLE IF NOT EXISTS conversation_summaries (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  conversation_id BIGINT UNSIGNED NOT NULL,
  seq INT NOT NULL COMMENT '第几段,从1开始',
  from_msg_id BIGINT UNSIGNED NOT NULL COMMENT '这段覆盖的消息区间,闭区间',
  upto_msg_id BIGINT UNSIGNED NOT NULL,
  content TEXT NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_conv_seq (conversation_id, seq),
  KEY idx_conv_upto (conversation_id, upto_msg_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='分段摘要,一段一行只追加';
