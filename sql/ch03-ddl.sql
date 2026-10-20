-- =============================================================
-- ch03 · 知识库与 Dense 检索 · 建表 DDL
-- MySQL 保存原文、发布资格、抽取暂存与批次身份；Milvus 只保存 Dense 向量。
-- 与 app/models/knowledge.py 的独立 KnowledgeBase 对齐，不扩展现有 CH02 四表。
-- =============================================================

SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id                 BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '与 Milvus 对齐的正整数主键',
  source_key         VARCHAR(255) COLLATE utf8mb4_bin NOT NULL COMMENT '稳定来源键：doc:相对路径或 qa:暂存ID',
  ordinal            INT             NOT NULL                COMMENT '来源内从 0 开始的连续序号',
  category           VARCHAR(255)    NOT NULL                COMMENT '分类，进入向量',
  questions          TEXT            NOT NULL                COMMENT '问法，进入向量',
  answer             TEXT            NOT NULL                COMMENT '答案原文，进入向量',
  section_path       VARCHAR(512)    NULL                    COMMENT '完整章节路径，仅元数据',
  content_type       VARCHAR(32)     NULL                    COMMENT '内容类型，仅元数据',
  is_key_clause      TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款，仅元数据',
  content_hash       CHAR(64) COLLATE utf8mb4_bin NOT NULL   COMMENT '三格与格式规范指纹',
  embedding_revision VARCHAR(128)    NOT NULL                COMMENT '编码模型与推理配置标识',
  is_active          TINYINT(1)      NOT NULL DEFAULT 1      COMMENT '是否具备在线发布资格',
  prev_chunk_id      BIGINT UNSIGNED NULL                    COMMENT '同来源前一块',
  next_chunk_id      BIGINT UNSIGNED NULL                    COMMENT '同来源后一块',
  vector_id          VARCHAR(64)     NULL                    COMMENT 'Milvus 主键，成功后回填',
  vectorize_status   ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '向量同步状态',
  created_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uq_chunks_source_ordinal (source_key, ordinal),
  KEY idx_chunks_source_key (source_key),
  KEY idx_chunks_active_status (is_active, vectorize_status),
  KEY idx_chunks_ordinal (source_key, ordinal),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_batches (
  id                BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '批次主键',
  batch_no          VARCHAR(64) COLLATE utf8mb4_bin NOT NULL COMMENT '稳定批次号',
  source_ids        JSON            NOT NULL                COMMENT '本批最终回答消息 ID 列表',
  snapshot_upper_id BIGINT UNSIGNED NOT NULL                COMMENT '快照的最终回答消息上界',
  input_hash        CHAR(64) COLLATE utf8mb4_bin NOT NULL   COMMENT '批次输入指纹',
  prompt_version    VARCHAR(64)     NOT NULL                COMMENT '抽取 Prompt 规范版本',
  status            ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT 'pending/done',
  created_at        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at        DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uq_batches_batch_no (batch_no),
  KEY idx_batches_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽取批次';

CREATE TABLE qa_extraction_staging (
  id                 BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no           VARCHAR(64)     NOT NULL                COMMENT '来源抽取批次',
  source_ref         VARCHAR(255)    NULL                    COMMENT '来源会话或导出标识',
  source_ids         JSON            NOT NULL                COMMENT '来源最终回答消息 ID',
  source_fingerprint CHAR(64) COLLATE utf8mb4_bin NOT NULL   COMMENT '来源与问答规范化指纹',
  question           TEXT            NOT NULL                COMMENT '真实用户问法',
  answer             TEXT            NOT NULL                COMMENT '有原文支持的答案',
  category           VARCHAR(255)    NOT NULL                COMMENT '知识分类',
  status             ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '候选知识处置状态',
  conflict_group     VARCHAR(64)     NULL                    COMMENT '持久冲突组身份',
  evidence_refs      JSON            NOT NULL                COMMENT '处置证据引用',
  reason             VARCHAR(512)    NULL                    COMMENT '处置原因',
  knowledge_id       BIGINT UNSIGNED NULL                    COMMENT '关联的已发布知识',
  created_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at         DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  UNIQUE KEY uq_staging_source_fingerprint (source_fingerprint),
  KEY idx_staging_batch_no (batch_no),
  KEY idx_staging_status (status),
  KEY idx_staging_conflict_group (conflict_group),
  CONSTRAINT fk_staging_knowledge FOREIGN KEY (knowledge_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 暂存表';
