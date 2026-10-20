-- =============================================================
-- ch03 · 合成历史客服对话(喂挖知识 job)。只补缺失项,保留已有记录。
-- =============================================================
SET NAMES utf8mb4;

-- 段1 查询:只核对本脚本的固定会话,不影响其他 seed 用户。
SELECT COUNT(*) AS before_conv FROM conversations
 WHERE (session_id = 'ch03-seed-session-1' AND user_id = 'seed-u1')
    OR (session_id = 'ch03-seed-session-2' AND user_id = 'seed-u2');

-- 标识若已被其他用户占用,列出冲突;后续跳过该会话,不向其追加消息。
SELECT session_id, user_id AS occupied_by FROM conversations
 WHERE (session_id = 'ch03-seed-session-1' AND user_id <> 'seed-u1')
    OR (session_id = 'ch03-seed-session-2' AND user_id <> 'seed-u2');

-- 段2 写入:按固定身份补齐会话和每条消息,不删除会话或关联工单。
START TRANSACTION;

INSERT INTO conversations (session_id, user_id, status)
SELECT 'ch03-seed-session-1', 'seed-u1', '已结束'
 WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE session_id = 'ch03-seed-session-1');

INSERT INTO conversations (session_id, user_id, status)
SELECT 'ch03-seed-session-2', 'seed-u2', '已结束'
 WHERE NOT EXISTS (SELECT 1 FROM conversations WHERE session_id = 'ch03-seed-session-2');

SET @c1 = (SELECT id FROM conversations WHERE session_id = 'ch03-seed-session-1' AND user_id = 'seed-u1');
SET @c2 = (SELECT id FROM conversations WHERE session_id = 'ch03-seed-session-2' AND user_id = 'seed-u2');

-- 同一组问答共用轮次身份,历史问答直接标为 completed,避免被当成中断轮次。
INSERT INTO messages (conversation_id, role, content, turn_id, turn_status)
SELECT @c1, 'user', '你们发货一般多久啊', 'ch03-seed-turn-1', 'completed'
 WHERE @c1 IS NOT NULL AND NOT EXISTS (
   SELECT 1 FROM messages WHERE conversation_id = @c1 AND turn_id = 'ch03-seed-turn-1' AND role = 'user'
 );

INSERT INTO messages (conversation_id, role, content, turn_id, turn_status)
SELECT @c1, 'assistant', '现货商品付款后 48 小时内发货,预售以商品详情页标注时间为准。', 'ch03-seed-turn-1', 'completed'
 WHERE @c1 IS NOT NULL AND NOT EXISTS (
   SELECT 1 FROM messages WHERE conversation_id = @c1 AND turn_id = 'ch03-seed-turn-1' AND role = 'assistant'
 );

INSERT INTO messages (conversation_id, role, content, turn_id, turn_status)
SELECT @c2, 'user', '满多少包邮', 'ch03-seed-turn-2', 'completed'
 WHERE @c2 IS NOT NULL AND NOT EXISTS (
   SELECT 1 FROM messages WHERE conversation_id = @c2 AND turn_id = 'ch03-seed-turn-2' AND role = 'user'
 );

INSERT INTO messages (conversation_id, role, content, turn_id, turn_status)
SELECT @c2, 'assistant', '单笔订单满 99 元包邮,未满收取 6 元运费,偏远地区另计。', 'ch03-seed-turn-2', 'completed'
 WHERE @c2 IS NOT NULL AND NOT EXISTS (
   SELECT 1 FROM messages WHERE conversation_id = @c2 AND turn_id = 'ch03-seed-turn-2' AND role = 'assistant'
 );

COMMIT;

-- 段3 验证:无身份冲突且从空数据开始时,应有 2 个会话和 4 条 completed 消息。
SELECT COUNT(*) AS after_conv FROM conversations
 WHERE (session_id = 'ch03-seed-session-1' AND user_id = 'seed-u1')
    OR (session_id = 'ch03-seed-session-2' AND user_id = 'seed-u2');
SELECT COUNT(*) AS after_msg FROM messages
 WHERE (conversation_id = @c1 AND turn_id = 'ch03-seed-turn-1')
    OR (conversation_id = @c2 AND turn_id = 'ch03-seed-turn-2');
SELECT conversation_id, turn_id, role, turn_status, content FROM messages
 WHERE (conversation_id = @c1 AND turn_id = 'ch03-seed-turn-1')
    OR (conversation_id = @c2 AND turn_id = 'ch03-seed-turn-2')
 ORDER BY conversation_id, id;
