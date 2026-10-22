-- CH04 专用 MySQL 测试库；仅首次空卷初始化执行，已有卷由管理员显式执行。
-- 不清库、不修改 mewhelp、mewhelp_test 或 mewhelp_ch03_test。
SET NAMES utf8mb4;
CREATE DATABASE IF NOT EXISTS mewhelp_ch04_test
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
GRANT ALL PRIVILEGES ON mewhelp_ch04_test.* TO 'mewhelp'@'%';
