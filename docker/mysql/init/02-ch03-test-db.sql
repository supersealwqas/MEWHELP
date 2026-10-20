-- CH03 专用 MySQL 测试库；只在空数据卷首次初始化时执行。
-- 已有数据卷请由管理员手动执行同等语句，不要删除已有数据库。
SET NAMES utf8mb4;

CREATE DATABASE IF NOT EXISTS mewhelp_ch03_test
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
GRANT ALL PRIVILEGES ON mewhelp_ch03_test.* TO 'mewhelp'@'%';
FLUSH PRIVILEGES;
