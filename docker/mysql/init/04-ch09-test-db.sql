-- CH09 专用 MySQL 测试库；仅首次空卷初始化执行，已有卷由管理员显式执行。
-- 不清库、不修改应用库；数据库级授权转义下划线，避免匹配其他数据库。
SET NAMES utf8mb4;
CREATE DATABASE IF NOT EXISTS mewhelp_ch09_test
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
GRANT ALL PRIVILEGES ON `mewhelp\_ch09\_test`.* TO 'mewhelp'@'%';
