-- 本章专用容器首次初始化时创建隔离测试库；不操作其他 Docker 项目。
CREATE DATABASE IF NOT EXISTS mewhelp_test CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
GRANT ALL PRIVILEGES ON mewhelp_test.* TO 'mewhelp'@'%';
