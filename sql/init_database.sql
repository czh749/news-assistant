-- ======================================
-- 智能新闻助手 - 数据库初始化脚本
-- ======================================

-- 创建数据库
CREATE DATABASE IF NOT EXISTS news_db 
    DEFAULT CHARACTER SET utf8mb4 
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE news_db;

-- ======================================
-- 新闻表
-- ======================================
CREATE TABLE IF NOT EXISTS news (
    id VARCHAR(16) PRIMARY KEY COMMENT '新闻唯一ID（MD16）',
    title VARCHAR(500) NOT NULL COMMENT '新闻标题',
    content LONGTEXT COMMENT '新闻正文内容',
    summary TEXT COMMENT '新闻摘要（AI生成）',
    source VARCHAR(50) DEFAULT 'unknown' COMMENT '新闻来源',
    url VARCHAR(1000) NOT NULL COMMENT '原文链接',
    publish_time DATETIME COMMENT '发布时间',
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '抓取时间',
    author VARCHAR(100) COMMENT '作者',
    category VARCHAR(50) COMMENT '分类',
    tags JSON COMMENT '标签列表（JSON数组）',
    has_summary BOOLEAN DEFAULT FALSE COMMENT '是否已生成摘要',
    has_embedding BOOLEAN DEFAULT FALSE COMMENT '是否已向量化',
    status ENUM('active', 'deleted', 'archived') DEFAULT 'active' COMMENT '状态',
    extra JSON COMMENT '扩展字段（JSON对象）',
    
    INDEX idx_source (source),
    INDEX idx_publish_time (publish_time),
    INDEX idx_crawl_time (crawl_time),
    INDEX idx_has_summary (has_summary),
    INDEX idx_has_embedding (has_embedding),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='新闻主表';

-- ======================================
-- 用户表
-- ======================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) UNIQUE NOT NULL COMMENT '飞书用户ID',
    name VARCHAR(100) COMMENT '用户名称',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- ======================================
-- 用户兴趣表
-- ======================================
CREATE TABLE IF NOT EXISTS user_interests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL COMMENT '用户ID',
    interest_keyword VARCHAR(100) NOT NULL COMMENT '兴趣关键词',
    weight FLOAT DEFAULT 1.0 COMMENT '权重（0-1）',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    
    UNIQUE KEY uk_user_interest (user_id, interest_keyword),
    INDEX idx_user_id (user_id),
    INDEX idx_keyword (interest_keyword)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户兴趣表';

-- ======================================
-- 推送记录表
-- ======================================
CREATE TABLE IF NOT EXISTS push_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL COMMENT '用户ID',
    push_type VARCHAR(50) NOT NULL COMMENT '推送类型（daily/push）',
    content TEXT COMMENT '推送内容',
    news_ids JSON COMMENT '关联新闻ID列表',
    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '发送时间',
    status ENUM('pending', 'sent', 'failed') DEFAULT 'pending' COMMENT '状态',
    
    INDEX idx_user_id (user_id),
    INDEX idx_sent_at (sent_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='推送记录表';

-- ======================================
-- 插入测试数据
-- ======================================
-- INSERT INTO users (user_id, name) VALUES ('test_user_001', '测试用户');
--docker exec -i mysql-news mysql -uroot -p123456 news_db < init_database.sql