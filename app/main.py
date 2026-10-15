"""FastAPI 应用程序核心入口模块。

负责构建 FastAPI 主应用实例，配置全局基础元数据（服务名称、版本号与描述），
并完成各业务 API 路由器的集中注册与挂载。
"""

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router

# 实例化主 FastAPI 应用，注入全局服务元数据
app = FastAPI(
    title="MewHelp E-Commerce Customer Service",
    description="MewHelp 电商智能客服系统与售后工单结构化提取服务 API",
    version="0.1.0",
)

# 注册售后诉求提取接口路由
app.include_router(extract_router)
# 注册客服多轮对话流式接口路由
app.include_router(chat_router)
