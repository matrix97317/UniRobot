# -*- coding: utf-8 -*-
"""Unirobot brain http server."""
import logging
import time
import traceback
import io
from typing import Optional
from typing import Dict
from typing import Any

from fastapi import (
    FastAPI,
    Body,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from unirobot.utils.msgpack_numpy import Packer, unpackb
from unirobot.brain.utils.filter_algo import SimpleKalmanFilter

logger = logging.getLogger(__name__)


class FastAPIHTTPPolicyServer:
    """使用 FastAPI 提供 HTTP 策略服务 - 支持二进制数据"""

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8443,
        metadata: Optional[dict] = None,
        max_batch_size: int = 10,
        infer_chunk_step: int = 0,
        use_kf: bool = False,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._max_batch_size = max_batch_size
        self._metadata = metadata or {}
        self._active_requests = 0
        self._total_requests = 0
        self._infer_cnt = 0
        self._action = None
        self._infer_chunk_step = infer_chunk_step
        self._use_kf = use_kf
        self._kl = None
        self._packer = Packer()

        # 创建 FastAPI 应用
        self._app = self.create_app()

    def create_app(self) -> FastAPI:
        """创建 FastAPI 应用"""
        app = FastAPI(
            title="HTTP Policy Server - Binary Support",
            description="通过 HTTP REST API 提供策略推理服务，支持二进制数据",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # 添加 CORS 中间件
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 注册路由
        self._register_routes(app)
        return app

    def _register_routes(self, app: FastAPI):
        """注册路由"""

        @app.get("/", response_model=Dict[str, Any])
        async def root():
            """根端点"""
            return {
                "message": "HTTP Policy Server with Binary Support is running",
                "status": "healthy",
                "endpoints": {
                    "health_check": "/healthz",
                    "metadata": "/metadata",
                    "msgpack_inference": "/msgpack_infer",
                    "stats": "/stats",
                    "docs": "/docs",
                },
            }

        @app.get("/healthz", response_model=Dict[str, Any])
        async def health_check():
            """健康检查端点"""
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "active_connections": self._active_requests,
            }

        @app.get("/metadata", response_model=Dict[str, Any])
        async def get_metadata():
            """获取服务器元数据"""
            return self._metadata

        @app.post("/msgpack_infer")
        async def msgpack_infer(data: bytes = Body(...)):
            """MsgPack 格式推理端点"""
            self._active_requests += 1
            self._total_requests += 1

            try:
                start_time = time.monotonic()
                request_id = f"msgpack_{int(time.time()*1000)}"

                logger.info(
                    f"收到 MsgPack 推理请求 {request_id}, 数据大小: {len(data)} bytes"
                )

                # 解析 msgpack 数据
                obs = unpackb(data)

                # 执行推理
                infer_start = time.perf_counter()
                if self._infer_cnt == 0:
                    self._action = self._policy(obs)
                    # import pickle
                    # with open("action.pkl","wb") as fin:
                    #     pickle.dump(self._action,fin)

                model_action = self._action[self._infer_cnt, :]
                model_action[5] = self._action[self._infer_cnt + 30, 5]
                if model_action[5] > 25:

                    # model_action[2] =  self._action[self._infer_cnt+60, 2]
                    model_action[3] = self._action[self._infer_cnt + 50, 3]
                # if model_action[5]<2:
                #     model_action[5] =  self._action[self._infer_cnt+60, 5]

                self._infer_cnt = self._infer_cnt + 1
                if self._infer_cnt >= self._infer_chunk_step:
                    self._infer_cnt = 0
                infer_time = time.perf_counter() - infer_start
                if self._use_kf:
                    if self._active_requests == 1:
                        self._kl = SimpleKalmanFilter(
                            process_variance=0.01,
                            measurement_variance=0.1,
                            initial_position=model_action,
                        )
                    self._kl.predict()
                    filtered_position = self._kl.update(model_action)
                    model_action = filtered_position

                # 构建响应
                response_data = {
                    "action": model_action,
                    "server_timing": {
                        "infer_ms": infer_time * 1000,
                        "total_ms": (time.monotonic() - start_time) * 1000,
                    },
                    "request_id": request_id,
                    "status": "success",
                    "timestamp": time.time(),
                }

                # 返回 msgpack 响应
                packed_response = self._packer.pack(response_data)

                logger.info(
                    f"MsgPack 推理完成 {request_id}, 耗时: {infer_time*1000:.2f}ms"
                )

                return StreamingResponse(
                    io.BytesIO(packed_response),
                    media_type="application/msgpack",
                    headers={
                        "X-Request-ID": request_id,
                        "X-Inference-Time": f"{infer_time*1000:.2f}ms",
                    },
                )

            except Exception as e:
                logger.error(f"MsgPack 推理失败: {e}")
                error_data = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "status": "error",
                    "timestamp": time.time(),
                }
                packed_error = self._packer.pack(error_data)

                return StreamingResponse(
                    io.BytesIO(packed_error),
                    media_type="application/msgpack",
                    status_code=500,
                )
            finally:
                self._active_requests -= 1

    def serve_forever(self):
        """启动服务器"""
        logger.info(f"启动 FastAPI HTTP 服务器在 {self._host}:{self._port}")
        logger.info(f"API 文档: http://{self._host}:{self._port}/docs")
        uvicorn.run(
            self._app,
            host=self._host,
            port=self._port,
            log_level="info",
            access_log=True,
        )
