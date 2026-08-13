"""diart Web Console: 网页版可视化客户端（中继 + 静态页面服务）。

架构
----
    浏览器 (UI)  --WS /ws-->  web_ui.py  --WS ws://host:port-->  diart serve.py
         <--RTTM/状态--        (中继+日志)        <--RTTM 行----

- 浏览器负责采集音频（麦克风/文件），按 0.5s/16000Hz 分块并编码为
  base64(float32 LE PCM) 后经本服务转发给 diart 服务端（与 client.py 协议一致）。
- 服务端返回的 RTTM 文本（每条 WS 消息 = 一个 5s 滑动窗口的全部线段，可能多行）
  由本服务原样透传给浏览器，同时写入会话日志：
    logs/<会话ID>.log    带时间戳的完整日志（人读）
    logs/<会话ID>.rttm   纯 RTTM 行（标准格式，可评估）
    logs/<会话ID>.json   会话元数据 + 全部 RTTM 消息（供历史回放）
    logs/sessions.json   会话索引

用法
----
    终端1: diart serve
    终端2: python -m diart.console.web_ui
    浏览器: http://127.0.0.1:8000
"""

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web
from websocket import WebSocketConnectionClosedException, create_connection

LOG = logging.getLogger("diart.web")

BROWSER_WS = "/ws"
RECONNECT_DELAY = 3.0  # s, 上游断线重连间隔
HEARTBEAT_INTERVAL = 5.0  # s, 服务端在线探测间隔

STATIC_DIR = Path(__file__).resolve().parent / "web_ui"


def _now_str() -> str:
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"


def _wallclock() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# 会话：上游 diart 服务连接 + 日志落盘
# --------------------------------------------------------------------------- #
class Session:
    """一个 diart 推理会话：持有到 serve.py 的 WebSocket 连接，并把 RTTM
    结果写入日志文件、转发给浏览器。断线自动重连（会话仍视为进行中）。"""

    def __init__(
        self,
        sid: str,
        mode: str,
        server_host: str,
        server_port: int,
        log_dir: Path,
        emit: Callable[[dict], None],
    ):
        self.sid = sid
        self.mode = mode  # "mic" | "file"
        self.server_host = server_host
        self.server_port = server_port
        self.log_dir = log_dir
        self.emit = emit  # 线程安全地推送消息给浏览器（内部转 asyncio loop）

        self.started_at = _wallclock()
        self.ended_at: float | None = None
        self.num_messages = 0
        self.num_lines = 0
        self.messages: list[dict[str, Any]] = []  # [{t, text}] 供历史回放

        self._state = "idle"
        # 有界缓冲：上游短暂离线时音频不丢弃（最多 60 块 ≈ 30s，旧块优先淘汰）
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=60)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._rttm_fh = None
        self._log_fh = None
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.rttm_path = log_dir / f"{sid}.rttm"
        self.log_path = log_dir / f"{sid}.log"
        self.json_path = log_dir / f"{sid}.json"

    # -- 生命周期 -------------------------------------------------------- #
    def start(self) -> None:
        self._rttm_fh = open(self.rttm_path, "w", encoding="utf-8")
        self._log_fh = open(self.log_path, "w", encoding="utf-8")
        self._write_log_line(
            f"# diart 会话开始 [{self.mode}] -> {self.server_host}:{self.server_port}"
        )
        self._thread.start()

    def stop(self) -> None:
        """请求结束会话（可重入）。"""
        self._stop_evt.set()
        self._queue.put(None)  # 唤醒发送循环

    def join(self, timeout: float = 8.0) -> None:
        self._thread.join(timeout=timeout)

    @property
    def active(self) -> bool:
        return self._thread.is_alive() and not self._stop_evt.is_set()

    @property
    def connected(self) -> bool:
        return self._state == "online"

    # -- 音频上行 --------------------------------------------------------- #
    def send_audio(self, base64_data: str) -> None:
        """浏览器音频块（base64 text）→ 上游。离线时暂存，重连后按序补发。"""
        if self._stop_evt.is_set():
            return
        while self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._queue.put_nowait(base64_data)
        except queue.Full:
            pass

    # -- 线程主体 --------------------------------------------------------- #
    def _run(self) -> None:
        self._state = "connecting"
        self.emit({"type": "status", "state": "connecting", "session_id": self.sid})
        while not self._stop_evt.is_set():
            try:
                ws = create_connection(
                    f"ws://{self.server_host}:{self.server_port}",
                    timeout=10,
                    enable_multithread=True,
                )
            except Exception as exc:
                self._state = "offline"
                self.emit(
                    {"type": "status", "state": "offline", "session_id": self.sid}
                )
                if self._stop_evt.is_set():
                    break
                LOG.warning("上游连接失败: %s，%.1fs 后重试", exc, RECONNECT_DELAY)
                self._stop_evt.wait(RECONNECT_DELAY)
                continue

            self._state = "online"
            self.emit({"type": "status", "state": "online", "session_id": self.sid})
            recv_t = threading.Thread(target=self._recv_loop, args=(ws,), daemon=True)
            recv_t.start()
            try:
                while not self._stop_evt.is_set():
                    try:
                        item = self._queue.get(timeout=0.5)
                    except queue.Empty:
                        continue  # 空闲唤醒：继续等待，不得跳出循环
                    if item is None:
                        LOG.debug("发送循环: 收到停止哨兵")
                        break
                    ws.send(item)
            except (WebSocketConnectionClosedException, OSError) as exc:
                LOG.info("上游连接中断: %r", exc)
            except Exception:
                LOG.exception("发送音频块失败")
            finally:
                LOG.debug("发送循环退出，关闭上游连接")
                try:
                    ws.close()
                except Exception as exc:
                    LOG.debug("close 异常: %r", exc)
                recv_t.join(timeout=2.0)
                LOG.debug("recv 线程已退出 (alive=%s)", recv_t.is_alive())
                self._state = "offline"
                # 主动停止时不推送 offline（避免 UI 闪现），重连/异常时才推送
                if not self._stop_evt.is_set():
                    self.emit(
                        {"type": "status", "state": "offline", "session_id": self.sid}
                    )
        self._finalize()

    def _recv_loop(self, ws) -> None:
        while not self._stop_evt.is_set():
            try:
                msg = ws.recv()
            except WebSocketConnectionClosedException as exc:
                LOG.debug("recv: 连接关闭 %r", exc)
                break
            except OSError as exc:
                LOG.debug("recv: OSError %r", exc)
                break
            except Exception:
                LOG.exception("接收上游消息失败")
                break
            if not msg or not isinstance(msg, str):
                LOG.debug("recv: 空消息/非文本 (msg=%r), 连接关闭", msg)
                break
            self._handle_message(msg)

    def _handle_message(self, text: str) -> None:
        t = _wallclock()
        with self._lock:
            self.num_messages += 1
            self.messages.append({"t": t, "text": text})
            if self._rttm_fh is not None:
                for line in text.splitlines():
                    if line.strip():
                        self._rttm_fh.write(line.strip() + "\n")
                        self._log_fh.write(f"[{_now_str()}] {line.strip()}\n")
                        self.num_lines += 1
                self._rttm_fh.flush()
                self._log_fh.flush()
        # 原样转发整条消息（前端按“消息=一个窗口批次”去重叠渲染）
        self.emit({"type": "rttm", "t": t, "text": text})

    def _write_log_line(self, line: str) -> None:
        if self._log_fh is not None:
            self._log_fh.write(f"[{_now_str()}] {line}\n")
            self._log_fh.flush()

    def _finalize(self) -> None:
        self.ended_at = _wallclock()
        with self._lock:
            for fh in (self._rttm_fh, self._log_fh):
                if fh is not None:
                    fh.close()
            self._rttm_fh = self._log_fh = None
        meta = {
            "id": self.sid,
            "mode": self.mode,
            "server": f"{self.server_host}:{self.server_port}",
            "start": self.started_at,
            "end": self.ended_at,
            "duration": round(self.ended_at - self.started_at, 2),
            "num_messages": self.num_messages,
            "num_lines": self.num_lines,
            "messages": self.messages,
        }
        try:
            self.json_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            LOG.exception("写入会话 JSON 失败")
        self.emit({"type": "status", "state": "stopped", "session_id": self.sid})
        LOG.info("会话 %s 结束：%d 条 RTTM 行", self.sid, self.num_lines)


# --------------------------------------------------------------------------- #
# 服务
# --------------------------------------------------------------------------- #
class WebConsole:
    def __init__(self, args):
        self.args = args
        self.server_host = args.server_host
        self.server_port = args.server_port
        self.log_dir = Path(args.log_dir).expanduser().resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._browsers: list[web.WebSocketResponse] = []
        self._session: Session | None = None
        self._session_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server_online: bool | None = None
        self._index_html: str | None = None

    # -- 浏览器消息推送 --------------------------------------------------- #
    @staticmethod
    async def _send(ws: web.WebSocketResponse, msg: dict) -> None:
        try:
            await ws.send_str(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    def emit(self, msg: dict) -> None:
        """可被任意线程调用：会话线程 -> asyncio loop。"""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self._loop)

    async def _broadcast(self, msg: dict) -> None:
        payload = json.dumps(msg, ensure_ascii=False)
        for ws in list(self._browsers):
            try:
                await ws.send_str(payload)
            except Exception:
                self._browsers.remove(ws)

    # -- 静态页面 ---------------------------------------------------------- #
    def _load_index(self) -> str:
        if self._index_html is None:
            path = STATIC_DIR / "index.html"
            self._index_html = path.read_text(encoding="utf-8")
        return self._index_html

    # -- 浏览器 WS --------------------------------------------------------- #
    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        self._browsers.append(ws)
        LOG.info("浏览器已连接 (%d)", len(self._browsers))
        # 初始状态
        await self._send(
            ws,
            {
                "type": "init",
                "server_host": self.server_host,
                "server_port": self.server_port,
                "server_online": self._server_online,
            },
        )
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await self._send(ws, {"type": "error", "msg": "非法消息"})
                        continue
                    await self._handle_browser_msg(ws, data)
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break  # 浏览器断开，必须退出循环才能执行清理
                # BINARY/PING/PONG 忽略
        finally:
            if ws in self._browsers:
                self._browsers.remove(ws)
            # 唯一客户端断开时，结束进行中的会话，避免上游连接悬挂
            await self._stop_session()
            LOG.info("浏览器已断开 (%d)", len(self._browsers))
        return ws

    async def _handle_browser_msg(self, ws: web.WebSocketResponse, data: dict) -> None:
        mtype = data.get("type")
        if mtype == "config":
            self.server_host = str(data.get("server_host") or self.server_host)
            self.server_port = int(data.get("server_port") or self.server_port)
            await self._send(
                ws,
                {
                    "type": "init",
                    "server_host": self.server_host,
                    "server_port": self.server_port,
                    "server_online": self._server_online,
                },
            )
        elif mtype == "start":
            await self._start_session(ws, data)
        elif mtype == "audio":
            if self._session is not None and self._session.active:
                data_str = data.get("data", "")
                if data_str:
                    self._session.send_audio(data_str)
        elif mtype == "stop":
            await self._stop_session()
        else:
            await self._send(ws, {"type": "error", "msg": f"未知消息: {mtype}"})

    async def _start_session(self, ws: web.WebSocketResponse, data: dict) -> None:
        async with self._session_lock:
            if self._session is not None and self._session.active:
                await self._send(
                    ws, {"type": "error", "msg": "已有会话进行中，请先停止"}
                )
                return
            mode = data.get("mode", "mic")
            if mode not in ("mic", "file"):
                await self._send(ws, {"type": "error", "msg": f"未知输入模式: {mode}"})
                return
            sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
            self._session = Session(
                sid, mode, self.server_host, self.server_port, self.log_dir, self.emit
            )
            self._session.start()
            LOG.info("会话 %s 开始 (mode=%s)", sid, mode)

    async def _stop_session(self) -> None:
        async with self._session_lock:
            sess, self._session = self._session, None
            if sess is not None:
                sess.stop()
                sess.join(timeout=8.0)
                self._update_index()
                LOG.info("会话 %s 已停止", sess.sid)

    # -- 会话索引 / API ---------------------------------------------------- #
    def _session_meta_path(self) -> Path:
        return self.log_dir / "sessions.json"

    def _scan_sessions(self) -> list[dict]:
        """启动时从磁盘恢复会话索引（元数据 + 轻量信息）。"""
        sessions = []
        for jpath in sorted(self.log_dir.glob("*.json")):
            if jpath.name == "sessions.json":
                continue
            try:
                meta = json.loads(jpath.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            sessions.append(
                {
                    "id": meta.get("id", jpath.stem),
                    "mode": meta.get("mode", "?"),
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                    "duration": meta.get("duration"),
                    "num_messages": meta.get("num_messages", 0),
                    "num_lines": meta.get("num_lines", 0),
                    "server": meta.get("server", ""),
                    "has_rttm": (self.log_dir / f"{jpath.stem}.rttm").exists(),
                }
            )
        sessions.sort(key=lambda s: s.get("start") or 0, reverse=True)
        return sessions

    def _update_index(self) -> None:
        try:
            self._session_meta_path().write_text(
                json.dumps(
                    {"sessions": self._scan_sessions()}, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
        except OSError:
            LOG.exception("更新会话索引失败")

    def _load_session_meta(self, sid: str) -> dict | None:
        path = self.log_dir / f"{sid}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    # -- REST API ---------------------------------------------------------- #
    def _routes(self) -> web.Application:
        app = web.Application()

        async def index(_: web.Request) -> web.Response:
            return web.Response(text=self._load_index(), content_type="text/html")

        async def api_sessions(_: web.Request) -> web.Response:
            return web.json_response({"sessions": self._scan_sessions()})

        async def api_session_detail(request: web.Request) -> web.Response:
            meta = self._load_session_meta(request.match_info["sid"])
            if meta is None:
                raise web.HTTPNotFound()
            return web.json_response(meta)

        async def api_session_file(request: web.Request) -> web.Response:
            sid = request.match_info["sid"]
            kind = request.match_info["kind"]  # rttm | json | log
            fname = {"rttm": "rttm", "json": "json", "log": "log"}.get(kind)
            if fname is None:
                raise web.HTTPNotFound()
            path = self.log_dir / f"{sid}.{fname}"
            if not path.exists():
                raise web.HTTPNotFound()
            ctype = {
                "rttm": "text/plain",
                "json": "application/json",
                "log": "text/plain",
            }[kind]
            return web.Response(
                text=path.read_text(encoding="utf-8"),
                content_type=ctype,
                headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
            )

        async def api_session_delete(request: web.Request) -> web.Response:
            sid = request.match_info["sid"]
            deleted = []
            for ext in (".rttm", ".log", ".json"):
                p = self.log_dir / f"{sid}{ext}"
                if p.exists():
                    p.unlink()
                    deleted.append(p.name)
            self._update_index()
            return web.json_response({"deleted": deleted})

        app.router.add_get("/", index)
        app.router.add_get("/index.html", index)
        app.router.add_get("/api/sessions", api_sessions)
        app.router.add_get("/api/sessions/{sid}", api_session_detail)
        app.router.add_get("/api/sessions/{sid}/files/{kind}", api_session_file)
        app.router.add_delete("/api/sessions/{sid}", api_session_delete)
        app.router.add_get(BROWSER_WS, self._ws_handler)
        return app

    # -- 上游在线探测 ------------------------------------------------------- #
    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            # 会话进行中由会话状态直接反映，不再额外探测
            if self._session is not None and self._session.active:
                continue
            # 探测是阻塞 TCP/WS 操作，放线程池执行，避免卡住事件循环
            online = await asyncio.to_thread(self._probe_server)
            # 每次探测都推送，方便新连接的浏览器及时拿到当前状态
            if online != self._server_online or self._browsers:
                self._server_online = online
                await self._broadcast({"type": "server", "online": online})

    def _probe_server(self) -> bool:
        """探测上游是否可用。必须完成完整 WebSocket 握手后立即关闭：
        仅裸 TCP 连接会让上游 websocket-server 的 read_http_headers 读不到
        GET 请求而打印 AssertionError。握手成功的连接无副作用（diart 只在
        收到消息时才替换 self.client，且会话进行中不探测）。"""
        try:
            ws = create_connection(
                f"ws://{self.server_host}:{self.server_port}",
                timeout=2,
            )
            ws.close()
            return True
        except Exception:
            return False

    # -- 启动 -------------------------------------------------------------- #
    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        runner = web.AppRunner(self._routes())
        await runner.setup()
        site = web.TCPSite(runner, self.args.host, self.args.port)
        await site.start()
        LOG.info("diart Web Console: http://%s:%d", self.args.host, self.args.port)
        LOG.info("上游 diart 服务: ws://%s:%d", self.server_host, self.server_port)
        LOG.info("日志目录: %s", self.log_dir)
        if self.args.open:
            threading.Timer(
                0.5,
                lambda: webbrowser.open(f"http://{self.args.host}:{self.args.port}"),
            ).start()
        await self._heartbeat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="diart 网页版可视化客户端（中继服务 + 页面服务）"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        type=str,
        help="页面服务监听地址 (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", default=8003, type=int, help="页面服务端口 (default: 8000)"
    )
    parser.add_argument(
        "--server-host",
        default="localhost",
        type=str,
        help="diart serve 服务地址 (default: localhost)",
    )
    parser.add_argument(
        "--server-port",
        default=7007,
        type=int,
        help="diart serve 服务端口 (default: 7007)",
    )
    parser.add_argument(
        "--log-dir", default="logs", type=str, help="会话日志目录 (default: logs)"
    )
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser


def run():
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    console = WebConsole(args)
    try:
        asyncio.run(console.run())
    except KeyboardInterrupt:
        LOG.info("退出")


if __name__ == "__main__":
    run()
