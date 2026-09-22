from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from pymetasploit3.msfrpc import MsfRpcClient
from config import MetasploitConfig


class MetasploitClient:
    def __init__(self, config: MetasploitConfig) -> None:
        self.config = config
        self._client: Optional[MsfRpcClient] = None

    def connect(self) -> None:
        self._client = MsfRpcClient(
            password=self.config.password,
            server=self.config.host,
            port=self.config.port,
            ssl=self.config.ssl,
            username=self.config.username,
        )

    def _ensure_connected(self) -> None:
        if self._client is None:
            self.connect()

    def search_modules(self, query: str) -> List[Dict]:
        self._ensure_connected()
        results = self._client.modules.search(query)
        modules = []
        for r in results[:20]:
            modules.append({
                "fullname": r.get("fullname", ""),
                "name": r.get("name", ""),
                "rank": r.get("rank", ""),
                "type": r.get("type", ""),
            })
        return modules

    def execute(
        self,
        module_path: str,
        rhosts: str,
        rport: int,
        payload: str = "generic/shell_reverse_tcp",
        extra_options: Optional[Dict] = None,
    ) -> Dict:
        self._ensure_connected()
        try:
            module_type = "exploit" if module_path.startswith("exploit/") else "auxiliary"
            module = self._client.modules.use(module_type, module_path)
            module["RHOSTS"] = rhosts
            module["RPORT"] = rport
            if extra_options:
                for k, v in extra_options.items():
                    module[k] = v

            if module_type == "exploit":
                payload_obj = self._client.modules.use("payload", payload)
                result = module.execute(payload=payload_obj)
            else:
                result = module.execute()

            time.sleep(2)
            sessions = self._client.sessions.list
            new_session = max(sessions.keys()) if sessions else None
            return {
                "status": "success",
                "job_id": result.get("job_id"),
                "session_id": new_session,
            }
        except Exception as exc:
            return {"status": "failure", "error": str(exc)}

    def run_post_module(
        self,
        session_id: str,
        module: str,
        options: Dict,
    ) -> Dict:
        self._ensure_connected()
        try:
            post = self._client.modules.use("post", module)
            post["SESSION"] = session_id
            for k, v in options.items():
                post[k] = v
            result = post.execute()
            return {"status": "success", "result": str(result)}
        except Exception as exc:
            return {"status": "failure", "error": str(exc)}

    def list_sessions(self) -> Dict:
        self._ensure_connected()
        return self._client.sessions.list

    def kill_session(self, session_id: str) -> bool:
        self._ensure_connected()
        try:
            self._client.sessions.session(session_id).stop()
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        self._client = None
