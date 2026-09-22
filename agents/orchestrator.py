from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from agents.base import BaseAgent
from agents.recon import ReconAgent
from agents.exploit import ExploitAgent
from agents.lateral import LateralAgent
from agents.exfil import ExfilAgent
from agents.evade import EvadeAgent
from algorithms.ecs import ExploitChainScore
from mdp.state import AttackState, AttackAction, KILL_CHAIN_PHASES
from memory.buffer import SharedMemoryBuffer, MemoryEntry


ORCHESTRATOR_SYSTEM_PROMPT = """
You are the OrchestratorAgent directing a multi-agent red-team evaluation within an authorised
sandboxed environment. You decompose the overall attack objective into specialist tasks, dispatch
agents in kill-chain sequence, monitor execution outcomes, and trigger replanning when defenses
are detected.

Current agent roster:
- ReconAgent: reconnaissance, asset discovery, CPE mapping
- ExploitAgent: CVE lookup, Metasploit module selection, exploit execution
- LateralAgent: lateral movement graph, pivoting to high-value hosts
- EvadeAgent: process injection, log suppression, timestomping
- ExfilAgent: data staging, encrypted exfiltration

Dispatch the appropriate agent for each kill-chain phase. If a phase fails, analyse the defensive
observation and adjust the technique selection before re-attempting.
"""


class OrchestratorAgent(BaseAgent):
    def __init__(
        self,
        llm_config,
        exec_config,
        memory: SharedMemoryBuffer,
        recon: ReconAgent,
        exploit: ExploitAgent,
        lateral: LateralAgent,
        exfil: ExfilAgent,
        evade: EvadeAgent,
        ecs: ExploitChainScore,
        eta: float = 0.6,
    ) -> None:
        self.recon = recon
        self.exploit = exploit
        self.lateral = lateral
        self.exfil = exfil
        self.evade = evade
        self.ecs = ecs
        self.eta = eta
        self._q_values: Dict[str, float] = {}
        super().__init__(
            name="OrchestratorAgent",
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            llm_config=llm_config,
            exec_config=exec_config,
            memory=memory,
        )

    def _register_tools(self) -> None:
        self.tools.register(
            name="dispatch_agent",
            fn=self._dispatch_agent,
            schema={
                "name": "dispatch_agent",
                "description": "Dispatch a specialist agent to execute a kill-chain phase task.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "enum": ["ReconAgent", "ExploitAgent", "LateralAgent", "ExfilAgent", "EvadeAgent"],
                        },
                        "task": {"type": "string"},
                        "phase": {"type": "string"},
                    },
                    "required": ["agent_name", "task", "phase"],
                },
            },
        )
        self.tools.register(
            name="read_memory",
            fn=self._read_memory,
            schema={
                "name": "read_memory",
                "description": "Retrieve relevant past engagement episodes from the shared memory buffer.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 3},
                    },
                    "required": ["query"],
                },
            },
        )

    def _dispatch_agent(
        self,
        agent_name: str,
        task: str,
        phase: str,
    ) -> Dict:
        agent_map = {
            "ReconAgent": self.recon,
            "ExploitAgent": self.exploit,
            "LateralAgent": self.lateral,
            "ExfilAgent": self.exfil,
            "EvadeAgent": self.evade,
        }
        agent = agent_map.get(agent_name)
        if agent is None:
            return {"error": f"Unknown agent: {agent_name}"}
        state = self._current_state
        success, log, result = agent.run(state, task)
        self.memory.append_message("system", f"{agent_name} [{phase}]: {log[:500]}")
        return {
            "agent": agent_name,
            "phase": phase,
            "success": success,
            "result": str(result)[:1000],
        }

    def _read_memory(self, query: str, top_k: int = 3) -> Dict:
        entries = self.memory.retrieve(query, top_k=top_k, success_only=True)
        return {
            "entries": [
                {"action": e.action_taken, "outcome": e.outcome}
                for e in entries
            ]
        }

    def select_action(
        self,
        state: AttackState,
        ecs_score: float,
    ) -> AttackAction:
        next_phase = int(np.argmin(state.phase_complete))
        phase_name = KILL_CHAIN_PHASES[next_phase] if next_phase < len(KILL_CHAIN_PHASES) else "done"
        phase_to_agent = {
            "reconnaissance": "ReconAgent",
            "weaponization": "ExploitAgent",
            "delivery": "ExploitAgent",
            "exploitation": "ExploitAgent",
            "installation": "LateralAgent",
            "command_control": "EvadeAgent",
            "actions_objectives": "ExfilAgent",
        }
        agent_name = phase_to_agent.get(phase_name, "ExploitAgent")
        q_val = self._q_values.get(phase_name, 0.0)
        combined_score = q_val + self.eta * ecs_score
        self.memory.update_state("last_ecs_score", ecs_score)
        self.memory.update_state("last_action_phase", phase_name)
        return AttackAction(
            agent_name=agent_name,
            technique_id=f"T-{phase_name[:4].upper()}",
            target_host=self.memory.get_state("primary_target", ""),
        )

    def update_q_value(
        self,
        phase: str,
        reward: float,
        gamma: float,
        next_q: float,
    ) -> None:
        lr = 0.1
        old = self._q_values.get(phase, 0.0)
        self._q_values[phase] = old + lr * (reward + gamma * next_q - old)

    def run(self, state: AttackState, task: str) -> Tuple[bool, str, Any]:
        self._current_state = state
        success, steps, result = self._react_loop(state, task)
        return success, "\n".join(s.observation for s in steps), result

    def record_episode(
        self,
        state_desc: str,
        action: str,
        outcome: str,
        phase: int,
        success: bool,
    ) -> None:
        entry = MemoryEntry(
            state_description=state_desc,
            action_taken=action,
            outcome=outcome,
            phase_reached=phase,
            success=success,
        )
        self.memory.add_episode(entry)
