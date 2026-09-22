from __future__ import annotations
import time
import anthropic
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from config import LLMConfig, ExecutionConfig
from memory.buffer import SharedMemoryBuffer
from mdp.state import AttackState, AttackAction


@dataclass
class ReActStep:
    thought: str
    action_name: str
    action_input: Dict
    observation: str
    success: bool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Callable] = {}
        self._schemas: Dict[str, Dict] = {}

    def register(self, name: str, fn: Callable, schema: Dict) -> None:
        self._tools[name] = fn
        self._schemas[name] = schema

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name](**kwargs)

    def available_tools(self) -> List[Dict]:
        return list(self._schemas.values())


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_config: LLMConfig,
        exec_config: ExecutionConfig,
        memory: SharedMemoryBuffer,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.llm_config = llm_config
        self.exec_config = exec_config
        self.memory = memory
        self._client = anthropic.Anthropic(api_key=llm_config.api_key)
        self.tools = ToolRegistry()
        self._register_tools()

    @abstractmethod
    def _register_tools(self) -> None:
        pass

    @abstractmethod
    def run(self, state: AttackState, task: str) -> Tuple[bool, str, Any]:
        pass

    def _build_context(self, state: AttackState, task: str) -> str:
        similar = self.memory.retrieve(task, top_k=2, success_only=True)
        memory_block = ""
        if similar:
            memory_block = "\n".join([
                f"Prior success: {e.action_taken} -> {e.outcome}"
                for e in similar
            ])
        recent = self.memory.get_recent_messages(n=6)
        history_block = "\n".join([
            f"{m['role'].upper()}: {m['content']}" for m in recent
        ])
        phase_names = [
            "recon", "weaponize", "deliver",
            "exploit", "install", "c2", "exfil",
        ]
        phase_status = ", ".join([
            f"{phase_names[i]}={'DONE' if state.phase_complete[i] else 'PENDING'}"
            for i in range(len(phase_names))
        ])
        compromised = int(np.sum(state.host_compromised)) if hasattr(state, 'host_compromised') else 0
        context = (
            f"TASK: {task}\n"
            f"KILL-CHAIN STATUS: {phase_status}\n"
            f"HOSTS COMPROMISED: {compromised}\n"
            f"DEFENSE POSTURE: {float(np.mean(state.defense_posture)):.2f}\n"
        )
        if memory_block:
            context += f"RELEVANT PRIOR ENGAGEMENTS:\n{memory_block}\n"
        if history_block:
            context += f"RECENT ACTIONS:\n{history_block}\n"
        return context

    def _react_loop(
        self,
        state: AttackState,
        task: str,
        max_steps: Optional[int] = None,
    ) -> Tuple[bool, List[ReActStep], Any]:
        max_steps = max_steps or self.exec_config.react_max_steps
        steps: List[ReActStep] = []
        context = self._build_context(state, task)
        messages = [{"role": "user", "content": context}]
        final_result: Any = None

        for _ in range(max_steps):
            response = self._client.messages.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                system=self.system_prompt,
                messages=messages,
                tools=self.tools.available_tools(),
            )
            thought = ""
            tool_call = None
            for block in response.content:
                if block.type == "text":
                    thought = block.text
                elif block.type == "tool_use":
                    tool_call = block

            if response.stop_reason == "end_turn" or tool_call is None:
                final_result = thought
                steps.append(ReActStep(
                    thought=thought,
                    action_name="finish",
                    action_input={},
                    observation=thought,
                    success=True,
                ))
                break

            try:
                observation = self.tools.call(tool_call.name, **tool_call.input)
                success = True
            except Exception as exc:
                observation = f"ERROR: {exc}"
                success = False

            step = ReActStep(
                thought=thought,
                action_name=tool_call.name,
                action_input=tool_call.input,
                observation=str(observation),
                success=success,
            )
            steps.append(step)
            self.memory.log_tool_output(self.name, tool_call.name, observation)

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": str(observation),
                }],
            })
            final_result = observation

        overall_success = any(s.success for s in steps)
        return overall_success, steps, final_result

    def _completion_check(self, state: AttackState, phase_idx: int) -> bool:
        return bool(state.phase_complete[phase_idx])


import numpy as np
