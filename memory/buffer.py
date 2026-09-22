from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer


@dataclass
class MemoryEntry:
    state_description: str
    action_taken: str
    outcome: str
    phase_reached: int
    success: bool
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        return {
            "state_description": self.state_description,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "phase_reached": self.phase_reached,
            "success": self.success,
        }


class SharedMemoryBuffer:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_entries: int = 512,
    ) -> None:
        self._encoder = SentenceTransformer(embedding_model)
        self._knowledge_base: List[MemoryEntry] = []
        self._max_entries = max_entries
        self._state: Dict[str, Any] = {}
        self._tool_outputs: List[Dict] = []
        self._conversation_history: List[Dict] = []

    def update_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def log_tool_output(self, agent: str, tool: str, output: Any) -> None:
        self._tool_outputs.append({
            "agent": agent,
            "tool": tool,
            "output": output,
        })

    def append_message(self, role: str, content: str) -> None:
        self._conversation_history.append({"role": role, "content": content})

    def get_recent_messages(self, n: int = 10) -> List[Dict]:
        return self._conversation_history[-n:]

    def add_episode(self, entry: MemoryEntry) -> None:
        embedding = self._encoder.encode(
            entry.state_description,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        entry.embedding = embedding
        self._knowledge_base.append(entry)
        if len(self._knowledge_base) > self._max_entries:
            self._knowledge_base.pop(0)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        success_only: bool = False,
    ) -> List[MemoryEntry]:
        if not self._knowledge_base:
            return []
        pool = [e for e in self._knowledge_base if e.success] if success_only else self._knowledge_base
        if not pool:
            pool = self._knowledge_base
        query_emb = self._encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.stack([e.embedding for e in pool])
        scores = embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [pool[i] for i in top_indices]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def export(self, path: str) -> None:
        records = [e.to_dict() for e in self._knowledge_base]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)

    def clear_tool_outputs(self) -> None:
        self._tool_outputs.clear()

    def clear_conversation(self) -> None:
        self._conversation_history.clear()

    def snapshot(self) -> Dict:
        return {
            "state_keys": list(self._state.keys()),
            "knowledge_base_size": len(self._knowledge_base),
            "conversation_turns": len(self._conversation_history),
            "tool_calls_logged": len(self._tool_outputs),
        }
