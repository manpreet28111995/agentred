from __future__ import annotations
import json
import os
import requests
from typing import Dict, List, Optional
import stix2
from config import AttackConfig


CWE_TO_ATTACK_TECHNIQUE: Dict[str, str] = {
    "CWE-89": "T1190",
    "CWE-78": "T1059",
    "CWE-79": "T1190",
    "CWE-119": "T1203",
    "CWE-416": "T1203",
    "CWE-22": "T1083",
    "CWE-287": "T1078",
    "CWE-306": "T1078",
    "CWE-502": "T1059",
    "CWE-434": "T1505",
    "CWE-918": "T1090",
    "CWE-611": "T1190",
    "CWE-77": "T1059",
    "CWE-732": "T1222",
    "CWE-352": "T1556",
}

TACTIC_PHASE_MAPPING: Dict[str, int] = {
    "reconnaissance": 0,
    "resource-development": 0,
    "initial-access": 2,
    "execution": 3,
    "persistence": 4,
    "privilege-escalation": 4,
    "defense-evasion": 5,
    "credential-access": 5,
    "discovery": 0,
    "lateral-movement": 4,
    "collection": 6,
    "command-and-control": 5,
    "exfiltration": 6,
    "impact": 6,
}


class AttackClient:
    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self._store: Optional[stix2.MemoryStore] = None
        self._technique_index: Dict[str, Dict] = {}

    def load(self) -> None:
        if os.path.exists(self.config.local_cache_path):
            with open(self.config.local_cache_path, "r") as f:
                bundle = json.load(f)
        else:
            resp = requests.get(self.config.stix_url, timeout=30)
            resp.raise_for_status()
            bundle = resp.json()
            os.makedirs(os.path.dirname(self.config.local_cache_path), exist_ok=True)
            with open(self.config.local_cache_path, "w") as f:
                json.dump(bundle, f)

        self._store = stix2.MemoryStore()
        self._store.load_from_dict(bundle)
        self._build_index()

    def _build_index(self) -> None:
        for obj in self._store.query([stix2.Filter("type", "=", "attack-pattern")]):
            ext_refs = obj.get("external_references", [])
            technique_id = next(
                (r["external_id"] for r in ext_refs if r.get("source_name") == "mitre-attack"),
                None,
            )
            if technique_id:
                phases = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
                self._technique_index[technique_id] = {
                    "id": technique_id,
                    "name": obj.get("name", ""),
                    "phases": phases,
                    "stix_id": obj.get("id"),
                    "description": obj.get("description", "")[:300],
                }

    def get_technique(self, technique_id: str) -> Optional[Dict]:
        return self._technique_index.get(technique_id)

    def techniques_for_tactic(self, tactic: str) -> List[Dict]:
        return [
            t for t in self._technique_index.values()
            if tactic in t.get("phases", [])
        ]

    def cwe_to_technique(self, cwe_id: str) -> Optional[str]:
        return CWE_TO_ATTACK_TECHNIQUE.get(cwe_id)

    def tactic_to_phase_index(self, tactic: str) -> int:
        return TACTIC_PHASE_MAPPING.get(tactic.lower(), -1)

    def lateral_movement_techniques(self) -> List[Dict]:
        return self.techniques_for_tactic("lateral-movement")

    def exfiltration_techniques(self) -> List[Dict]:
        return self.techniques_for_tactic("exfiltration")

    def defense_evasion_techniques(self) -> List[Dict]:
        return self.techniques_for_tactic("defense-evasion")

    def all_technique_ids(self) -> List[str]:
        return list(self._technique_index.keys())
