from __future__ import annotations
import requests
import socket
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from agents.base import BaseAgent
from mdp.state import AttackState


RECON_SYSTEM_PROMPT = """
You are ReconAgent, a specialist reconnaissance agent operating within an authorised red-team evaluation.
Your objective is to enumerate target hosts, identify open ports, running services, and software versions,
and map each discovered service to CPE identifiers for downstream vulnerability analysis.
Proceed methodically: first discover live hosts, then enumerate services, then extract CPE strings.
Reason step by step before calling any tool. Report only observed facts.
"""


class ReconAgent(BaseAgent):
    def __init__(self, llm_config, exec_config, memory, shodan_api_key: str = "") -> None:
        self.shodan_api_key = shodan_api_key
        super().__init__(
            name="ReconAgent",
            system_prompt=RECON_SYSTEM_PROMPT,
            llm_config=llm_config,
            exec_config=exec_config,
            memory=memory,
        )

    def _register_tools(self) -> None:
        self.tools.register(
            name="nmap_scan",
            fn=self._nmap_scan,
            schema={
                "name": "nmap_scan",
                "description": "Run an Nmap service and version detection scan against a target IP range.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "IP or CIDR range"},
                        "ports": {"type": "string", "description": "Port specification, e.g. '1-1024' or 'top1000'"},
                    },
                    "required": ["target"],
                },
            },
        )
        self.tools.register(
            name="shodan_lookup",
            fn=self._shodan_lookup,
            schema={
                "name": "shodan_lookup",
                "description": "Query Shodan for a given IP address to retrieve banners, services, and CVEs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ip": {"type": "string", "description": "Target IP address"},
                    },
                    "required": ["ip"],
                },
            },
        )
        self.tools.register(
            name="cpe_lookup",
            fn=self._cpe_lookup,
            schema={
                "name": "cpe_lookup",
                "description": "Convert a service name and version string to a NIST CPE 2.3 identifier.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "version": {"type": "string"},
                        "vendor": {"type": "string"},
                    },
                    "required": ["product", "version"],
                },
            },
        )

    def _nmap_scan(self, target: str, ports: str = "top1000") -> Dict:
        if ports == "top1000":
            args = ["nmap", "-sV", "--top-ports", "1000", "-oX", "-", target]
        else:
            args = ["nmap", "-sV", "-p", ports, "-oX", "-", target]
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=120)
            return {"raw_xml": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {"error": str(exc)}

    def _shodan_lookup(self, ip: str) -> Dict:
        if not self.shodan_api_key:
            return {"error": "Shodan API key not configured"}
        url = f"https://api.shodan.io/shodan/host/{ip}"
        try:
            resp = requests.get(url, params={"key": self.shodan_api_key}, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "os": data.get("os"),
                    "ports": data.get("ports", []),
                    "services": [
                        {
                            "port": svc.get("port"),
                            "transport": svc.get("transport"),
                            "product": svc.get("product"),
                            "version": svc.get("version"),
                            "cpe": svc.get("cpe", []),
                        }
                        for svc in data.get("data", [])
                    ],
                    "vulns": list(data.get("vulns", {}).keys()),
                }
            return {"error": f"HTTP {resp.status_code}"}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def _cpe_lookup(self, product: str, version: str, vendor: str = "*") -> Dict:
        url = "https://services.nvd.nist.gov/rest/json/cpes/2.0"
        params = {"keywordSearch": f"{vendor} {product} {version}", "resultsPerPage": 5}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                products = resp.json().get("products", [])
                cpe_strings = [p["cpe"]["cpeName"] for p in products if "cpe" in p]
                return {"cpe_strings": cpe_strings}
            return {"error": f"HTTP {resp.status_code}"}
        except requests.RequestException as exc:
            return {"error": str(exc)}

    def run(self, state: AttackState, task: str) -> Tuple[bool, str, Any]:
        success, steps, result = self._react_loop(state, task)
        return success, "\n".join(s.observation for s in steps), result

    def build_asset_inventory(
        self,
        state: AttackState,
        target_range: str,
    ) -> List[Dict]:
        task = f"Enumerate all live hosts in {target_range}. For each host identify open ports, services, versions, and CPE strings."
        success, _, result = self.run(state, task)
        if isinstance(result, dict) and "services" in result:
            return result["services"]
        return []
