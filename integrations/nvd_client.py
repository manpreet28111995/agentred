from __future__ import annotations
import time
import requests
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from config import NVDConfig


class NVDClient:
    def __init__(self, config: NVDConfig) -> None:
        self.config = config
        self._session = requests.Session()
        if config.api_key:
            self._session.headers.update({"apiKey": config.api_key})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get(self, params: Dict) -> Dict:
        time.sleep(self.config.request_delay)
        resp = self._session.get(self.config.base_url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def query_by_cpe(self, cpe_name: str, results_per_page: int = 200) -> List[Dict]:
        params = {
            "cpeName": cpe_name,
            "resultsPerPage": results_per_page,
        }
        raw = self._get(params)
        return self._parse_vulnerabilities(raw)

    def query_by_keyword(self, keyword: str, cvss_min: float = 7.0) -> List[Dict]:
        params = {
            "keywordSearch": keyword,
            "cvssV3Severity": self._cvss_to_severity(cvss_min),
            "resultsPerPage": self.config.results_per_page,
        }
        raw = self._get(params)
        return self._parse_vulnerabilities(raw)

    def get_cve(self, cve_id: str) -> Optional[Dict]:
        params = {"cveId": cve_id}
        raw = self._get(params)
        vulns = self._parse_vulnerabilities(raw)
        return vulns[0] if vulns else None

    def _parse_vulnerabilities(self, raw: Dict) -> List[Dict]:
        results = []
        for item in raw.get("vulnerabilities", []):
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            metrics = cve.get("metrics", {})
            cvss_score = 0.0
            exploitability = 0.0
            if "cvssMetricV31" in metrics:
                m = metrics["cvssMetricV31"][0]["cvssData"]
                cvss_score = m.get("baseScore", 0.0)
                exploitability = m.get("exploitabilityScore", 0.0)
            elif "cvssMetricV30" in metrics:
                m = metrics["cvssMetricV30"][0]["cvssData"]
                cvss_score = m.get("baseScore", 0.0)
                exploitability = m.get("exploitabilityScore", 0.0)
            elif "cvssMetricV2" in metrics:
                m = metrics["cvssMetricV2"][0]["cvssData"]
                cvss_score = m.get("baseScore", 0.0)
                exploitability = m.get("exploitabilityScore", 0.0)

            weaknesses = cve.get("weaknesses", [])
            cwe_id = ""
            if weaknesses:
                descs = weaknesses[0].get("description", [])
                if descs:
                    cwe_id = descs[0].get("value", "")

            cpe_list = []
            for config in cve.get("configurations", []):
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        if match.get("vulnerable"):
                            cpe_list.append(match.get("criteria", ""))

            results.append({
                "cve_id": cve_id,
                "cvss_base_score": cvss_score,
                "cvss_exploitability": exploitability / 10.0,
                "cwe_id": cwe_id,
                "cpe_list": cpe_list,
                "description": cve.get("descriptions", [{}])[0].get("value", ""),
            })
        results.sort(key=lambda x: x["cvss_base_score"], reverse=True)
        return results

    @staticmethod
    def _cvss_to_severity(score: float) -> str:
        if score >= 9.0:
            return "CRITICAL"
        elif score >= 7.0:
            return "HIGH"
        elif score >= 4.0:
            return "MEDIUM"
        return "LOW"
