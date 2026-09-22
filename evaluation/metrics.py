from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats
from algorithms.execution import KillChainResult


@dataclass
class ConfigurationResult:
    config_id: str
    scenario: str
    result: KillChainResult
    exploit_selection_correct: bool
    phases_recovered_after_disruption: int
    phases_disrupted: int


@dataclass
class EvaluationSummary:
    asr: float
    kcr: float
    esa: float
    arr: float
    tto_hours: float
    asr_std: float
    kcr_std: float
    esa_std: float
    arr_std: float
    tto_std: float
    n: int

    def to_dict(self) -> Dict:
        return {
            "ASR": f"{self.asr:.1f} ± {self.asr_std:.1f}",
            "KCR": f"{self.kcr:.1f} ± {self.kcr_std:.1f}",
            "ESA": f"{self.esa:.1f} ± {self.esa_std:.1f}",
            "ARR": f"{self.arr:.1f} ± {self.arr_std:.1f}",
            "TTO_h": f"{self.tto_hours:.2f} ± {self.tto_std:.2f}",
            "n": self.n,
        }


def attack_success_rate(results: List[KillChainResult]) -> Tuple[float, float]:
    rates = [r.attack_success_rate * 100.0 for r in results]
    return float(np.mean(rates)), float(np.std(rates, ddof=1))


def kill_chain_completion_rate(results: List[KillChainResult]) -> Tuple[float, float]:
    rates = []
    for r in results:
        completed = float(np.all(r.phase_vector > 0.5))
        rates.append(completed * 100.0)
    return float(np.mean(rates)), float(np.std(rates, ddof=1))


def exploit_selection_accuracy(
    config_results: List[ConfigurationResult],
) -> Tuple[float, float]:
    correct = [1.0 if cr.exploit_selection_correct else 0.0 for cr in config_results]
    mean = float(np.mean(correct)) * 100.0
    std = float(np.std(correct, ddof=1)) * 100.0
    return mean, std


def adaptive_recovery_rate(
    config_results: List[ConfigurationResult],
) -> Tuple[float, float]:
    rates = []
    for cr in config_results:
        if cr.phases_disrupted == 0:
            continue
        rate = cr.phases_recovered_after_disruption / cr.phases_disrupted
        rates.append(rate * 100.0)
    if not rates:
        return 0.0, 0.0
    return float(np.mean(rates)), float(np.std(rates, ddof=1))


def time_to_objective(results: List[KillChainResult]) -> Tuple[float, float]:
    times = [r.time_to_objective for r in results]
    return float(np.mean(times)), float(np.std(times, ddof=1))


def compute_summary(
    results: List[KillChainResult],
    config_results: List[ConfigurationResult],
) -> EvaluationSummary:
    asr_m, asr_s = attack_success_rate(results)
    kcr_m, kcr_s = kill_chain_completion_rate(results)
    esa_m, esa_s = exploit_selection_accuracy(config_results)
    arr_m, arr_s = adaptive_recovery_rate(config_results)
    tto_m, tto_s = time_to_objective(results)
    return EvaluationSummary(
        asr=asr_m, asr_std=asr_s,
        kcr=kcr_m, kcr_std=kcr_s,
        esa=esa_m, esa_std=esa_s,
        arr=arr_m, arr_std=arr_s,
        tto_hours=tto_m, tto_std=tto_s,
        n=len(results),
    )


def wilcoxon_test(
    method_a: List[float],
    method_b: List[float],
) -> Tuple[float, float]:
    stat, p_val = stats.wilcoxon(method_a, method_b, alternative="two-sided")
    return float(stat), float(p_val)


def per_scenario_analysis(
    config_results: List[ConfigurationResult],
) -> pd.DataFrame:
    rows = []
    scenarios = list({cr.scenario for cr in config_results})
    for scenario in sorted(scenarios):
        subset = [cr for cr in config_results if cr.scenario == scenario]
        results = [cr.result for cr in subset]
        asr_m, _ = attack_success_rate(results)
        kcr_m, _ = kill_chain_completion_rate(results)
        rows.append({
            "Scenario": scenario,
            "Configs": len(subset),
            "ASR": round(asr_m, 1),
            "KCR": round(kcr_m, 1),
        })
    return pd.DataFrame(rows).sort_values("Scenario")


def chain_analysis(config_results: List[ConfigurationResult]) -> Dict:
    chain_lengths = []
    for cr in config_results:
        for rec in cr.result.records:
            chain_lengths.append(1)
    return {
        "mean_chain_length": float(np.mean(chain_lengths)) if chain_lengths else 0.0,
        "std_chain_length": float(np.std(chain_lengths)) if chain_lengths else 0.0,
        "total_engagements": len(config_results),
    }
