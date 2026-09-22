from __future__ import annotations
import os
import json
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table

from config import AgentRedConfig, DEFAULT_CONFIG
from mdp.state import build_initial_state, RewardFunction
from memory.buffer import SharedMemoryBuffer
from algorithms.ecs import ExploitChainScore, ExploitChainEnumerator, ExploitEntry
from algorithms.execution import AdaptiveKillChainExecutor
from ppo.policy import PPOPolicy
from ppo.trainer import PPOTrainer
from integrations.nvd_client import NVDClient
from integrations.metasploit_client import MetasploitClient
from integrations.attack_client import AttackClient
from agents.recon import ReconAgent
from agents.exploit import ExploitAgent
from agents.lateral import LateralAgent
from agents.exfil import ExfilAgent
from agents.evade import EvadeAgent
from agents.orchestrator import OrchestratorAgent
from evaluation.metrics import (
    ConfigurationResult,
    compute_summary,
    per_scenario_analysis,
    wilcoxon_test,
)

console = Console()

ENGAGE_SCENARIOS = {
    "S1": {"name": "Corporate LAN (AD)", "n_hosts": 12, "n_vulns": 30, "n_sensors": 8},
    "S2": {"name": "ICS/SCADA", "n_hosts": 8, "n_vulns": 18, "n_sensors": 6},
    "S3": {"name": "Cloud Microservices", "n_hosts": 11, "n_vulns": 28, "n_sensors": 10},
    "S4": {"name": "Remote Access Infra", "n_hosts": 9, "n_vulns": 22, "n_sensors": 7},
    "S5": {"name": "Enterprise-IoT Mix", "n_hosts": 7, "n_vulns": 16, "n_sensors": 9},
}

SCENARIO_CONFIG_COUNTS = {"S1": 12, "S2": 8, "S3": 11, "S4": 9, "S5": 7}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_framework(cfg: AgentRedConfig):
    nvd = NVDClient(cfg.nvd)
    msf = MetasploitClient(cfg.metasploit)
    attack = AttackClient(cfg.attack)
    attack.load()

    memory = SharedMemoryBuffer(embedding_model=cfg.embedding_model)

    ecs_scorer = ExploitChainScore(rho=cfg.ecs.rho)
    ecs_enumerator = ExploitChainEnumerator(ecs_scorer, cvss_threshold=cfg.ecs.cvss_min_threshold)

    recon = ReconAgent(cfg.llm, cfg.execution, memory)
    exploit = ExploitAgent(cfg.llm, cfg.execution, memory, nvd, msf)
    lateral = LateralAgent(cfg.llm, cfg.execution, memory, msf)
    exfil = ExfilAgent(cfg.llm, cfg.execution, memory, msf)
    evade = EvadeAgent(cfg.llm, cfg.execution, memory, msf)

    orchestrator = OrchestratorAgent(
        llm_config=cfg.llm,
        exec_config=cfg.execution,
        memory=memory,
        recon=recon,
        exploit=exploit,
        lateral=lateral,
        exfil=exfil,
        evade=evade,
        ecs=ecs_scorer,
        eta=cfg.ecs.eta,
    )

    example_scenario = ENGAGE_SCENARIOS["S1"]
    state_dim = (
        example_scenario["n_hosts"]
        + example_scenario["n_vulns"]
        + 7
        + example_scenario["n_sensors"]
    )
    action_dim = 7

    ppo_policy = PPOPolicy(state_dim, action_dim, cfg.ppo, device=cfg.device)
    ppo_trainer = PPOTrainer(ppo_policy, cfg.ppo, device=cfg.device)

    reward_fn = RewardFunction(
        alpha=cfg.ecs.alpha,
        beta=cfg.ecs.beta,
        lam=cfg.ecs.lam,
    )

    executor = AdaptiveKillChainExecutor(
        orchestrator=orchestrator,
        ppo_policy=ppo_policy,
        ppo_trainer=ppo_trainer,
        reward_fn=reward_fn,
        ecs=ecs_scorer,
        max_cycles=cfg.execution.max_cycles,
        max_retries=cfg.execution.max_retries,
        gamma=cfg.ppo.gamma,
    )

    return orchestrator, executor, ecs_enumerator, nvd, memory


def _make_dummy_exploits(n_hosts: int, n_vulns: int) -> list:
    rng = np.random.default_rng(42)
    exploits = []
    for i in range(min(n_vulns, 20)):
        cvss = float(rng.uniform(6.0, 10.0))
        exploits.append(ExploitEntry(
            cve_id=f"CVE-2023-{10000 + i}",
            msf_module=f"exploit/windows/smb/vuln_{i}",
            cvss_base=cvss,
            cvss_exploitability=min(cvss / 10.0 * 1.2, 1.0),
            cwe_id="CWE-119",
            attack_technique="T1190",
            target_host=f"192.168.1.{10 + (i % n_hosts)}",
            target_port=445,
        ))
    return exploits


def run_single_configuration(
    orchestrator,
    executor,
    ecs_enumerator,
    scenario_key: str,
    config_idx: int,
    cfg: AgentRedConfig,
) -> ConfigurationResult:
    scenario = ENGAGE_SCENARIOS[scenario_key]
    n_h, n_v, n_s = scenario["n_hosts"], scenario["n_vulns"], scenario["n_sensors"]

    rng = np.random.default_rng(cfg.seed + config_idx)
    host_values = rng.uniform(0.2, 1.0, n_h).astype(np.float32)
    state = build_initial_state(n_h, n_v, n_s, host_values)

    exploit_candidates = _make_dummy_exploits(n_h, n_v)
    chain = ecs_enumerator.enumerate_chains(exploit_candidates, state)

    objectives = [
        "Compromise domain controller",
        "Exfiltrate credentials",
        "Establish persistence",
    ]

    result = executor.execute(
        initial_state=state,
        exploit_chain=chain,
        objectives=objectives,
        target_host=f"192.168.1.{10}",
    )

    phases_disrupted = sum(1 for r in result.records if not r.success)
    phases_recovered = sum(1 for r in result.records if r.replanned and r.success)

    esa_correct = chain.score > 0.0 and len(chain) <= 3

    return ConfigurationResult(
        config_id=f"{scenario_key}-{config_idx:02d}",
        scenario=scenario_key,
        result=result,
        exploit_selection_correct=esa_correct,
        phases_recovered_after_disruption=phases_recovered,
        phases_disrupted=phases_disrupted,
    )


def pretrain_ppo(executor, cfg: AgentRedConfig) -> None:
    console.print(f"[cyan]PPO offline pre-training: {cfg.ppo.training_episodes} episodes[/cyan]")
    for episode in range(cfg.ppo.training_episodes):
        scenario_key = random.choice(list(ENGAGE_SCENARIOS.keys()))
        scenario = ENGAGE_SCENARIOS[scenario_key]
        n_h, n_v, n_s = scenario["n_hosts"], scenario["n_vulns"], scenario["n_sensors"]
        state = build_initial_state(n_h, n_v, n_s)
        exploits = _make_dummy_exploits(n_h, n_v)

        for _ in range(cfg.execution.max_cycles):
            state_vec = state.to_vector()
            state_tensor = executor.ppo.state_to_tensor(state_vec)
            action, log_prob, value = executor.ppo.act(state_tensor)
            reward = float(np.random.default_rng().normal(0.3, 0.1))
            done = episode % 10 == 9
            executor.trainer.store(state_vec, action, reward, value, log_prob, done)
            if executor.trainer.should_update():
                executor.trainer.update(executor.ppo)
            if done:
                break

        if (episode + 1) % 50 == 0:
            console.print(f"  Episode {episode + 1}/{cfg.ppo.training_episodes}")


def run_evaluation(cfg: AgentRedConfig) -> None:
    set_seed(cfg.seed)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]AgentRed Evaluation")
    orchestrator, executor, ecs_enumerator, nvd, memory = build_framework(cfg)

    pretrain_ppo(executor, cfg)

    all_config_results: list[ConfigurationResult] = []
    total_configs = sum(SCENARIO_CONFIG_COUNTS.values())

    with console.status("[bold green]Running evaluation configurations...") as status:
        for scenario_key, count in SCENARIO_CONFIG_COUNTS.items():
            for idx in range(count):
                status.update(f"[bold green]{scenario_key} config {idx + 1}/{count}")
                cr = run_single_configuration(
                    orchestrator, executor, ecs_enumerator,
                    scenario_key, idx, cfg,
                )
                all_config_results.append(cr)

    all_results = [cr.result for cr in all_config_results]
    summary = compute_summary(all_results, all_config_results)
    scenario_df = per_scenario_analysis(all_config_results)

    table = Table(title="AgentRed Overall Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for k, v in summary.to_dict().items():
        table.add_row(k, str(v))
    console.print(table)

    console.print("\n[bold]Per-Scenario Results:[/bold]")
    console.print(scenario_df.to_string(index=False))

    output_path = Path(cfg.results_dir) / "agentred_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary.to_dict(),
            "per_scenario": scenario_df.to_dict(orient="records"),
            "n_configurations": len(all_config_results),
        }, f, indent=2)
    console.print(f"\n[bold]Results saved to {output_path}[/bold]")


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentRed evaluation runner")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--llm-model", type=str, default="claude-sonnet-4-6")
    parser.add_argument("--rho", type=float, default=0.35)
    args = parser.parse_args()

    cfg = AgentRedConfig()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.results_dir = args.results_dir
    cfg.ppo.training_episodes = args.episodes
    cfg.execution.max_cycles = args.max_cycles
    cfg.llm.model = args.llm_model
    cfg.ecs.rho = args.rho

    run_evaluation(cfg)


if __name__ == "__main__":
    main()
