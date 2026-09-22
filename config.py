from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 0.2
    api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    epochs_per_update: int = 10
    batch_size: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    training_episodes: int = 200
    convergence_kappa: float = 0.05


@dataclass
class ECSConfig:
    rho: float = 0.35
    cvss_min_threshold: float = 7.0
    eta: float = 0.6
    alpha: float = 1.0
    beta: float = 0.5
    lam: float = 0.3


@dataclass
class ExecutionConfig:
    max_cycles: int = 20
    max_retries: int = 5
    esi_threshold: float = 0.4
    defense_shift_threshold: float = 0.15
    phase_timeout_seconds: int = 300
    react_max_steps: int = 10


@dataclass
class NVDConfig:
    base_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    api_key: str = field(default_factory=lambda: os.getenv("NVD_API_KEY", ""))
    results_per_page: int = 200
    request_delay: float = 0.6


@dataclass
class MetasploitConfig:
    host: str = field(default_factory=lambda: os.getenv("MSF_HOST", "127.0.0.1"))
    port: int = 55553
    username: str = field(default_factory=lambda: os.getenv("MSF_USER", "msf"))
    password: str = field(default_factory=lambda: os.getenv("MSF_PASS", ""))
    ssl: bool = False


@dataclass
class AttackConfig:
    stix_url: str = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    local_cache_path: str = "data/attack_stix21.json"


@dataclass
class EngageConfig:
    host: str = field(default_factory=lambda: os.getenv("ENGAGE_HOST", "127.0.0.1"))
    api_port: int = 8443
    api_key: str = field(default_factory=lambda: os.getenv("ENGAGE_API_KEY", ""))
    scenarios: list = field(default_factory=lambda: ["S1", "S2", "S3", "S4", "S5"])


@dataclass
class AgentRedConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ecs: ECSConfig = field(default_factory=ECSConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    nvd: NVDConfig = field(default_factory=NVDConfig)
    metasploit: MetasploitConfig = field(default_factory=MetasploitConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    engage: EngageConfig = field(default_factory=EngageConfig)
    embedding_model: str = "all-MiniLM-L6-v2"
    seed: int = 42
    results_dir: str = "results"
    device: str = "cuda"


DEFAULT_CONFIG = AgentRedConfig()
