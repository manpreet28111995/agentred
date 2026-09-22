# AgentRed: Autonomous Multi-Agent LLM Framework

Implementation accompanying the paper submitted to IEEE Transactions on Dependable and Secure Computing.

## Requirements

- Python 3.10+
- Metasploit Framework with MSFRPC enabled (`msfrpcd -P <password> -S`)
- DARPA ENGAGE sandbox access
- NVD API key (free at nvd.nist.gov)
- Anthropic API key

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your credentials.

## ATT&CK Data

```bash
mkdir -p data
```

The AttackClient fetches STIX 2.1 data on first run and caches it to `data/attack_stix21.json`.

## Running the Evaluation

```bash
python main.py \
  --device cpu \
  --seed 42 \
  --episodes 200 \
  --max-cycles 20 \
  --rho 0.35 \
  --results-dir results
```

Results are written to `results/agentred_results.json`.

## Structure

```
agentred/
├── config.py               Hyperparameters and API configuration
├── main.py                 Experiment runner
├── mdp/state.py            MDP state, action, reward definitions
├── memory/buffer.py        Shared memory buffer with semantic retrieval
├── agents/
│   ├── base.py             BaseAgent with ReAct loop
│   ├── orchestrator.py     OrchestratorAgent
│   ├── recon.py            ReconAgent
│   ├── exploit.py          ExploitAgent
│   ├── lateral.py          LateralAgent
│   ├── evade.py            EvadeAgent
│   └── exfil.py            ExfilAgent
├── algorithms/
│   ├── ecs.py              Exploit Chain Score (Algorithm 1)
│   └── execution.py        Adaptive Kill-Chain Execution (Algorithm 2)
├── ppo/
│   ├── network.py          Actor-Critic network
│   ├── policy.py           PPO policy wrapper
│   └── trainer.py          PPO training loop
├── integrations/
│   ├── nvd_client.py       NVD API v2.0 client
│   ├── metasploit_client.py Metasploit MSFRPC client
│   └── attack_client.py    MITRE ATT&CK STIX 2.1 client
└── evaluation/
    └── metrics.py          ASR, KCR, ESA, ARR, TTO computation
```

## Ethical Notice

This code is released for authorised red-team evaluation and security research only.
Deployment against systems without explicit written authorisation is prohibited.
All experiments in the accompanying paper were conducted within the DARPA ENGAGE sandbox.
