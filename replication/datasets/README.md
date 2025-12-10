# Datasets

Two datasets for evaluating and training IntelliSA.

## oracle/ — Test Set

**Purpose**: Ground truth for evaluating detection accuracy (RQ1, RQ2, RQ3)

241 manually labeled IaC scripts with 213 security smell instances:
- Ansible: 81 scripts, 44 smells
- Chef: 80 scripts, 104 smells
- Puppet: 80 scripts, 65 smells

Each smell is labeled with (file, line, smell_type). Scripts are from real GitHub repositories, labeled by 7 independent raters.

**Files**:
- `{ansible,chef,puppet}.csv` — Ground truth labels
- `{ansible,chef,puppet}/` — Raw IaC scripts

## training/ — Student Model Training Data

**Purpose**: Pseudo-labeled data for training IntelliSA's student model (RQ3)

2,300 instances labeled by Claude-4 as TP/FP from GLITCH detections:
- Train: 1,840 instances
- Validation: 230 instances  
- Test: 230 instances

Rigorous deduplication ensures zero overlap with oracle test set.

**Files**:
- `{train,dev,test}_data/` — Pseudo-labeled instances per technology
- Each instance: code snippet + TP/FP label from LLM teacher

