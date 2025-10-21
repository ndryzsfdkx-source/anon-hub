# IntelliSA

> An Intelligent Analyzer for IaC Security Smell Detection via Rule and Neural Inference

This is the research hub for **IntelliSA**, a system that combines static analysis with Large Language Models to reduce false positives in Infrastructure-as-Code (IaC) security scanning.

**Paper**: "IntelliSA: An Intelligent Analyzer for IaC Security Smell Detection via Rule and Neural Inference"

## Overview

**Problem**: Static analysis tools generate high false positive rates, causing alert fatigue and preventing widespread adoption.

**Solution**: IntelliSA uses rule-based detection combined with neural inference to distinguish true security vulnerabilities from false alarms while maintaining high recall.

**Target**: 9 security smell categories across Ansible, Chef, and Puppet.

## Repositories

### 1. IntelliSA-Experiments

Early-stage methodology exploration and evaluation.

- Comparative evaluation of pure vs post-filter LLM approaches
- Pseudo-labeled dataset generation

**GitHub**: `IntelliSA-Experiments`

### 2. IntelliSA-Models

Systematic training pipeline for encoder models.

- Broad candidate selection and Focused hyperparameter tuning
- Final optimization and Multi-seed stability testing

**GitHub**: `IntelliSA-Models`

### 3. IntelliSA-CLI

Production-ready CLI tool implementing the IntelliSA method.

- Post-filters detections using neural inference
- Supports Ansible, Chef, Puppet
- Outputs SARIF, JSONL, CSV formats
- Ready for CI/CD integration

**GitHub**: `intellisa-cli`

## Paper Materials

- Pre-print: `paper/preprint.pdf`
- Camera-ready: `paper/camera-ready.pdf`
- Supplementary: `paper/supplementary/`
- Citation: `paper/citation.bib`

## Artifact Reproducibility

See `artifact/release-manifest.yaml` for pinned commit SHAs, model versions, dataset versions, and tool dependencies used to generate paper results.

Reproduction scripts:

```bash
./artifact/checkout-pinned.sh
./artifact/reproduce-paper-results.sh
```

## Documentation

- [Links](links.md) - Deep links to sub-repos and resources
- User guide for CLI Tool: `intellisa-cli/docs/USER_HANDBOOK.md`

## License

Research code: Apache 2.0 (see individual repo licenses)

## Citation

```bibtex
PLACEHOLDER
```
