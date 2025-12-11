# Prompts

Prompt templates used across three research questions for smell detection and false positive filtering.

## rq1_pure_llm/ — Direct Detection Baselines

**Purpose**: Evaluate pure LLM performance on detecting all 9 security smells

Two prompting strategies tested with Claude-4, Grok-4, GPT-5:

- `01_definition_based.txt` — Semantic definitions of each smell
- `02_static_analysis_rules.txt` — Formal rules + keyword heuristics from GLITCH

## rq2_false_positive_filtering/ — False Positive Detection

**Purpose**: Evaluate LLM's ability to filter GLITCH false positives

Two strategies on 4 noisy smell types:

- `01_definition_based.txt` — Definitions + 5-line code context
- `02_static_analysis_rules.txt` — Formal rules + keyword heuristics

## rq3_pseudo_labels/ — Training Data Generation

**Purpose**: Generate pseudo-labels for knowledge distillation

- `01_claude4_teacher_prompt.txt` — Production prompt
- `smell_definitions.yaml` — Rich definitions with key characteristics and false positive examples
- `static_analysis_rules.yaml` — Formal rules for 4 target smells
