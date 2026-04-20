## Cross-lingual News Consistency Checker
 
Detects when international news outlets contradict each other —
in any language, without translation.
 
---
 
## Quick Start (3 steps)
 
### Step 1 — Create a new conda environment
```bash
conda create -n news_nli python=3.14
```
 
### Step 2 — Activate the environment
```bash
conda activate news_nli
```
 
### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
conda install -c conda-forge plotly
```
> First install takes 3–5 minutes (PyTorch + Transformers are large).