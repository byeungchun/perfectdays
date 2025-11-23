## Command list

### News extraction from pqrquet files
- It is in task_general repo under bis.ghe
- script: drs/drsai/news_analysis/news_sentiment_analysis_enko_news_exraction.py

### Embedding generation for news
- It uses Gemma embedding model
- It is in task_general repo under bis.ghe
- Model file path: /rdrive/huggingface_cache/hub/**embeddinggemma**
- script: drs/drasai/news_analysis/embedding_newsarticle.sh

### Setiment analysis for news in pqrquet files
- script: /notebooks/eda_newsembedding.py
- setup: two GPUs required in devspace, venv: openaioss in devspace
- parameters:
  - --news source: RTRS or 3PTY
  - --start year: e.g., 2018
  - --language: en or ko

### Simulation procedure for strategy01
- script: /scripts/run-strategy01-all.ps1
- setup: venv: perfectdays in vdi
- parameters: defined in /config/*.yaml files

