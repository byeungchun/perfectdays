# %%
import os
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv

# %%
load_dotenv(r"/rdrive/workspace/perfectdays/.env")

# %%
NEWS_PARQUET_MONTH_DIR = os.environ["NEWS_PARQUET_MONTH_DIR"]
INPUT_DIR = Path(NEWS_PARQUET_MONTH_DIR)
OUTPUT_DIR = Path("/rdrive/rtrs_news")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# File pattern: yyyy-mm.parquet, year >= 2020
month_file_re = re.compile(r"(\d{4})-(\d{2})\.parquet$")
parquet_paths = sorted(
    p for p in INPUT_DIR.glob("*.parquet")
    if month_file_re.fullmatch(p.name) and int(p.name[:4]) >= 2020
)

if not parquet_paths:
    raise FileNotFoundError(f"No monthly parquet files (yyyy-mm.parquet, year>=2000) found in {INPUT_DIR}")

# %%
# gemma 27-it
model_id = r'/rdrive_pvc/huggingface_cache/hub/models--google--gemma-3-27b-it/snapshots/005ad3404e59d6023443cb575daa05336842228a'

def build_pipeline(device_id: str):
    return pipeline(
        "text-generation",
        model=model_id,
        dtype="auto",
        device_map=device_id,
    )

# Build one pipeline per GPU
pipe0 = build_pipeline("cuda:0")
pipe1 = build_pipeline("cuda:1")

# Keep a default alias to preserve original behavior where `pipe` is used
pipe = pipe0

# Warm-up/test call (runs once)
messages = [
    {"role": "user", "content": "Are you ready for news sentiment analysis?"},
]
outputs = pipe(messages, max_new_tokens=256)
print(outputs[0]["generated_text"][-1])

# %%
sentiment_prompt = (
    "You are a bilingual (Korean/English) financial market analyst. Read the Korean news excerpt and assess the expected market impact on affected financial assets. "
    "Follow the instructions precisely and return only strict JSON.\n"
    "\nTask\n"
    "1) Classify sentiment using ONLY one of: very positive, positive, neutral, negative, very negative.\n"
    "2) Base your decision on expected near-term market impact (equity prices, credit spreads/default risk, FX, interest rates, macro risk).\n"
    "3) Provide a single-sentence justification under 50 words that cites concrete drivers (e.g., earnings/guidance, regulation, policy, demand/supply, prices, FX, rates, geopolitics, disruptions).\n"
    "4) Identify the most relevant NAICS industry code(s) mentioned or clearly implied. Use the most specific 6-digit codes when possible. Include the code and its label in the form \"CODE - Label\". If unclear, return an empty array []. Limit to up to 3 entries.\n"
    "5) List directly mentioned or clearly implicated company names (Korean or English official names). Do not guess. If none, return an empty array [].\n"
    "\nLabeling guidance (be conservative)\n"
    "- very positive: strong broad upside (e.g., beat + raised guidance, major policy support, material cost relief) likely to lift assets.\n"
    "- positive: modest upside or favorable development.\n"
    "- neutral: limited, mixed, or insufficient information.\n"
    "- negative: modest downside (e.g., miss, mild regulatory risk, soft demand).\n"
    "- very negative: severe downside (e.g., default/bankruptcy risk, sanctions, major accidents, sharp demand collapse).\n"
    "\nFormatting rules\n"
    "- Output must be STRICT JSON with keys: sentiment, justification, related_industry_naics, related_company_names.\n"
    "- Values: sentiment = string; justification = string (<50 words); related_industry_naics = array of strings; related_company_names = array of strings.\n"
    "- No extra text, explanations, comments, or markdown. JSON only.\n"
    "- If the article is in Korean, you may write the justification in Korean or English; keep it concise and factual.\n"
    "\nExample response\n"
    "{{\n"
    "  \"sentiment\": \"very negative\",\n"
    "  \"justification\": \"수출 감소와 가격 하락이 예고되며 마진 압박이 커져 단기 주가와 신용 스프레드에 부정적.\",\n"
    "  \"related_industry_naics\": [\"334413 - Semiconductor and Related Device Manufacturing\"],\n"
    "  \"related_company_names\": [\"Samsung Electronics\", \"SK hynix\"]\n"
    "}}\n"
    "\nText: {text}\n"
)

# %%
def analyse_sentiment(text: str, llm_pipe=None) -> dict[str, object]:
    allowed = {"very positive", "positive", "very negative", "negative", "neutral"}

    def normalize_sentiment(s: str) -> str:
        s = (s or "").strip().lower()
        mapping = {
            "very positive": "very positive",
            "positive": "positive",
            "very negative": "very negative",
            "negative": "negative",
            "neutral": "neutral",
            "pos": "positive",
            "neg": "negative",
            "bullish": "positive",
            "bearish": "negative",
            "unknown": "neutral",
            "error": "neutral",
        }
        return mapping.get(s, "neutral")

    def trim_words(s: str, max_words: int = 50) -> str:
        if not isinstance(s, str):
            return ""
        words = s.strip().split()
        return s.strip() if len(words) <= max_words else " ".join(words[:max_words])

    def to_string_list(val) -> list[str]:
        if val is None:
            return []
        if isinstance(val, list):
            out = []
            for x in val:
                if isinstance(x, (str, int, float)):
                    out.append(str(x))
            return out
        if isinstance(val, (str, int, float)):
            return [str(val)]
        return []

    def make_fallback(justification: str) -> dict[str, object]:
        return {
            "sentiment": "neutral",
            "justification": trim_words(justification, 50),
            "related_industry_naics": [],
            "related_company_names": [],
        }

    if not isinstance(text, str) or not text.strip():
        return make_fallback("Input text missing or empty.")

    safe_text = text.replace("{", "{{").replace("}", "}}").strip()
    prompt_content = sentiment_prompt.replace("{text}", safe_text)

    messages = [
        {"role": "user", "content": prompt_content},
    ]
    p = llm_pipe or pipe
    try:
        outputs = p(messages, max_new_tokens=256)
    except Exception as exc:
        return make_fallback(f"Pipeline failure: {exc}")

    content = ""
    try:
        gen = outputs[0].get("generated_text", "")
        if isinstance(gen, list):
            last = gen[-1]
            if isinstance(last, dict) and "content" in last:
                content = (last["content"] or "").strip()
            else:
                content = (str(last) or "").strip()
        elif isinstance(gen, dict) and "content" in gen:
            content = (gen["content"] or "").strip()
        elif isinstance(gen, str):
            content = gen.strip()
        else:
            content = (str(gen) or "").strip()
    except Exception as exc:
        return make_fallback(f"Malformed pipeline output: {exc}")

    def parse_json_payload(s: str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    parsed = parse_json_payload(content)
    if not isinstance(parsed, dict):
        return make_fallback(content or "Model returned non-JSON output.")

    sentiment = normalize_sentiment(parsed.get("sentiment", "neutral"))
    if sentiment not in allowed:
        sentiment = "neutral"

    justification = trim_words(parsed.get("justification", ""))
    related_industry_naics = to_string_list(parsed.get("related_industry_naics"))
    related_company_names = to_string_list(parsed.get("related_company_names"))

    return {
        "sentiment": sentiment,
        "justification": justification,
        "related_industry_naics": related_industry_naics,
        "related_company_names": related_company_names,
    }

# %%
FILTER_LANG = 'ko'
FILTER_SRC = '3PTY'

# %%
for parquet_path in parquet_paths:
    parquet_file = parquet_path.name
    print(f"Processing {parquet_file} ...")

    outfilename = f"{parquet_file[:-8]}_{FILTER_SRC}_{FILTER_LANG}.sentiment.parquet"
    outfile = OUTPUT_DIR / outfilename

    if os.path.exists(outfile):
        print(f"Skip {parquet_file} - already existing")
        continue

    # Read
    dfnews = pd.read_parquet(parquet_path)
    print(f"{parquet_file} original shape: {dfnews.shape}")

    # Basic id
    dfnews['ids'] = dfnews.index.astype(str)

    # Filter
    dfnews = dfnews[(dfnews.get('lang_code') == FILTER_LANG) & (dfnews.get('src') == FILTER_SRC)]
    print(f"{parquet_file} after filter ({FILTER_LANG},{FILTER_SRC}): {dfnews.shape}")

    # Ensure required columns exist
    required_columns = {"title", "content"}
    missing = required_columns - set(dfnews.columns)
    if missing:
        print(f"Skipping {parquet_file}: missing required columns {missing}")
        continue

    # Drop NA records for title/content and build text without apply
    dfnews = dfnews.dropna(subset=['title', 'content'])
    print(f"{parquet_file} after dropna(title,content): {dfnews.shape}")

    # 1% sampling (if empty after drop, this will stay empty)
    if not dfnews.empty:
        dfnews = dfnews.sample(frac=0.01, random_state=42)
    print(f"{parquet_file} after 1% sample: {dfnews.shape}")

    if dfnews.empty:
        outfile = OUTPUT_DIR / f"{parquet_file[:-8]}_{FILTER_SRC}_{FILTER_LANG}.sentiment.parquet"
        pd.DataFrame([]).to_parquet(outfile)
        print(f"Wrote empty output {outfile}")
        continue

    # Build text column safely (vectorized)
    dfnews['title'] = dfnews['title'].astype(str)
    dfnews['content'] = dfnews['content'].astype(str)
    dfnews['text'] = dfnews['title'] + '\n\n' + dfnews['content']

    # Validate required columns for downstream
    required_columns_for_loop = {"text", "title"}
    missing2 = required_columns_for_loop - set(dfnews.columns)
    if missing2:
        print(f"Skipping {parquet_file}: missing required columns {missing2}")
        continue

    # Up to 5 samples max (preserve original behavior)
    sample_size = max(len(dfnews), 5)
    print(f"{parquet_file} sample_size: {sample_size}")

    sentiment_samples: list[dict] = []
    if sample_size > 0:
        sample_df = dfnews.sample(n=sample_size, random_state=0)

        def _to_str(x) -> str:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return ""
            return str(x)

        max_chars = 5000

        rows = list(sample_df.itertuples(index=True))
        num_rows = len(rows)

        def process_row(row, llm_pipe):
            try:
                title = _to_str(getattr(row, "title", "")).strip()
                body = _to_str(getattr(row, "text", "")).strip()

                combined_text = (title + "\n\n" + body).strip() if (title or body) else ""
                if len(combined_text) > max_chars:
                    combined_text = combined_text[:max_chars]

                result = analyse_sentiment(combined_text, llm_pipe=llm_pipe) if combined_text else {
                    "sentiment": "neutral",
                    "justification": "No content provided.",
                    "related_industry_naics": [],
                    "related_company_names": [],
                }

                return {
                    "index": row.Index,
                    "headline": title,
                    "sentiment": result.get("sentiment", "neutral"),
                    "justification": result.get("justification", ""),
                    "related_industry_naics": result.get("related_industry_naics", []),
                    "related_company_names": result.get("related_company_names", []),
                }
            except Exception as exc:
                return {
                    "index": row.Index,
                    "headline": _to_str(getattr(row, "title", "")).strip(),
                    "sentiment": "neutral",
                    "justification": f"Processing error: {exc}",
                    "related_industry_naics": [],
                    "related_company_names": [],
                }

        # Two-GPU concurrent processing, round-robin
        pipes = [pipe0, pipe1]
        results_buffer = [None] * num_rows
        with ThreadPoolExecutor(max_workers=2) as executor, tqdm(total=num_rows, desc=f"{parquet_file}") as pbar:
            futures = {}
            for i, row in enumerate(rows):
                assigned_pipe = pipes[i % 2]
                fut = executor.submit(process_row, row, assigned_pipe)
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                results_buffer[i] = fut.result()
                pbar.update(1)

        sentiment_samples = [res for res in results_buffer if res is not None]

    # Write per-file output
    
    pd.DataFrame(sentiment_samples).to_parquet(outfile)
    print(f"Wrote {outfile}")