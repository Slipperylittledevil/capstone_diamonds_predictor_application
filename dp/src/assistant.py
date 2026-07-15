"""
assistant.py — a natural-language assistant for the diamonds dataset.

ARCHITECTURE — and why it is built this way
-------------------------------------------
The naive approach is: hand the question to an LLM, have it write pandas code, exec()
the result. Do not do this. exec()-ing model-generated code against your data is an
arbitrary-code-execution hole, and it fails silently in weird ways.

This assistant uses the safe pattern instead — three separate stages:

    1. UNDERSTAND   LLM converts the question into a STRUCTURED FILTER (JSON).
                    The LLM's only job is translation. It never touches the data.
    2. RETRIEVE     Our own code applies that filter with pandas, validating every
                    column name and operator against a whitelist. Deterministic,
                    auditable, and impossible to weaponise.
    3. EXPLAIN      LLM turns the numbers we computed into plain English.
                    It explains real numbers; it never invents them.

The LLM sits at the edges — language in, language out. The middle is ordinary,
inspectable Python. That is the architecture you want any time an LLM is near real data.

Runs WITHOUT an OpenAI key: a keyword parser handles stage 1 and a template handles
stage 3. Degraded, but functional and free.
"""
import json
import os
import re

import numpy as np
import pandas as pd

from src.config import CLARITY_ORDER, CUT_ORDER, COLOR_ORDER

# ---------------------------------------------------------------------------
# [WHITELISTS] — nothing outside these lists can ever reach the dataframe.
# This is the security boundary. An LLM asking to filter on a column that does
# not exist, or with an operator we don't allow, is simply refused.
# ---------------------------------------------------------------------------
ALLOWED_COLUMNS = {
    "carat", "price", "depth", "table", "x", "y", "z", "volume",
    "cut", "color", "clarity", "cut_rank", "color_rank", "clarity_rank",
}
ALLOWED_OPS = {">", ">=", "<", "<=", "==", "!=", "in", "between"}

SCHEMA_PROMPT = f"""You convert questions about a diamonds dataset into JSON filters.

Columns:
- carat (float, 0.2-5.0), price (int USD, 326-18823), depth (float %), table (float %)
- x, y, z (float mm), volume (float mm3)
- cut (categorical): {CUT_ORDER}          (worst -> best)
- color (categorical): {COLOR_ORDER}      (worst -> best)
- clarity (categorical): {CLARITY_ORDER}  (worst -> best)

Respond with ONLY a JSON object, no prose, no markdown fences:
{{
  "filters": [{{"column": "carat", "op": ">=", "value": 1.0}}],
  "sort_by": "price",
  "ascending": true,
  "limit": 5,
  "aggregate": "mean"
}}

"aggregate" is one of: none, mean, median, count, min, max.
Omit any key that does not apply. "op" is one of: >, >=, <, <=, ==, !=, in, between.
For "between", value is a two-element list. For "in", value is a list.
"""


class DiamondAssistant:
    def __init__(self, df: pd.DataFrame, model: str = "gpt-4o-mini",
                 api_key: str | None = None):
        self.df = df
        self.model = model
        self.client = None

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=key)
            except Exception as e:
                print(f"[assistant] OpenAI unavailable ({e}) — using offline fallback.")

    # =======================================================================
    # STAGE 1 — UNDERSTAND: question -> structured filter
    # =======================================================================
    def _parse_llm(self, question: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,                       # [determinism: same question, same filter]
            messages=[
                {"role": "system", "content": SCHEMA_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)

    def _parse_fallback(self, question: str) -> dict:
        """[No API key? Keyword matching. Crude, but it covers the common shapes.]"""
        q = question.lower()
        filters = []

        m = re.search(r"(?:under|below|less than|cheaper than)\s*\$?([\d,]+)", q)
        if m:
            filters.append({"column": "price", "op": "<", "value": float(m.group(1).replace(",", ""))})
        m = re.search(r"(?:over|above|more than|at least)\s*\$?([\d,]+)", q)
        if m:
            filters.append({"column": "price", "op": ">", "value": float(m.group(1).replace(",", ""))})
        m = re.search(r"([\d.]+)\s*carat", q)
        if m:
            c = float(m.group(1))
            filters.append({"column": "carat", "op": "between", "value": [c - 0.05, c + 0.05]})

        for grade in CLARITY_ORDER:
            if re.search(rf"\b{grade.lower()}\b", q):
                filters.append({"column": "clarity", "op": "==", "value": grade}); break
        for grade in CUT_ORDER:
            if grade.lower() in q:
                filters.append({"column": "cut", "op": "==", "value": grade}); break

        agg = "none"
        if any(w in q for w in ["average", "mean", "typical"]): agg = "mean"
        elif "median" in q:                                     agg = "median"
        elif any(w in q for w in ["how many", "count", "number of"]): agg = "count"
        elif any(w in q for w in ["cheapest", "lowest", "minimum"]):  agg = "min"
        elif any(w in q for w in ["most expensive", "highest", "maximum", "priciest"]): agg = "max"

        return {"filters": filters, "aggregate": agg, "limit": 5,
                "sort_by": "price", "ascending": "cheap" in q or "lowest" in q}

    def understand(self, question: str) -> dict:
        spec = self._parse_llm(question) if self.client else self._parse_fallback(question)
        return self._validate(spec)

    # =======================================================================
    # [VALIDATION] — the security boundary. Nothing gets past this unchecked.
    # =======================================================================
    def _validate(self, spec: dict) -> dict:
        clean = []
        for f in spec.get("filters", []):
            col, op = f.get("column"), f.get("op")
            if col not in ALLOWED_COLUMNS:
                print(f"[assistant] rejected unknown column: {col!r}");  continue
            if op not in ALLOWED_OPS:
                print(f"[assistant] rejected unknown operator: {op!r}"); continue
            clean.append(f)

        agg = spec.get("aggregate", "none")
        if agg not in {"none", "mean", "median", "count", "min", "max"}:
            agg = "none"

        sort_by = spec.get("sort_by")
        if sort_by not in ALLOWED_COLUMNS:
            sort_by = None

        return {
            "filters": clean,
            "aggregate": agg,
            "sort_by": sort_by,
            "ascending": bool(spec.get("ascending", True)),
            "limit": int(min(max(spec.get("limit", 5), 1), 20)),
        }

    # =======================================================================
    # STAGE 2 — RETRIEVE: apply the filter with plain, boring pandas
    # =======================================================================
    def retrieve(self, spec: dict):
        d = self.df
        for f in spec["filters"]:
            c, op, v = f["column"], f["op"], f["value"]
            if   op == ">":       d = d[d[c] >  v]
            elif op == ">=":      d = d[d[c] >= v]
            elif op == "<":       d = d[d[c] <  v]
            elif op == "<=":      d = d[d[c] <= v]
            elif op == "==":      d = d[d[c] == v]
            elif op == "!=":      d = d[d[c] != v]
            elif op == "in":      d = d[d[c].isin(v)]
            elif op == "between": d = d[d[c].between(v[0], v[1])]

        facts = {"matches": int(len(d))}
        if len(d):
            facts["price_mean"]   = round(float(d["price"].mean()), 2)
            facts["price_median"] = round(float(d["price"].median()), 2)
            facts["price_min"]    = int(d["price"].min())
            facts["price_max"]    = int(d["price"].max())
            facts["carat_mean"]   = round(float(d["carat"].mean()), 3)
            facts["top_cut"]      = d["cut"].mode().iat[0]
            facts["top_clarity"]  = d["clarity"].mode().iat[0]
            facts["top_color"]    = d["color"].mode().iat[0]

        if spec["sort_by"] and len(d):
            d = d.sort_values(spec["sort_by"], ascending=spec["ascending"])

        cols = ["carat", "cut", "color", "clarity", "depth", "table", "price"]
        sample = d[cols].head(spec["limit"]) if len(d) else d
        return facts, sample

    # =======================================================================
    # STAGE 3 — EXPLAIN: numbers -> plain English
    # =======================================================================
    def explain(self, question, spec, facts, sample) -> str:
        if facts["matches"] == 0:
            return ("No diamonds in the dataset match that description. "
                    "Try relaxing one of the constraints — a wider carat range or a "
                    "higher price ceiling usually opens things up.")

        if self.client:
            # [The LLM is handed ONLY facts we computed. It cannot invent a number,
            #  because it is never asked to produce one — just to narrate ours.]
            payload = {
                "question": question,
                "filter_applied": spec["filters"],
                "computed_facts": facts,
                "example_rows": sample.to_dict(orient="records"),
            }
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content":
                        "You are a plain-spoken diamond expert. Answer the question using ONLY "
                        "the numbers in computed_facts and example_rows. Never invent a figure. "
                        "Two or three short paragraphs. Explain what the numbers mean for a "
                        "buyer, in ordinary language. No hype, no sales pressure."},
                    {"role": "user", "content": json.dumps(payload, default=str)},
                ],
            )
            return resp.choices[0].message.content.strip()

        # [Offline template fallback]
        return (
            f"{facts['matches']:,} diamonds match.\n\n"
            f"They average {facts['carat_mean']} carat and ${facts['price_mean']:,.0f}, "
            f"with a median of ${facts['price_median']:,.0f} — the gap between mean and "
            f"median tells you how skewed the price range is. The spread runs from "
            f"${facts['price_min']:,} to ${facts['price_max']:,}.\n\n"
            f"The most common grades in this group are {facts['top_cut']} cut, "
            f"colour {facts['top_color']}, clarity {facts['top_clarity']}."
        )

    # =======================================================================
    # The whole pipeline
    # =======================================================================
    def ask(self, question: str, show_work: bool = True) -> str:
        spec = self.understand(question)
        facts, sample = self.retrieve(spec)

        if show_work:
            print(f"Q: {question}")
            print(f"  [filter]  {spec['filters']}")
            print(f"  [matches] {facts['matches']:,}")
            if len(sample):
                print(sample.to_string(index=False))
            print()

        answer = self.explain(question, spec, facts, sample)
        print(answer)
        print("-" * 70)
        return answer
