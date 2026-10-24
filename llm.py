"""
llm.py — the ONLY file that knows which AI provider is used (currently Groq free tier).
Swap providers by editing this file alone.
"""

import json
import os
from groq import Groq

from charts import TEMPLATE_HELP

# Tried in order; if one is retired/unavailable the next is used.
MODELS = [m for m in [os.environ.get("GROQ_MODEL"), "openai/gpt-oss-20b"] if m]

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key or key == "your_key_here":
            raise RuntimeError("GROQ_API_KEY is missing. Add your free key to the .env file (console.groq.com/keys).")
        _client = Groq(api_key=key)
    return _client


LANGUAGE_INSTRUCTIONS = {
    "english": "Write in clear, simple English.",
    "hindi": "Write in simple Hindi (Devanagari script). Keep well-known technical terms (e.g. likelihood, sample) in English inside brackets when helpful.",
    "hinglish": "Write in natural Hinglish: Hindi conversational tone in Roman script mixed with English technical terms, the way Indian students actually talk. Not pure Hindi, not pure English.",
}

LEVEL_INSTRUCTIONS = {
    "basic": "Depth: BSc level. Intuition first, minimal proofs, very simple numbers.",
    "advanced": "Depth: MSc level. Include formal definitions, theorems/conditions, and a short derivation or proof idea where relevant.",
}

PROMPT = """You are an expert statistics teacher. A student wants to truly understand a topic AND be able to write a full-marks exam answer AND know how it is used in real surveys/industry.

{lang}
{level}

TOPIC: {topic}

REFERENCE MATERIAL from the student's curated books (primary, authoritative source; use it first.
If it does not fully cover the topic you may add general knowledge, but say so):
{context}

FORMAT RULES (very important):
- Use EXACTLY these seven markdown headings, in this order, each starting with "## ":
  ## Concept in Simple Words
  ## Small Example
  ## Real-World Case Study
  ## Best Graph or Chart
  ## Advantages
  ## Disadvantages and Limitations
  ## Exam Tip
- Write ALL mathematics in LaTeX: inline as $...$ and displayed equations as $$...$$ on their own lines. Never use \\( \\) or \\[ \\]. Never use the $ sign for money (write "Rs." or "rupees").
- Use short paragraphs and "- " bullet lists. Use **bold** for key terms.
- Small Example: fully worked numeric example the student can follow by hand.
- Real-World Case Study: numbered steps exactly as an investigator / statistical officer / analyst would carry it out.
- Exam Tip: what an examiner expects (definition, formula, steps, conclusion) so no marks are lost.
- Best Graph or Chart: name the chart type(s) and say why. The app will draw the real chart itself, so do NOT draw ASCII art.
- Just before the SOURCE line write exactly ONE line of the form
  CHART: {"type":"<template>","params":{...}}
  choosing the single template below that best illustrates THIS topic (params optional; use {} if unsure).
  Only add a second CHART line if a second, different visual truly helps.
  Templates: {TEMPLATES}
- The very last line must be exactly one of:
  SOURCE: reference material
  SOURCE: general knowledge
  SOURCE: mixed
"""


def build_prompt(topic, chunks, language, level):
    if chunks:
        ctx = "\n\n---\n\n".join(chunks)
    else:
        ctx = "(No matching reference material was found — use general statistical knowledge and end with SOURCE: general knowledge.)"
    return (PROMPT
            .replace("{lang}", LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["english"]))
            .replace("{level}", LEVEL_INSTRUCTIONS.get(level, LEVEL_INSTRUCTIONS["basic"]))
            .replace("{topic}", topic)
            .replace("{TEMPLATES}", TEMPLATE_HELP)
            .replace("{context}", ctx))


def explain_topic(topic, chunks, language="english", level="basic") -> str:
    client = _get_client()
    prompt = build_prompt(topic, chunks, language, level)
    last_err = None
    for model in MODELS:
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.35,
                max_tokens=3000,
            )
            return r.choices[0].message.content
        except Exception as e:  # try next model
            last_err = e
    raise last_err


# ═══════════════════════ Industry Analysis Engine: plan selection + interpretation ═══════════════════════

from stats_engine import ANALYSIS_HELP

PLAN_PROMPT = """You are a senior statistical analyst. A user has uploaded a dataset and stated an objective.
Choose exactly ONE analysis from the list below that best matches the objective, and the exact column
names it needs (copy column names EXACTLY as given in the profile — do not invent or rename columns).

DATASET PROFILE (row/column summary, not the raw data):
{profile}

USER'S OBJECTIVE:
{objective}

AVAILABLE ANALYSES (name{{required params}}):
{analyses}

Reply with ONLY a JSON object, nothing else, no markdown fences:
{{"analysis": "<name>", "params": {{...}}, "reasoning": "<one sentence: why this analysis fits the objective>"}}
If nothing else fits, use {{"analysis": "describe", "params": {{}}, "reasoning": "..."}}.
"""


def choose_analysis(profile: dict, objective: str) -> dict:
    client = _get_client()
    prompt = PLAN_PROMPT.replace("{profile}", json.dumps(profile)[:6000]) \
                         .replace("{objective}", objective) \
                         .replace("{analyses}", ANALYSIS_HELP)
    last_err = None
    for model in MODELS:
        try:
            r = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=400,
                response_format={"type": "json_object"},
            )
            data = json.loads(r.choices[0].message.content)
            if "analysis" in data:
                data.setdefault("params", {})
                return data
        except Exception as e:
            last_err = e
    return {"analysis": "describe", "params": {}, "reasoning": f"Falling back to a dataset overview ({last_err})."}


INTERPRET_PROMPT = """You are a senior statistical analyst writing the narrative for a report that a survey
officer, manager, or student will read. All numbers below are ALREADY CORRECTLY CALCULATED by code —
do not recompute or alter any number; just explain what they mean in context.

{lang}
Target audience: industry / applied — be precise and decision-oriented, but still clear. This can be read
by a student too, so briefly define any technical term the first time it appears.

OBJECTIVE THE USER STATED:
{objective}

ANALYSIS PERFORMED: {analysis_title}

COMPUTED RESULT (JSON, already correct):
{result_json}

RELIABILITY ASSESSMENT (already computed):
{reliability_json}

Write the report using EXACTLY these four markdown headings, in this order, each starting with "## ":
## What We Did
## What the Numbers Show
## Conclusion for the Objective
## Caveats and Reliability

Rules:
- "What We Did": 2-4 sentences, plain language, name the test/method and why it fits the objective.
- "What the Numbers Show": interpret the statistic, p-value/confidence interval, and effect size from the JSON above in plain words. Use $...$ for any inline math and $$...$$ for standalone equations — never \\( \\) or \\[ \\].
- "Conclusion for the Objective": answer the user's actual objective directly, in decision-oriented language (e.g. "Yes, there is a statistically significant difference...").
- "Caveats and Reliability": mention the reliability level and the 1-2 most important factors from the reliability JSON, in plain words. Never claim near-100% accuracy; reliability is about the assumptions and sample, not a blanket accuracy number.
- Do not invent any number not present in the JSON above.
"""


def interpret_result(objective: str, analysis_title: str, result: dict, reliability_obj: dict, language="english") -> str:
    client = _get_client()
    prompt = (INTERPRET_PROMPT
              .replace("{lang}", LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["english"]))
              .replace("{objective}", objective)
              .replace("{analysis_title}", analysis_title)
              .replace("{result_json}", json.dumps(result)[:4000])
              .replace("{reliability_json}", json.dumps(reliability_obj)[:2000]))
    last_err = None
    for model in MODELS:
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                temperature=0.3, max_tokens=1400)
            return r.choices[0].message.content
        except Exception as e:
            last_err = e
    raise last_err
