#!/usr/bin/env python3

import argparse
import os
import subprocess
import requests
import json
import shutil
import re
from urllib.parse import urlparse

from openai import OpenAI
import anthropic

# -------------------------
# API KEY SELECTION
# -------------------------
def get_api_key():
    if os.getenv("GEMINI_API_KEY"):
        return ("gemini", os.getenv("GEMINI_API_KEY"))
    elif os.getenv("ANTHROPIC_API_KEY"):
        return ("anthropic", os.getenv("ANTHROPIC_API_KEY"))
    elif os.getenv("OPENAI_API_KEY"):
        return ("openai", os.getenv("OPENAI_API_KEY"))
    elif os.getenv("HUGGINGFACE_API_KEY"):
        return ("huggingface", os.getenv("HUGGINGFACE_API_KEY"))
    else:
        raise ValueError(
            "No API key found. Set one of: GEMINI_API_KEY, ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, or HUGGINGFACE_API_KEY."
        )

# -------------------------
# HEADER FETCH
# -------------------------
def get_headers(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return dict(response.headers)
    except requests.RequestException as e:
        print(f"Error fetching headers: {e}")
        return {}

# -------------------------
# AI EXTENSION GENERATION
# -------------------------
def get_ai_extensions(url, headers, api_type, api_key, max_extensions):
    prompt = f"""
Given the following URL and HTTP headers, suggest the most likely file extensions
for fuzzing this endpoint.

Respond with valid JSON only, no commentary.
Format:
{{"extensions": [".php", ".json", ".bak"]}}

Limit to at most {max_extensions} extensions.

URL: {url}
Headers: {headers}
"""

    system_msg = (
        "You are a security assistant helping with web fuzzing. "
        "Return only valid JSON. No prose."
    )

    # ---------- OpenAI ----------
    if api_type == "openai":
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

    # ---------- Anthropic ----------
    elif api_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            temperature=0,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

    # ---------- Gemini ----------
    elif api_type == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-pro",
            generation_config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

        resp = model.generate_content(
            f"SYSTEM:\n{system_msg}\n\nUSER:\n{prompt}"
        )
        raw = resp.text.strip()

    # ---------- Local HuggingFace (Qwen) ----------
    elif api_type == "huggingface":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print("🤖 Loading local Qwen model…")

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

        output = output[:, inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    else:
        raise ValueError("Unsupported API type")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("❌ AI returned invalid JSON:")
        print(raw)
        raise

# -------------------------
# FFUF OUTPUT COLORIZER
# -------------------------
def colorize_ffuf_output(line):
    line = line.decode(errors="ignore").strip()
    m = re.search(r"\[Status: (\d{3})", line)
    if m:
        code = int(m.group(1))
        if code == 200:
            color = "\033[92m"
        elif code in (301, 302):
            color = "\033[94m"
        elif code == 403:
            color = "\033[91m"
        else:
            color = "\033[0m"
        print(f"{color}{line}\033[0m")
    else:
        print(line)

# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ffufai – AI-powered ffuf wrapper (OpenAI / Claude / Gemini / Local)"
    )
    parser.add_argument("--ffuf-path", default="ffuf")
    parser.add_argument("--max-extensions", type=int, default=5)
    args, unknown = parser.parse_known_args()

    try:
        url = unknown[unknown.index("-u") + 1]
    except (ValueError, IndexError):
        print("❌ You must provide -u URL")
        return

    parsed = urlparse(url)
    if "FUZZ" not in parsed.path:
        print("⚠️ FUZZ keyword not found in URL path")

    base_url = url.replace("FUZZ", "")
    headers = get_headers(base_url)

    api_type, api_key = get_api_key()
    print(f"🔌 Using AI backend: {api_type}")

    try:
        ext_data = get_ai_extensions(
            url, headers, api_type, api_key, args.max_extensions
        )
        exts = ext_data.get("extensions", [])
        if not exts:
            raise ValueError("Empty extension list")
    except Exception as e:
        print(f"⚠️ AI failed, using fallback extensions: {e}")
        exts = [".php", ".html", ".json", ".bak"]

    exts = exts[: args.max_extensions]
    exts = ",".join(e if e.startswith(".") else f".{e}" for e in exts)

    ffuf_cmd = [args.ffuf_path] + unknown + ["-e", exts]

    if not shutil.which(args.ffuf_path):
        print("❌ ffuf binary not found")
        return

    print("▶️ Running:", " ".join(ffuf_cmd))
    with subprocess.Popen(
        ffuf_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as proc:
        for line in proc.stdout:
            colorize_ffuf_output(line)

if __name__ == "__main__":
    main()
