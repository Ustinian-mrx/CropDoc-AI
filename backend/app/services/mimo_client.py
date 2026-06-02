import json
import os

import httpx
from dotenv import load_dotenv


load_dotenv()


MIMO_API_KEY = os.getenv("MIMO_API_KEY")
MIMO_API_BASE_URL = os.getenv("MIMO_API_BASE_URL")
MIMO_MODEL = os.getenv("MIMO_MODEL")


def is_mimo_configured() -> bool:
    return bool(MIMO_API_KEY and MIMO_API_BASE_URL and MIMO_MODEL)

def print_mimo_config_status():
    print("[MIMO] configured:", is_mimo_configured())
    print("[MIMO] base url:", MIMO_API_BASE_URL)
    print("[MIMO] model:", MIMO_MODEL)
    print("[MIMO] api key exists:", bool(MIMO_API_KEY))


def build_prompt(prediction: dict) -> str:
    return f"""
你是一个农业病虫害诊断助手。

请根据以下 AI 图像识别结果，生成结构化防治建议。

作物：{prediction["crop"]}
识别结果：{prediction["disease"]}
是否健康：{prediction["is_healthy"]}
置信度：{prediction["confidence"]:.4f}

要求：
1. 输出必须是 JSON
2. 不要输出 Markdown
3. 不要添加 JSON 以外的解释
4. 内容要谨慎，不能替代农技专家诊断
5. 输出内容必须是中文
6. 字段必须包含：
   title: string
   risk_level: low / medium / high / uncertain
   symptoms: string[]
   actions: string[]
   prevention: string[]
   disclaimer: string
"""


def generate_advice_with_mimo(prediction: dict) -> dict | None:
    if not is_mimo_configured():
        return None

    prompt = build_prompt(prediction)

    headers = {
        "Authorization": f"Bearer {MIMO_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MIMO_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是一个谨慎、专业的农业病虫害防治建议助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3,
    }

    try:
        response = httpx.post(
            MIMO_API_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        content = data["choices"][0]["message"]["content"]

        return json.loads(content)

    except Exception as exc:
        print(f"[MIMO] request failed: {exc}")
        return None
