import os
import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="HanPath AI Grader")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ganti ke domain spesifik saat production
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

class GradeRequest(BaseModel):
    original: str
    summary: str
    level: str = "hsk6"

class SentenceFix(BaseModel):
    original: str
    suggestion: str
    reason: str
    error_tag: str

class GradeResponse(BaseModel):
    overall_score: int
    content_score: int
    conciseness_score: int
    language_score: int
    structure_score: int
    strengths: list[str]
    weaknesses: list[str]
    sentence_fixes: list[SentenceFix]
    error_tags: list[str]
    target_length: int
    actual_length: int
    ratio_percent: float

SYSTEM_PROMPT = """你是HSK 6写作考试的批改老师，专门批改"缩写"（摘要写作）题型。

评分标准：
- 内容完整性 (25%): 是否保留了原文核心观点
- 简洁性 (25%): 是否在400字左右（约原文的40%）
- 语言准确性 (25%): 语法、词汇
- 结构连贯性 (25%): 段落、句子之间的逻辑

严重扣分：
- 加入原文没有的个人观点
- 遗漏原文主要论点
- 逐字抄写原文
- 语法错误超过3处

只输出JSON，不要markdown包裹，不要解释。
{
  "overall_score": <0-100>,
  "content_score": <0-100>,
  "conciseness_score": <0-100>,
  "language_score": <0-100>,
  "structure_score": <0-100>,
  "strengths": ["中文，每条一句话"],
  "weaknesses": ["中文，每条一句话"],
  "sentence_fixes": [
    {"original":"...","suggestion":"...","reason":"...","error_tag":"..."}
  ],
  "error_tags": ["..."]
}

error_tag 只能取以下值：
missing_main_point, added_opinion, copied_too_much, de_DE_dei,
measure_word, word_order, complement, ba_construction,
bei_construction, separable_verb, collocation, near_synonym,
register, conjunction, redundant, other_grammar"""

ALLOWED_TAGS = {
    "missing_main_point","added_opinion","copied_too_much","de_DE_dei",
    "measure_word","word_order","complement","ba_construction",
    "bei_construction","separable_verb","collocation","near_synonym",
    "register","conjunction","redundant","other_grammar"
}

def count_han(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))

@app.post("/api/grade-summary", response_model=GradeResponse)
async def grade_summary(req: GradeRequest):
    if not LLM_API_KEY:
        raise HTTPException(500, "LLM_API_KEY not configured")

    src_len = count_han(req.original)
    out_len = count_han(req.summary)
    if src_len < 50:
        raise HTTPException(400, "Teks asli terlalu pendek (min 50 Hanzi)")
    if out_len < 20:
        raise HTTPException(400, "Ringkasan terlalu pendek (min 20 Hanzi)")

    target_len = round(src_len * 0.4)
    ratio = round(out_len / src_len * 100, 1)

    user_msg = f"""原文（{src_len}字）：
{req.original}

学生的缩写（{out_len}字，目标约{target_len}字，压缩率{ratio}%）：
{req.summary}

按HSK 6缩写标准批改，只输出JSON。"""

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        data["target_length"] = target_len
        data["actual_length"] = out_len
        data["ratio_percent"] = ratio
        data["error_tags"] = [t for t in data.get("error_tags", []) if t in ALLOWED_TAGS]
        for fix in data.get("sentence_fixes", []):
            if fix.get("error_tag") not in ALLOWED_TAGS:
                fix["error_tag"] = "other_grammar"
        return GradeResponse(**data)
    except json.JSONDecodeError as e:
        raise HTTPException(502, f"LLM returned invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(500, f"Grading failed: {e}")

@app.get("/api/health")
def health():
    return {"status":"ok","model":LLM_MODEL}

@app.get("/")
def root():
    return {"service":"HanPath AI Grader","docs":"/docs"}
