# api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from webhook_sender import send_to_n8n
from memory_engine import search_similar, store_reflections
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

app = FastAPI()

# Allow React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Store conversation history per session
conversations = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

class EmailRequest(BaseModel):
    session_id: str
    name: str
    email: str

@app.post("/chat")
async def chat(req: ChatRequest):
    sid = req.session_id

    if sid not in conversations:
        conversations[sid] = {
            "messages": [],
            "data": {}
        }

    session = conversations[sid]
    session["messages"].append({"role": "user", "content": req.message})

    # System prompt — EchoSense conversational agent
    system = """
You are EchoSense AI — a warm, witty emotional check-in assistant for students.
Your job is to have a short natural conversation to understand how the student is feeling.

In the conversation, naturally collect:
- Their current mood (1-10)
- Hours of sleep last night
- Study load today (Low/Medium/High)
- Stress level (1-10)
- A brief reflection on their day

Ask ONE question at a time. Be friendly, slightly funny, never clinical.
Once you have all 5 pieces of info, end your message with exactly:
[READY_TO_SEND]

Do NOT say you're an AI. Do NOT give medical advice.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}] + session["messages"],
        max_tokens=300,
        temperature=0.85
    )

    reply = response.choices[0].message.content.strip()
    session["messages"].append({"role": "assistant", "content": reply})

    ready = "[READY_TO_SEND]" in reply
    clean_reply = reply.replace("[READY_TO_SEND]", "").replace("READY_TO_SEND", "").strip()

    return {
        "reply": clean_reply,
        "ready_to_send": ready
    }


@app.post("/send-email")
async def send_email(req: EmailRequest):
    sid = req.session_id
    session = conversations.get(sid, {})
    messages = session.get("messages", [])

    # Ask LLM to extract structured data from conversation
    extraction_prompt = f"""
Extract the following from this conversation as JSON only (no explanation):
{{
  "mood": <number 1-10>,
  "sleep_hours": <number>,
  "study_load": <"Low" or "Medium" or "High">,
  "stress_self": <number 1-10>,
  "reflection": "<their reflection text>"
}}

Conversation:
{chr(10).join([m['role'] + ': ' + m['content'] for m in messages])}

Return ONLY the JSON object, nothing else.
"""

    extraction = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": extraction_prompt}],
        max_tokens=200,
        temperature=0
    )

    import json
    raw = extraction.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    data["name"] = req.name
    data["email"] = req.email
    data["stress_index"] = round(
        (10 - data["mood"]) * 2 +
        max(0, 8 - data["sleep_hours"]) * 1.5 +
        {"Low": 1, "Medium": 2, "High": 3}.get(data["study_load"], 2) * 3 +
        (10 - data["stress_self"]) * 1.0,
        2
    )

    # Emotion detection based on mood + stress
    mood = float(data.get("mood", 5))
    stress = float(data.get("stress_self", 5))

    if mood >= 7 and stress <= 4:
        data["dominant_emotion"] = "joy"
    elif mood >= 5 and stress <= 6:
        data["dominant_emotion"] = "neutral"
    elif stress >= 8:
        data["dominant_emotion"] = "high_stress"
    elif mood <= 3:
        data["dominant_emotion"] = "sadness"
    else:
        data["dominant_emotion"] = "fear"

    # RAG — find similar reflections
    similar = search_similar(data["reflection"], n_results=3)
    similar_texts = "\n".join(
        [f'- "{s["text"]}"' for s in similar]
    ) if similar else "No similar reflections yet."

    # Generate personalized email
    insight_prompt = f"""
You are EchoSense AI — witty, warm, slightly unhinged student companion.
Write a SHORT personalized message (4-6 sentences) for {req.name}.
Be funny, warm, reference their data naturally.
Do NOT start with "Hey {req.name}". No medical advice. No "I'm an AI".

Data: Mood {data['mood']}/10, Sleep {data['sleep_hours']}hrs, 
Stress {data['stress_self']}/10, Load {data['study_load']}
Reflection: "{data['reflection']}"

Similar community reflections (RAG context):
{similar_texts}
"""

    insight_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": insight_prompt}],
        max_tokens=300,
        temperature=0.9
    )

    insight = insight_response.choices[0].message.content.strip()

    # Build HTML email
    mood_emoji = "🌟" if data["mood"] >= 8 else "😊" if data["mood"] >= 5 else "💙"

    html_body = f"""
<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
body{{margin:0;padding:0;background:#f4f4f8;font-family:'Segoe UI',Arial,sans-serif;}}
.wrapper{{max-width:580px;margin:30px auto;background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08);}}
.header{{background:linear-gradient(135deg,#6C63FF 0%,#3B82F6 100%);padding:36px 32px;text-align:center;}}
.header h1{{color:white;margin:0;font-size:22px;font-weight:700;}}
.header p{{color:rgba(255,255,255,0.85);margin:6px 0 0;font-size:14px;}}
.body{{padding:32px;}}
.greeting{{font-size:20px;font-weight:600;color:#1a1a2e;margin:0 0 16px;}}
.insight{{font-size:15px;line-height:1.75;color:#444;background:#f8f7ff;border-left:4px solid #6C63FF;border-radius:0 12px 12px 0;padding:18px 20px;margin:0 0 28px;}}
.stats-title{{font-size:12px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:0.08em;margin:0 0 12px;}}
.footer{{background:#f8f7ff;padding:20px 32px;text-align:center;border-top:1px solid #eee;}}
.footer p{{font-size:12px;color:#aaa;margin:4px 0;}}
.brand{{font-weight:700;color:#6C63FF;}}
</style></head><body>
<div class="wrapper">
  <div class="header"><h1>🧠 EchoSense AI</h1><p>Your personal emotional check-in</p></div>
  <div class="body">
    <p class="greeting">Hey {req.name} {mood_emoji}</p>
    <div class="insight">{insight}</div>
    <p class="stats-title">Your check-in snapshot</p>
    <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;">
      <tr>
        <td width="31%" style="background:#f4f4f8;border-radius:12px;padding:14px;text-align:center;">
          <div style="font-size:22px;font-weight:700;color:#6C63FF;">{data['mood']}/10</div>
          <div style="font-size:11px;color:#999;margin-top:4px;">Mood</div>
        </td>
        <td width="4%"></td>
        <td width="31%" style="background:#f4f4f8;border-radius:12px;padding:14px;text-align:center;">
          <div style="font-size:22px;font-weight:700;color:#6C63FF;">{data['sleep_hours']}hrs</div>
          <div style="font-size:11px;color:#999;margin-top:4px;">Sleep</div>
        </td>
        <td width="4%"></td>
        <td width="31%" style="background:#f4f4f8;border-radius:12px;padding:14px;text-align:center;">
          <div style="font-size:22px;font-weight:700;color:#6C63FF;">{data['stress_self']}/10</div>
          <div style="font-size:11px;color:#999;margin-top:4px;">Stress</div>
        </td>
      </tr>
    </table>
  </div>
  <div class="footer">
    <p>Generated by <span class="brand">EchoSense AI</span> · No personal data stored · Not medical advice</p>
    <p style="margin-top:8px;color:#bbb;">Stay real 🖤</p>
  </div>
</div></body></html>
"""

    payload = {
        "to": req.email,
        "subject": f"EchoSense AI — Your emotional check-in, {req.name} 🧠",
        "body": html_body,
        "name": req.name
    }

    send_to_n8n(payload)

    return {"status": "sent", "insight": insight}


@app.get("/")
def root():
    return {"status": "EchoSense AI backend running"}