# rag_email_agent.py
import os
from groq import Groq
from memory_engine import search_similar
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_personal_insight(row: dict) -> str:
    similar = search_similar(str(row.get("reflection", "")), n_results=3)

    similar_texts = "\n".join(
        [f'- "{s["text"]}"' for s in similar]
    ) if similar else "No similar reflections found yet."

    name = row.get("name", "Friend")
    emotion = row.get("dominant_emotion", "neutral")

    if emotion in ["joy", "neutral"]:
        tone = """You're writing to someone who's doing well or feeling okay.
Be witty, celebratory, playful. Match their energy. Maybe throw in a light joke.
Make them feel seen and keep it fun."""
    elif emotion in ["sadness", "fear"]:
        tone = """You're writing to someone who's struggling emotionally.
Be warm, gentle, and genuinely caring. No toxic positivity.
Acknowledge their feeling first before anything else.
Make them feel less alone."""
    elif emotion in ["anger", "disgust"]:
        tone = """You're writing to someone who's frustrated or fed up.
Be calm, grounding, and validating. Don't dismiss their frustration.
Acknowledge it fully, then gently redirect toward something solid."""
    else:
        tone = """You're writing to someone under heavy stress.
Be practical and reassuring. Give them one small, concrete thing to focus on.
Keep it short, calm, and real — no fluff."""

    prompt = f"""
You are EchoSense AI — a student wellbeing companion.

TONE INSTRUCTION:
{tone}

Rules:
- 4-6 sentences MAX
- Reference their actual data naturally
- Do NOT start with "Hey {name}"
- Do NOT say you're an AI
- Do NOT give medical advice
- No motivational poster endings

--- STUDENT DATA ---
Name: {name}
Mood: {row.get("mood", "?")} / 10
Sleep: {row.get("sleep_hours", "?")} hrs
Study load: {row.get("study_load", "?")}
Stress: {row.get("stress_self", "?")} / 10
Dominant emotion: {emotion}
Reflection: "{row.get("reflection", "")}"

--- SIMILAR COMMUNITY REFLECTIONS (RAG) ---
{similar_texts}

--- YOUR MESSAGE ---
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.9
    )

    return response.choices[0].message.content.strip()


def build_email_payload(row: dict, insight: str) -> dict:
    name = row.get("name", "Friend")
    email = row.get("email", "")
    subject = f"EchoSense AI — Your emotional check-in, {name} 🧠"
    mood = row.get("mood", "?")
    stress = row.get("stress_self", "?")
    sleep = row.get("sleep_hours", "?")
    emotion = row.get("dominant_emotion", "unknown")

    if str(mood) != "?":
        mood_val = float(mood)
        mood_emoji = "🌟" if mood_val >= 8 else "😊" if mood_val >= 5 else "💙"
    else:
        mood_emoji = "💭"

    html_body = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body {{ margin: 0; padding: 0; background-color: #f4f4f8; font-family: 'Segoe UI', Arial, sans-serif; }}
  .wrapper {{ max-width: 580px; margin: 30px auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
  .header {{ background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 100%); padding: 36px 32px; text-align: center; }}
  .header h1 {{ color: white; margin: 0; font-size: 22px; font-weight: 700; }}
  .header p {{ color: rgba(255,255,255,0.85); margin: 6px 0 0; font-size: 14px; }}
  .body {{ padding: 32px; }}
  .greeting {{ font-size: 20px; font-weight: 600; color: #1a1a2e; margin: 0 0 16px; }}
  .insight {{ font-size: 15px; line-height: 1.75; color: #444; background: #f8f7ff; border-left: 4px solid #6C63FF; border-radius: 0 12px 12px 0; padding: 18px 20px; margin: 0 0 28px; }}
  .stats-title {{ font-size: 12px; font-weight: 600; color: #999; text-transform: uppercase; letter-spacing: 0.08em; margin: 0 0 12px; }}
  .footer {{ background: #f8f7ff; padding: 20px 32px; text-align: center; border-top: 1px solid #eee; }}
  .footer p {{ font-size: 12px; color: #aaa; margin: 4px 0; line-height: 1.6; }}
  .brand {{ font-weight: 700; color: #6C63FF; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>🧠 EchoSense AI</h1>
    <p>Your personal emotional check-in report</p>
  </div>
  <div class="body">
    <p class="greeting">Hey {name} {mood_emoji}</p>
    <div class="insight">{insight}</div>
    <p class="stats-title">Your check-in snapshot</p>
    <table width="100%" cellpadding="0" cellspacing="0" style="margin:0 0 20px;">
      <tr>
        <td width="31%" style="background:#f4f4f8;border-radius:12px;padding:14px;text-align:center;">
          <div style="font-size:22px;font-weight:700;color:#6C63FF;">{mood}/10</div>
          <div style="font-size:11px;color:#999;margin-top:4px;">Mood</div>
        </td>
        <td width="4%"></td>
        <td width="31%" style="background:#f4f4f8;border-radius:12px;padding:14px;text-align:center;">
          <div style="font-size:22px;font-weight:700;color:#6C63FF;">{sleep}hrs</div>
          <div style="font-size:11px;color:#999;margin-top:4px;">Sleep</div>
        </td>
        <td width="4%"></td>
        <td width="31%" style="background:#f4f4f8;border-radius:12px;padding:14px;text-align:center;">
          <div style="font-size:22px;font-weight:700;color:#6C63FF;">{stress}/10</div>
          <div style="font-size:11px;color:#999;margin-top:4px;">Stress</div>
        </td>
      </tr>
    </table>
    <p class="stats-title">AI detected emotion</p>
    <span style="display:inline-block;background:#EEF2FF;color:#6C63FF;border-radius:20px;padding:6px 16px;font-size:13px;font-weight:600;">✨ {emotion.capitalize()}</span>
  </div>
  <div class="footer">
    <p>Generated by <span class="brand">EchoSense AI</span> based on your anonymous check-in.</p>
    <p>No personal data is stored. This is not medical advice.</p>
    <p style="margin-top:8px;color:#bbb;">Stay real 🖤</p>
  </div>
</div>
</body>
</html>
"""

    return {
        "to": email,
        "subject": subject,
        "body": html_body,
        "name": name
    }