# test_single.py
from rag_email_agent import generate_personal_insight, build_email_payload
from webhook_sender import send_to_n8n

# Use YOUR details for testing
test_row = {
    "name": "Seeni",
    "email": "seenipriya30@gmail.com",  # your email
    "mood": 3.0,
    "sleep_hours": 4.0,
    "study_load": 2,
    "stress_self": 8.0,
    "stress_index": 39.2,
    "dominant_emotion": "sadness",
    "reflection": "boring and sad"
}

print("🧠 Generating insight...")
insight = generate_personal_insight(test_row)
print(f"\n💬 Insight:\n{insight}\n")

payload = build_email_payload(test_row, insight)
print(f"📧 Sending to: {payload['to']}")

result = send_to_n8n(payload)
if result:
    print("✅ Check your inbox!")
else:
    print("❌ Something went wrong")