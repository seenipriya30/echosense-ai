# run_agent.py
import pandas as pd
from pipeline import load_data, run_emotion_analysis, get_new_responses_with_email
from memory_engine import store_reflections
from rag_email_agent import generate_personal_insight, build_email_payload
from webhook_sender import send_to_n8n

print("\n🧠 EchoSense AI Agent Starting...\n")

# Load + process data
df = load_data()
df = run_emotion_analysis(df)

# Store all reflections in ChromaDB memory
result = store_reflections(df)
print(f"📚 Memory: {result['added']} new reflections stored, {result['skipped']} already exist")

# Get students who opted in for personalized message
opted_in = get_new_responses_with_email(df)
print(f"📬 {len(opted_in)} students opted in for personalized message\n")

if len(opted_in) == 0:
    print("No new opted-in responses to process.")
else:
    for row in opted_in:
        print(f"Processing: {row.get('name', 'Unknown')} ({row.get('email', '')})")

        # RAG + LLM generate insight
        insight = generate_personal_insight(row)
        print(f"💬 Insight generated:\n{insight}\n")

        # Build email payload
        payload = build_email_payload(row, insight)

        # Send to n8n webhook
        send_to_n8n(payload)

print("\n✅ EchoSense Agent finished.")