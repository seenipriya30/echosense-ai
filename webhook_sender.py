# webhook_sender.py
import requests

N8N_WEBHOOK_URL = "https://echosense-ai.app.n8n.cloud/webhook/echosense"

def send_to_n8n(payload: dict) -> bool:
    try:
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"✅ Webhook sent to n8n for {payload.get('name')}")
            return True
        else:
            print(f"❌ n8n returned {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        return False