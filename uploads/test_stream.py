import requests
import json


url = "http://127.0.0.1:8000/query/stream"

payload = {
    "question": "What should a technician do if abnormal vibration is detected?"
}


response = requests.post(
    url,
    json=payload,
    stream=True,
    timeout=120,
)


print("=" * 70)
print("STREAMING TEST")
print("=" * 70)


for line in response.iter_lines(decode_unicode=True):

    if not line:
        continue

    if line.startswith("data: "):

        event = json.loads(
            line[6:]
        )

        print(event)