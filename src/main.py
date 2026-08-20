import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
chat_model = os.getenv("CHAT_MODEL")
embed_model = os.getenv("EMBED_MODEL")

print("NERI development environment")
print("----------------------------")
print(f"API Base URL configured: {bool(base_url)}")
print(f"API Key configured: {bool(api_key)}")
print(f"Chat model configured: {bool(chat_model)}")
print(f"Embedding model configured: {bool(embed_model)}")
print("Workspace setup is working.")