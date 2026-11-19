import os
import urllib.error
import urllib.request
from dotenv import load_dotenv

load_dotenv()

print("🔍 Testing your API Keys...\n" + "=" * 40)

# 1. Test IBM Cloud API Key
ibm_key = os.getenv("WATSONX_API_KEY")
try:
  data = (
      "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=" + ibm_key
  ).encode("utf-8")
  req = urllib.request.Request(
      "https://iam.cloud.ibm.com/identity/token", data=data
  )
  req.add_header("Content-Type", "application/x-www-form-urlencoded")
  with urllib.request.urlopen(req, timeout=10) as resp:
    if resp.status == 200:
      print("✅ IBM Cloud API Key: WORKING & VALID!")
except Exception as e:
  print("❌ IBM Key Error:", e)

# 2. Test Google Gemini API Key
gemini_key = os.getenv("GEMINI_API_KEY")
try:
  url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}"
  with urllib.request.urlopen(url, timeout=10) as resp:
    if resp.status == 200:
      print("✅ Google Gemini API Key: WORKING & VALID!")
except Exception as e:
  print("❌ Gemini Key Error:", e)

# 3. Test Groq API Key
groq_key = os.getenv("GROQ_API_KEY")
try:
  req = urllib.request.Request("https://api.groq.com/openai/v1/models")
  req.add_header("Authorization", f"Bearer {groq_key}")
  req.add_header("User-Agent", "Mozilla/5.0")
  with urllib.request.urlopen(req, timeout=10) as resp:
    if resp.status == 200:
      print("✅ Groq API Key: WORKING & VALID!")
except Exception as e:
  print("❌ Groq Key Error:", e)

print("=" * 40)