import json
import os
import httpx
import asyncio
from dotenv import load_dotenv

# 1. Load the .env file to get API key
load_dotenv()

# 2. Import the data loader function
try:
    from load_lsat import fetch_lsat_data
except ImportError:
    print("CRITICAL ERROR: Could not import 'load_lsat.py'.")
    print("Make sure both files are in the same folder.")
    exit()

# 3. Setup Gemini API
API_KEY = os.getenv("LLM_KEY")
MODEL = "gemini-2.0-flash"  # Use an available Gemini model (adjust as needed)
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"


async def get_llm_reasoning(lsat_problem):
    if not API_KEY:
        return "Error: LLM_KEY not found. Check your .env file."

    # --- Format Options ---
    options_text = ""
    for i, answer_text in enumerate(lsat_problem['answers']):
        options_text += f"({chr(65 + i)}): {answer_text}\n"

    # --- System Prompt ---
    system_prompt = """
You are a master logician.
Core Principle: The "type" of question (e.g., Flaw, Assumption) dictates the strategy.

Output Format:
1. Argument Breakdown: Premises and Conclusion.
2. Question Analysis: Identify the question type and strategy.
3. Strategic Evaluation: Analyze each choice based on the strategy.
4. Final Conclusion: State the correct answer letter.
"""

    # --- User Prompt ---
    user_prompt = f"""
Context: {lsat_problem['context']}
Question: {lsat_problem['question']}
Choices:
{options_text}
"""

    # --- Gemini Payload ---
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt + "\n" + user_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 1024
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    # --- Make Request ---
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(API_URL, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()
                # Gemini responses store text here:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"


# --- Main Test Runner ---
async def main():
    print("--- Starting Gemini Client ---")

    # 1. Fetch Data
    data = fetch_lsat_data(1)
    if not data:
        print("No data found. Exiting.")
        return

    # 2. Send to Gemini
    print(f"Analyzing Problem ID: {data[0]['id_string']}...")
    reasoning = await get_llm_reasoning(data[0])

    # 3. Print Result
    print("\n" + "=" * 30)
    print("AI REASONING OUTPUT:")
    print("=" * 30)
    print(reasoning)


if __name__ == "__main__":
    asyncio.run(main())