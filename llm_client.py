import json
import os
import httpx
import asyncio
from dotenv import load_dotenv

try:
    from load_lsat import fetch_lsat_data
except ImportError:
    pass

# API endpoint
MODEL = "gemini-2.5-flash-preview-09-2025"
MODEL_ENDPOINT = f"/v1beta/models/{MODEL}:generateContent"

async def get_llm_reasoning(lsat_problem, session):
    """
    Takes a single LSAT problem, formats it, and calls the Gemini API
    using a provided httpx.AsyncClient session.
    """
    
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
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.5}
    }

    headers = {"Content-Type": "application/json"}
    
    # --- Make Request ---
    try:
        api_key = session.api_key
        
        response = await session.post(
            f"{MODEL_ENDPOINT}?key={api_key}", 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'candidates' in data and data['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text'):
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                return f"Error: Invalid Gemini response format. {json.dumps(data)}"
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except httpx.ReadTimeout:
        return "Error: Request timed out."
    except Exception as e:
        return f"Request failed: {str(e)}"

# --- Main Test Runner ---
async def main_test():
    """
    A standalone test function for this file.
    """
    print("--- Starting LLM Client (Gemini) ---")
    
    # Load .env to get the key for the test
    load_dotenv()
    api_key = os.getenv("LLM_KEY")
    if not api_key:
        print("CRITICAL: LLM_KEY not found. Exiting test.")
        return

    # 1. Fetch Data
    data = fetch_lsat_data(1)
    if not data:
        print("No data found. Exiting.")
        return

    # 2. Send to LLM
    print(f"Analyzing Problem ID: {data[0]['id_string']}...")
    
    base_url = "https://generativelanguage.googleapis.com"
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as session:
        session.api_key = api_key # Add the key to the session
        reasoning = await get_llm_reasoning(data[0], session)
    
    # 3. Print Result
    print("\n" + "="*30)
    print("AI REASONING OUTPUT:")
    print("="*30)
    print(reasoning)

if __name__ == "__main__":
    asyncio.run(main_test())