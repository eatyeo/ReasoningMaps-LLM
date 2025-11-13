import asyncio
import os
import pandas as pd
import httpx
import time
from load_lsat import fetch_lsat_data
from llm_client import get_llm_reasoning
from reasoning_parser import ReasoningMap
from dotenv import load_dotenv

# --- CONFIGURATION ---
NUM_PROBLEMS_TO_ANALYZE = 100
RESULTS_FILE = "results.csv"
MAPS_DIR = "reasoning_maps"
# Wait 6.1 seconds to stay under Gemini's 10 requests/minute limit
RATE_LIMIT_DELAY = 6.1 

async def process_problem(problem, session):
    """
    Analyzes a single LSAT problem and returns a dictionary of results.
    We pass 'session' to reuse the same httpx client.
    """
    print(f"\n--- Analyzing Problem: {problem['id_string']} ---")

    # 1. GET LLM REASONING
    raw_text = await get_llm_reasoning(problem, session)
    
    # Check for API errors
    if raw_text.startswith("Error"):
        print(f"Failed to get LLM reasoning: {raw_text}")
        return {
            "id_string": problem['id_string'],
            "was_llm_correct": False,
            "llm_answer": "API Error", # Specific error type
            "correct_answer": chr(problem['label'] + ord('A')),
            "question_text": problem['question'],
            "error_message": raw_text,
            "map_filename": "N/A"
        }
        
    print("...Got reasoning from LLM.")

    # 2. PARSE AND BUILD MAP
    map = ReasoningMap(raw_text, problem)
    map.parse_reasoning()
    map.analyze_correctness()
    map.build_graph()
    
    # 3. VISUALIZE
    map_filename = f"{MAPS_DIR}/{problem['id_string']}_map.png"
    map.visualize(save_path=map_filename)

    llm_answer_char = "N/A (Parse Fail)" # Default if parser fails
    if map.llm_answer is not None:
         llm_answer_char = chr(map.llm_answer + ord('A'))

    # 4. RETURN RESULTS
    return {
        "id_string": problem['id_string'],
        "was_llm_correct": map.is_correct,
        "llm_answer": llm_answer_char,
        "correct_answer": chr(problem['label'] + ord('A')),
        "question_text": problem['question'],
        "error_message": "N/A",
        "map_filename": map_filename
    }

async def main():
    """
    The main function to run the end-to-end analysis.
    """
    print(f"--- Starting Reasoning Map Analysis for {NUM_PROBLEMS_TO_ANALYZE} problems ---")
    
    # Create directory for maps if it doesn't exist
    if not os.path.exists(MAPS_DIR):
        os.makedirs(MAPS_DIR)
        print(f"Created directory: {MAPS_DIR}")

    # 1. FETCH DATA
    lsat_data = fetch_lsat_data(num_samples=NUM_PROBLEMS_TO_ANALYZE)
    if not lsat_data:
        print("No LSAT data found. Exiting.")
        return
        
    print(f"Loaded {len(lsat_data)} problems.")
    
    # List to store all our results
    all_results = []
    
    load_dotenv()
    api_key = os.getenv("LLM_KEY")
    
    if not api_key:
         print("CRITICAL: LLM_KEY not found in .env. Exiting.")
         return
         
    base_url = "https://generativelanguage.googleapis.com"
    
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as session:
        session.api_key = api_key 
        
        for i, problem in enumerate(lsat_data):
            result = await process_problem(problem, session)
            all_results.append(result)
            
            # Don't sleep after the very last item
            if i < len(lsat_data) - 1:
                print(f"Waiting {RATE_LIMIT_DELAY}s to avoid rate limit...")
                await asyncio.sleep(RATE_LIMIT_DELAY) # Wait

    # 5. SAVE TO CSV
    print(f"\n--- Analysis Complete ---")
    
    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(RESULTS_FILE, index=False)
    print(f"All results saved to {RESULTS_FILE}")

    # 6. PRINT SUMMARY (moved to analyze_results.py)
    print(f"\nRun 'python analyze_results.py' to see the full report.")
    

if __name__ == "__main__":
    asyncio.run(main())