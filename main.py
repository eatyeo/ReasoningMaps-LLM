import asyncio
import os
import pandas as pd
import httpx
from load_lsat import fetch_lsat_data
from llm_client import get_llm_reasoning
from reasoning_parser import ReasoningMap

# --- CONFIGURATION ---
# run 5 for a test (will change to larger number later)
NUM_PROBLEMS_TO_ANALYZE = 5
RESULTS_FILE = "results.csv"
MAPS_DIR = "reasoning_maps"

async def process_problem(problem, session):
    """
    Analyzes a single LSAT problem and returns a dictionary of results.
    We pass 'session' to reuse the same httpx client.
    """
    print(f"\n--- Analyzing Problem: {problem['id_string']} ---")

    # 1. GET LLM REASONING
    raw_text = await get_llm_reasoning(problem, session)
    
    if "Error" in raw_text:
        print(f"Failed to get LLM reasoning: {raw_text}")
        return {
            "id_string": problem['id_string'],
            "was_llm_correct": False,
            "llm_answer": "Error",
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

    llm_answer_char = "N/A"
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
    
    # Create one httpx session to reuse connections (much faster)
    # Need to get the API key for this one-time client
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("LLM_KEY")
    
    if not api_key:
         print("CRITICAL: LLM_KEY not found in .env. Exiting.")
         return
         
    # This is the base URL without the ?key= part
    base_url = "https://generativelanguage.googleapis.com"
    
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as session:
        # Pass the API key to the function
        session.api_key = api_key 
        
        # Create a list of tasks to run concurrently
        tasks = [process_problem(problem, session) for problem in lsat_data]
        results = await asyncio.gather(*tasks)
        
        all_results.extend(results)

    # 5. SAVE TO CSV
    print(f"\n--- Analysis Complete ---")
    
    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(RESULTS_FILE, index=False)
    print(f"All results saved to {RESULTS_FILE}")

    # 6. PRINT SUMMARY
    if not df.empty:
        accuracy = df['was_llm_correct'].mean() * 100
        print(f"Overall LLM Accuracy: {accuracy:.2f}%")
        print(f"Total Errors (API or Parsing): {df[df['llm_answer'] == 'Error'].shape[0]}")
    

if __name__ == "__main__":
    asyncio.run(main())