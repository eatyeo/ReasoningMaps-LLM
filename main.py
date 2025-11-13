import asyncio
from load_lsat import fetch_lsat_data
from llm_client import get_llm_reasoning
from reasoning_parser import ReasoningMap

async def main():
    """
    The main function to run the end-to-end analysis.
    """
    print("--- Starting Reasoning Map Analysis ---")
    
    # 1. FETCH DATA
    # Let's just analyze the first problem for now.
    lsat_data = fetch_lsat_data(num_samples=1)
    if not lsat_data:
        print("No LSAT data found. Exiting.")
        return
        
    problem = lsat_data[0]
    print(f"Loaded Problem ID: {problem['id_string']}")

    # 2. GET LLM REASONING
    raw_text = await get_llm_reasoning(problem)
    
    if "Error" in raw_text:
        print("Failed to get LLM reasoning.")
        print(raw_text)
        return
        
    print("...Got reasoning from LLM.")

    # 3. PARSE AND BUILD MAP
    map = ReasoningMap(raw_text, problem)
    
    map.parse_reasoning()
    map.analyze_correctness()
    map.build_graph()
    
    # 4. VISUALIZE
    map_filename = f"{problem['id_string']}_map.png"
    map.visualize(save_path=map_filename)

    # 5. PRINT ANALYSIS
    print("\n--- Analysis Complete ---")
    print(f"LLM's Final Answer: {chr(map.llm_answer + ord('A'))}")
    print(f"Correct Answer: {chr(problem['label'] + ord('A'))}")
    print(f"Was LLM correct? {'YES' if map.is_correct else 'NO'}")
    print(f"Visualization saved to: {map_filename}")

if __name__ == "__main__":
    asyncio.run(main())
