import pandas as pd
import os
import re

RESULTS_FILE = "results.csv"

def categorize_question(question_text):
    """
    Analyzes the question text and assigns it to a
    major LSAT category based on keywords.
    """
    # Ensure question_text is a string
    if not isinstance(question_text, str):
        return "Other/Uncategorized"
        
    q_lower = question_text.lower() # Make it case-insensitive
    
    if "flaw" in q_lower or "vulnerable to criticism" in q_lower:
        return "Flaw"
    if "assumption" in q_lower:
        return "Assumption"
    if "strengthen" in q_lower or "supports" in q_lower or "helps to" in q_lower:
        return "Strengthen"
    if "weaken" in q_lower or "casts doubt" in q_lower:
        return "Weaken"
    if "infer" in q_lower or "must be true" in q_lower:
        return "Inference (Must Be True)"
    if "parallel" in q_lower or "similar to" in q_lower:
        return "Parallel Reasoning"
    if "main point" in q_lower or "main conclusion" in q_lower:
        return "Main Point"
    if "principle" in q_lower:
        return "Principle"
    if "reconcile" in q_lower or "explain the discrepancy" in q_lower:
        return "Resolve/Reconcile"
    if "accurately describes" in q_lower:
        return "Method of Reasoning"
        
    return "Other/Uncategorized"

def analyze_results():
    """
    Loads the results.csv file and performs a statistical analysis
    to find recurring patterns of error.
    """
    print(f"--- Analyzing Results from {RESULTS_FILE} ---")

    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        print("Please run 'python main.py' first to generate the results.")
        return

    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(RESULTS_FILE)
    
    total_problems = len(df)
    if total_problems == 0:
        print("Error: results.csv is empty.")
        return

    # --- ANALYSIS ---
    
    # 1. API Error Analysis
    api_errors = df[df['llm_answer'] == 'API Error']
    api_error_count = len(api_errors)
    
    print("\n--- API & Parsing Health ---")
    print(f"Total Problems Processed: {total_problems}")
    print(f"API Errors (e.g., 429 Limit): {api_error_count}")
    
    # 2. Filter out API errors to get a clean DataFrame for accuracy
    clean_df = df[df['llm_answer'] != 'API Error'].copy()

    if clean_df.empty:
        print("No successful API responses to analyze.")
        return
        
    # 3. Overall Statistics (on clean data)
    accuracy = clean_df['was_llm_correct'].mean() * 100
    correct_count = clean_df['was_llm_correct'].sum()
    incorrect_count = len(clean_df) - correct_count
    
    print("\n--- Overall Performance (on successful requests) ---")
    print(f"Total Successful Requests: {len(clean_df)}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {incorrect_count}")
    print(f"LLM Accuracy: {accuracy:.2f}%")

    # 4. Create the 'question_type' column on the clean DataFrame
    clean_df.loc[:, 'question_type'] = clean_df['question_text'].apply(categorize_question)

    # 5. Error Analysis (on clean data)
    error_df = clean_df[clean_df['was_llm_correct'] == False]
    
    if error_df.empty:
        print("\n--- Error Analysis ---")
        print("No errors found! The LLM was 100% correct on successful requests.")
        return

    print("\n--- Recurring Patterns of Error ---")
    print("The LLM struggled most with the following question types:")
    
    error_counts = error_df['question_type'].value_counts()
    
    for q_type, count in error_counts.items():
        print(f'  - {q_type} \n    (Failed {count} time(s))')

if __name__ == "__main__":
    analyze_results()