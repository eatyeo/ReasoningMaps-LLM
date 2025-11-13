# You must have the 'datasets' library installed:
# pip install datasets

from datasets import load_dataset
import json

def fetch_lsat_data(num_samples=5):
    """
    Loads the 'tasksource/lsat-lr' dataset from Hugging Face
    and prints a few samples to show its structure.
    
    This dataset contains actual LSAT Logical Reasoning questions.
    """
    dataset_name = "tasksource/lsat-lr"
    
    try:
        # Load the training split of the dataset
        dataset = load_dataset(dataset_name, split="train")
        
        print(f"Successfully loaded dataset: {dataset_name}")
        print(f"Total number of problems in 'train' split: {len(dataset)}")
        print("-" * 30)
        
        # --- Print the first few samples ---
        print(f"Showing the first {num_samples} samples:\n")
        
        for i in range(num_samples):
            sample = dataset[i]
            
            print(f"--- SAMPLE {i + 1} ---")
            
            # Use json.dumps for clean, indented printing of the dictionary
            print(json.dumps(sample, indent=2))
            
            # --- How this data maps to your project ---
            # sample['context']  -> The paragraph/argument to analyze
            # sample['question'] -> The question about the argument
            # sample['options']  -> List of multiple-choice answer strings
            # sample['label']    -> The *index* (0-4) of the correct answer
            
            print("\n")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection and have run:")
        print("pip install datasets")

if __name__ == "__main__":
    # Fetch and print 5 samples so you can see the data structure
    fetch_lsat_data(num_samples=5)