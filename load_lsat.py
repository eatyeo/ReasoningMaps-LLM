from datasets import load_dataset
import json

def fetch_lsat_data(num_samples=5):
    """
    Loads the 'tasksource/lsat-lr' dataset from Hugging Face
    and returns a LIST of samples.
    """
    dataset_name = "tasksource/lsat-lr"
    samples_list = []
    
    try:
        # Load the dataset
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split="train")
        
        # Get the requested number of samples
        count = min(num_samples, len(dataset))
        
        # store them in a list
        for i in range(count):
            samples_list.append(dataset[i])
            
        print(f"Successfully loaded {len(samples_list)} samples.")
        return samples_list

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

if __name__ == "__main__":
    # Test the function
    data = fetch_lsat_data(1)
    if data:
        print("Success! Data loaded correctly.")
        print(json.dumps(data[0], indent=2))