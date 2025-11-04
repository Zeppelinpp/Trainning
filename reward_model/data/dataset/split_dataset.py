import json
import random
from pathlib import Path

def split_dataset(input_file: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
    """
    Split JSONL dataset into train/val/test sets
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save split files
        train_ratio: Ratio for training set (default: 0.7)
        val_ratio: Ratio for validation set (default: 0.2)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_count = len(lines)
    print(f"Total samples: {total_count}")
    
    # Shuffle the data
    random.shuffle(lines)
    
    # Calculate split sizes
    train_size = int(total_count * train_ratio)
    val_size = int(total_count * val_ratio)
    test_size = total_count - train_size - val_size
    
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Split data
    train_data = lines[:train_size]
    val_data = lines[train_size:train_size + val_size]
    test_data = lines[train_size + val_size:]
    
    # Write to files
    with open(output_path / 'train.jsonl', 'w', encoding='utf-8') as f:
        for line in train_data:
            f.write(line + '\n')
    
    with open(output_path / 'val.jsonl', 'w', encoding='utf-8') as f:
        for line in val_data:
            f.write(line + '\n')
    
    with open(output_path / 'test.jsonl', 'w', encoding='utf-8') as f:
        for line in test_data:
            f.write(line + '\n')
    
    print(f"Successfully split dataset:")
    print(f"  - train.jsonl: {len(train_data)} samples")
    print(f"  - val.jsonl: {len(val_data)} samples")
    print(f"  - test.jsonl: {len(test_data)} samples")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_file = script_dir / "comparison_pairs_scored.jsonl"
    output_dir = script_dir
    
    split_dataset(str(input_file), str(output_dir), train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

