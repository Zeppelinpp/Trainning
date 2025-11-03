"""
Test script to verify dataset loading and format
"""
import torch
from transformers import AutoTokenizer
from dataset import FinancialRewardDataset

def test_dataset():
    print("Testing FinancialRewardDataset...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # Create dataset
    dataset = FinancialRewardDataset(
        data_path="data/train.jsonl",
        tokenizer=tokenizer,
        max_length=512,
        mode="pointwise"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test first sample
    sample = dataset[0]
    
    print("\nFirst sample:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  labels dtype: {sample['labels'].dtype}")
    print(f"  labels values: {sample['labels']}")
    
    # Verify labels are in valid range (0-4)
    assert sample['labels'].min() >= 0, "Labels should be >= 0"
    assert sample['labels'].max() <= 4, "Labels should be <= 4"
    assert sample['labels'].dtype == torch.long, "Labels should be long dtype"
    
    print("\n✅ All checks passed!")
    
    # Test a few more samples
    print("\nTesting 5 random samples:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"  Sample {i}: labels = {sample['labels'].tolist()}")
    
    print("\n✅ Dataset test completed successfully!")

if __name__ == "__main__":
    test_dataset()

