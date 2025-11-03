"""
多维度评分奖励模型训练脚本
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import json
from pathlib import Path

from model import FinancialRewardModel
from dataset_multidim import MultiDimRewardDataset, get_label_distribution


class MultiDimRewardTrainer:
    """多维度奖励模型训练器"""

    def __init__(
        self,
        model: FinancialRewardModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        output_dir: str,
        dimension_names: list = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dimension_names = dimension_names or ["depth", "professionalism", "accuracy"]

        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0

    def train_epoch(self, epoch: int) -> dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_per_dim_accuracy = torch.zeros(self.model.num_dimensions).to(self.device)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs["loss"]
            accuracy = outputs["accuracy"]
            per_dim_accuracy = outputs["per_dim_accuracy"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            total_per_dim_accuracy += per_dim_accuracy

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_accuracy / len(self.train_loader)
        avg_per_dim_accuracy = total_per_dim_accuracy / len(self.train_loader)

        metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
        }

        # Add per-dimension accuracy
        for i, dim_name in enumerate(self.dimension_names):
            metrics[f"accuracy_{dim_name}"] = avg_per_dim_accuracy[i].item()

        return metrics

    def validate(self, epoch: int) -> dict:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_per_dim_accuracy = torch.zeros(self.model.num_dimensions).to(self.device)

        # Confusion matrix for each dimension
        confusion_matrices = []
        for _ in range(self.model.num_dimensions):
            confusion_matrices.append(
                torch.zeros(
                    (self.model.num_classes, self.model.num_classes), dtype=torch.long
                )
            )

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs["loss"]
                accuracy = outputs["accuracy"]
                per_dim_accuracy = outputs["per_dim_accuracy"]
                predicted_labels = outputs["predicted_labels"]

                # Update metrics
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                total_per_dim_accuracy += per_dim_accuracy

                # Update confusion matrices
                for dim_idx in range(self.model.num_dimensions):
                    for true_label, pred_label in zip(
                        labels[:, dim_idx].cpu(), predicted_labels[:, dim_idx].cpu()
                    ):
                        confusion_matrices[dim_idx][true_label, pred_label] += 1

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.4f}"}
                )

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)
        avg_per_dim_accuracy = total_per_dim_accuracy / len(self.val_loader)

        metrics = {
            "loss": avg_loss,
            "accuracy": avg_accuracy,
        }

        # Add per-dimension metrics
        for i, dim_name in enumerate(self.dimension_names):
            metrics[f"accuracy_{dim_name}"] = avg_per_dim_accuracy[i].item()
            
            # Calculate per-dimension F1 score (macro average)
            cm = confusion_matrices[i]
            precisions = []
            recalls = []
            for c in range(self.model.num_classes):
                tp = cm[c, c].item()
                fp = cm[:, c].sum().item() - tp
                fn = cm[c, :].sum().item() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            avg_precision = sum(precisions) / len(precisions)
            avg_recall = sum(recalls) / len(recalls)
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            metrics[f"f1_{dim_name}"] = f1

        return metrics

    def train(self, num_epochs: int, save_every: int = 1):
        """完整训练流程"""
        history = {"train": [], "val": []}

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 70}")

            # Train
            train_metrics = self.train_epoch(epoch)
            history["train"].append(train_metrics)

            print("\nTrain Metrics:")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Validate
            val_metrics = self.validate(epoch)
            history["val"].append(val_metrics)

            print("\nValidation Metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_metrics)

            # Save best model
            if val_metrics["accuracy"] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics["accuracy"]
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"\n✅ New best model saved! Accuracy: {self.best_val_accuracy:.4f}")

        # Save training history
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\n{'=' * 70}")
        print("Training completed!")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        print(f"{'=' * 70}")

        return history

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }

        if is_best:
            checkpoint_path = self.output_dir / "best_model.pt"
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, checkpoint_path)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")


def main():
    # Configuration
    config = {
        "base_model_name": "bert-base-chinese",  # or your preferred model
        "data_file": "./reward_model/data/comparison_pairs_scored.jsonl",
        "output_dir": "./reward_model/outputs/multidim",
        "max_length": 2048,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "val_split": 0.1,
        "num_dimensions": 3,
        "num_classes": 5,
        "use_multi_head": True,
        "pooling_strategy": "last",
        "dimension_names": ["depth", "professionalism", "accuracy"],
    }

    print("=" * 70)
    print("多维度奖励模型训练")
    print("=" * 70)
    print("\n配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # Analyze data distribution
    print("\n" + "=" * 70)
    print("数据分布分析")
    print("=" * 70)
    stats = get_label_distribution(config["data_file"], config["dimension_names"])
    print(f"总样本数: {stats['total_samples']}")
    
    dimension_name_map = {
        "depth": "分析深度",
        "professionalism": "专业度",
        "accuracy": "数值准确性",
    }
    
    for dim in config["dimension_names"]:
        print(f"\n{dimension_name_map[dim]}分布:")
        for label in range(config["num_classes"]):
            pct = stats["overall_distribution"][dim][label] * 100
            print(f"  {label}分: {pct:5.1f}%")

    # Load tokenizer
    print("\n加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_name"])

    # Load dataset
    print("加载数据集...")
    full_dataset = MultiDimRewardDataset(
        data_file=config["data_file"],
        tokenizer=tokenizer,
        max_length=config["max_length"],
        include_prompt=True,
        dimensions=config["dimension_names"],
    )

    # Split dataset
    val_size = int(len(full_dataset) * config["val_split"])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    # Initialize model
    print("\n初始化模型...")
    model = FinancialRewardModel(
        base_model_name=config["base_model_name"],
        num_dimensions=config["num_dimensions"],
        num_classes=config["num_classes"],
        use_multi_head=config["use_multi_head"],
        pooling_strategy=config["pooling_strategy"],
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    # Use cosine annealing with warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps,
        num_cycles=0.5  # 0.5 means cosine goes from max to 0, default behavior
    )

    print(f"\n总训练步数: {total_steps}")
    print(f"预热步数: {warmup_steps}")
    print(f"学习率调度: Cosine Annealing with Warmup")

    # Create trainer
    trainer = MultiDimRewardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=config["output_dir"],
        dimension_names=config["dimension_names"],
    )

    # Train
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)

    history = trainer.train(num_epochs=config["num_epochs"], save_every=2)

    print("\n训练完成！")
    print(f"模型保存在: {config['output_dir']}")


if __name__ == "__main__":
    main()

