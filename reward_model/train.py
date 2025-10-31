import os
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import wandb
from pathlib import Path

from model import FinancialRewardModel, PairwiseRewardModel
from dataset import FinancialRewardDataset, collate_fn_pointwise, collate_fn_pairwise


class RewardModelTrainer:
    """Reward Model分布式训练器"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_distributed()
        self.setup_logging()
        self.setup_model_and_data()
        
    def setup_distributed(self):
        """初始化分布式训练环境"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            
            dist.init_process_group(
                backend=self.config['distributed']['backend'],
                init_method='env://'
            )
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device('cuda', self.local_rank)
        else:
            # 单机训练
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_main_process = (self.rank == 0)
    
    def setup_logging(self):
        """设置日志"""
        if self.is_main_process:
            Path(self.config['output']['output_dir']).mkdir(parents=True, exist_ok=True)
            Path(self.config['output']['logging_dir']).mkdir(parents=True, exist_ok=True)
            
            if self.config['wandb']['enabled']:
                wandb.init(
                    project=self.config['wandb']['project'],
                    name=self.config['wandb']['run_name'],
                    config=self.config
                )
    
    def setup_model_and_data(self):
        """初始化模型和数据"""
        model_config = self.config['model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['base_model'])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 根据配置选择模型类型
        weight_strategy = self.config['reward_dimensions']['weight_strategy']
        if weight_strategy == "multi_head":
            self.model = FinancialRewardModel(
                base_model_name=model_config['base_model'],
                num_dimensions=len(self.config['reward_dimensions']) - 2,  # 去掉weight_strategy和dimension_weights
                use_multi_head=True,
                pooling_strategy="last"
            )
            self.training_mode = "pointwise"
        elif weight_strategy in ["average", "weighted"]:
            self.model = FinancialRewardModel(
                base_model_name=model_config['base_model'],
                num_dimensions=3,
                use_multi_head=False,
                pooling_strategy="last"
            )
            self.training_mode = "pointwise"
        else:
            raise ValueError(f"Unknown weight strategy: {weight_strategy}")
        
        self.model = self.model.to(self.device)
        
        # 分布式包装
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self.config['distributed']['find_unused_parameters']
            )
        
        # 创建数据集
        self.train_dataset = FinancialRewardDataset(
            data_path=self.config['data']['train_file'],
            tokenizer=self.tokenizer,
            max_length=model_config['max_length'],
            mode=self.training_mode
        )
        
        self.val_dataset = FinancialRewardDataset(
            data_path=self.config['data']['val_file'],
            tokenizer=self.tokenizer,
            max_length=model_config['max_length'],
            mode=self.training_mode
        )
        
        # 创建数据加载器
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        collate_fn = collate_fn_pointwise if self.training_mode == "pointwise" else collate_fn_pairwise
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # 优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        num_training_steps = len(self.train_loader) * self.config['training']['num_epochs'] // self.config['training']['gradient_accumulation_steps']
        num_warmup_steps = int(num_training_steps * self.config['training']['warmup_ratio'])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs['loss'] / self.config['training']['gradient_accumulation_steps']
            loss.backward()
            
            if (step + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                total_loss += loss.item() * self.config['training']['gradient_accumulation_steps']
                
                if self.is_main_process:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * self.config['training']['gradient_accumulation_steps']:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                    if self.global_step % self.config['training']['logging_steps'] == 0:
                        if self.config['wandb']['enabled']:
                            wandb.log({
                                'train/loss': loss.item() * self.config['training']['gradient_accumulation_steps'],
                                'train/learning_rate': self.scheduler.get_last_lr()[0],
                                'train/epoch': epoch,
                                'train/global_step': self.global_step
                            })
                
                if self.global_step % self.config['training']['eval_steps'] == 0:
                    val_loss = self.evaluate()
                    if self.is_main_process and val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best_model')
                    self.model.train()
                
                if self.global_step % self.config['training']['save_steps'] == 0:
                    if self.is_main_process:
                        self.save_checkpoint(f'checkpoint-{self.global_step}')
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating", disable=not self.is_main_process):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs['loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        if self.is_main_process:
            if self.config['wandb']['enabled']:
                wandb.log({
                    'val/loss': avg_loss,
                    'val/global_step': self.global_step
                })
        
        return avg_loss
    
    def save_checkpoint(self, checkpoint_name: str):
        """保存检查点"""
        output_dir = Path(self.config['output']['output_dir']) / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }, output_dir / 'pytorch_model.bin')
        
        self.tokenizer.save_pretrained(output_dir)
    
    def train(self):
        """完整训练流程"""
        for epoch in range(self.config['training']['num_epochs']):
            train_loss = self.train_epoch(epoch)
            
            if self.is_main_process:
                val_loss = self.evaluate()
                
                if self.config['wandb']['enabled']:
                    wandb.log({
                        'epoch': epoch,
                        'train/epoch_loss': train_loss,
                        'val/epoch_loss': val_loss
                    })
        
        if self.is_main_process:
            self.save_checkpoint('final_model')
        
        if self.world_size > 1:
            dist.destroy_process_group()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    trainer = RewardModelTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()

