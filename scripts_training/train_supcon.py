"""
SupCon监督对比学习训练脚本（支持多GPU DDP训练）

用法：
  单GPU:  python scripts/train_supcon.py --config ...
  多GPU:  torchrun --nproc_per_node=4 scripts/train_supcon.py --config ...
"""
import argparse
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_model.supcon.datasets.supcon_dataset import (
    MultiSceneSupConDataset,
    SceneBatchSampler,
    SupConDataset,
    multi_scale_collate_fn,
)
from src.embedding_model.supcon.models.moco_queue import MoCoQueue
from src.embedding_model.utils.config_loader import load_config
from src.embedding_model.utils.supcon_training import (
    build_loss_func,
    build_model,
    get_stage_gate_weights,
    init_distributed,
    parse_queue_size_config,
    sanitize_wandb_project_name,
    train_one_batch,
    validate,
)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def print_main(is_main: bool, message: str):
    if is_main:
        print(message, flush=True)


def main():
    local_rank, rank, world_size = init_distributed()
    is_main = rank == 0
    is_ddp = world_size > 1

    parser = argparse.ArgumentParser(description='SupCon监督对比学习训练')
    parser.add_argument('--config', type=str, default='configs/supcon_config.yaml', help='训练配置文件路径')
    parser.add_argument('--use_eval', action='store_true', default=True, help='是否进行验证')
    parser.add_argument('--no_eval', dest='use_eval', action='store_false', help='禁用验证')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    args = parser.parse_args()

    supcon_config = load_config(args.config)
    data_config_path = supcon_config['supcon']['data']['data_config_path']
    log_info = lambda message: print_main(is_main, message)

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    log_info(f"使用设备: {device}，world_size={world_size}")

    checkpoint_dir = Path(supcon_config['supcon']['output']['checkpoint_dir'])
    if is_main:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(supcon_config['supcon']['output'].get('use_wandb', False))
    if use_wandb and wandb is None:
        raise RuntimeError("use_wandb=true 但当前环境未安装 wandb，请安装 wandb 或关闭 use_wandb")

    resume_wandb_run_id = None
    if args.resume and is_main and use_wandb:
        resume_checkpoint = torch.load(args.resume, map_location='cpu')
        resume_wandb_run_id = resume_checkpoint.get('wandb_run_id')
        if resume_wandb_run_id:
            log_info(f"W&B resume run id: {resume_wandb_run_id}")

    if is_main and use_wandb:
        raw_project = supcon_config['supcon']['output'].get('wandb_project', 'industrial-supcon')
        wandb_project = sanitize_wandb_project_name(raw_project)
        if wandb_project != raw_project:
            log_info(f"W&B project 名称包含非法字符，已自动清洗: '{raw_project}' -> '{wandb_project}'")
        wandb_kwargs = {
            'project': wandb_project,
            'config': supcon_config['supcon'],
        }
        if resume_wandb_run_id:
            wandb_kwargs.update({
                'id': resume_wandb_run_id,
                'resume': 'allow',
            })
        wandb.init(**wandb_kwargs)

    log_info("加载数据集...")
    log_info(f"数据配置文件: {data_config_path}")
    data_config = load_config(data_config_path)
    train_scene_cfgs = data_config['scenes']['train']
    val_scene_cfgs = data_config['scenes'].get('val', [])

    image_size_config = supcon_config['supcon']['data'].get('image_size', 224)
    if isinstance(image_size_config, int):
        image_sizes = [image_size_config]
    elif isinstance(image_size_config, list):
        image_sizes = image_size_config
    else:
        raise ValueError(f"image_size 必须是 int 或 List[int]，当前为 {type(image_size_config)}")

    model_image_size = max(image_sizes)
    val_image_size = max(image_sizes)
    log_info(f"图像尺度配置: {image_sizes}")
    if len(image_sizes) > 1:
        log_info(f"启用多尺度训练: 训练时每个 batch 随机选择 {image_sizes} 中的一个尺度")
        log_info(f"验证集固定使用尺度: {val_image_size}")
    log_info(f"模型初始化使用尺度: {model_image_size}")

    data_cfg = supcon_config['supcon']['data']
    mask_dilation_config = data_cfg.get('mask_dilation', {'enabled': False})
    copy_paste_config = data_cfg.get('copy_paste', {'enabled': False})
    train_augmentation_config = data_cfg.get('train_augmentation', {})
    log_info(f"Mask软膨胀配置: {mask_dilation_config}")
    log_info(f"Copy-paste干扰增强配置: {copy_paste_config}")
    log_info(f"训练增强配置: {train_augmentation_config}")

    train_dataset = MultiSceneSupConDataset(
        scene_cfgs=train_scene_cfgs,
        split='train',
        image_size=image_sizes,
        mask_dilation_config=mask_dilation_config,
        copy_paste_config=copy_paste_config,
        train_augmentation_config=train_augmentation_config,
    )
    train_scene_names = train_dataset.scene_names

    val_datasets = [
        SupConDataset(
            root=s['root'],
            split='val',
            image_size=val_image_size,
            name=s.get('name', s['root']),
            mask_dilation_config=mask_dilation_config,
        )
        for s in val_scene_cfgs
    ]
    val_scene_names = [s.get('name', s['root']) for s in val_scene_cfgs]

    train_counts = train_dataset.scene_sample_counts
    total_train_samples = sum(train_counts)
    log_info(
        f"训练集: scenes={train_dataset.num_scenes}, samples={total_train_samples}, "
        f"scene_samples=min/avg/max {min(train_counts)}/{total_train_samples / len(train_counts):.1f}/{max(train_counts)}"
    )
    if val_datasets:
        val_counts = [len(ds) for ds in val_datasets]
        total_val_samples = sum(val_counts)
        log_info(
            f"验证集: scenes={len(val_datasets)}, samples={total_val_samples}, "
            f"scene_samples=min/avg/max {min(val_counts)}/{total_val_samples / len(val_counts):.1f}/{max(val_counts)}"
        )

    batch_size = data_cfg['batch_size']
    num_workers = data_cfg['num_workers']
    pin_memory = data_cfg['pin_memory']
    if is_main:
        log_info(f"batch_size per GPU: {batch_size}, effective total batch_size: {batch_size * world_size} (world_size={world_size})")

    training_config = supcon_config['supcon']['training']
    total_steps = int(training_config.get('total_steps', 0))
    if total_steps <= 0:
        raise ValueError("training.total_steps 必须配置且 > 0；当前训练脚本不再支持 epochs")

    scene_sampling = training_config.get('scene_sampling', 'size_temperature')
    scene_sampling_alpha = float(training_config.get('scene_sampling_alpha', 0.5))
    train_sampler = SceneBatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        image_sizes=image_sizes,
        total_steps=total_steps,
        scene_sampling=scene_sampling,
        scene_sampling_alpha=scene_sampling_alpha,
        seed=int(training_config.get('seed', 0)),
        rank=rank,
        world_size=world_size,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=multi_scale_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None,
    )
    log_info(
        f"训练采样: total_steps={total_steps}, scene_sampling={scene_sampling}, "
        f"alpha={scene_sampling_alpha}, 单 DataLoader scenes={train_dataset.num_scenes}"
    )

    val_dataloaders = []
    if args.use_eval:
        val_dataloaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
            for ds in val_datasets
        ]

    moco_config = supcon_config['supcon'].get('moco', {})
    use_moco = moco_config.get('enabled', False)
    embedding_source = supcon_config['supcon'].get('inference', {}).get('embedding_source', 'representations')
    if embedding_source not in {'representations', 'projections'}:
        raise ValueError(f"inference.embedding_source 必须是 representations 或 projections，当前为 {embedding_source!r}")

    log_info("创建模型...")
    model_config = supcon_config['supcon']['model']
    model = build_model(
        model_config=model_config,
        moco_config=moco_config,
        use_moco=use_moco,
        image_size=model_image_size,
        freeze_backbone=bool(training_config.get('freeze_backbone', False)),
        device=device,
        log_fn=log_info,
    )

    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        log_info(f"模型已包装为 DDP，GPU 数量: {world_size}")

    raw_model = model.module if is_ddp else model

    queue_on_gpu = moco_config.get('queue_device', 'cpu').lower() == 'gpu'
    if use_moco:
        min_queue_size, max_queue_size = parse_queue_size_config(moco_config.get('queue_size', 16384))
        emb_dim = model_config['embedding_dim']
        moco_queues = []
        for count in train_dataset.scene_sample_counts:
            adaptive_size = min(count * 2, max_queue_size)
            adaptive_size = max(adaptive_size, min_queue_size)
            adaptive_size = max((adaptive_size // batch_size) * batch_size, batch_size)
            queue = MoCoQueue(queue_size=adaptive_size, embedding_dim=emb_dim)
            if queue_on_gpu:
                queue = queue.to(device)
            moco_queues.append(queue)
        queue_sizes = [queue.queue_size for queue in moco_queues]
        log_info(
            f"MoCo queues: scenes={len(moco_queues)}, device={'gpu' if queue_on_gpu else 'cpu'}, "
            f"size=min/avg/max {min(queue_sizes)}/{sum(queue_sizes) / len(queue_sizes):.1f}/{max(queue_sizes)}"
        )
    else:
        moco_queues = None

    loss_config = supcon_config['supcon']['loss']
    criterion = build_loss_func(
        loss_config=loss_config,
        moco_config=moco_config,
        use_moco=use_moco,
        device=device,
        log_fn=log_info,
    )

    backbone_params = []
    other_params = []
    if use_moco:
        named_parameters = raw_model.query_encoder.named_parameters()
    else:
        named_parameters = raw_model.named_parameters()
    for name, param in named_parameters:
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    backbone_lr_ratio = training_config.get('backbone_lr_ratio', 0.1)
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': float(training_config['learning_rate']) * backbone_lr_ratio},
        {'params': other_params, 'lr': float(training_config['learning_rate'])},
    ], weight_decay=training_config['weight_decay'])

    if training_config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_steps // 3), gamma=0.1)

    start_step = 1
    best_margin = float('-inf')
    last_train_metrics = {'loss': 0.0, 'contrastive_loss': 0.0}
    if args.resume:
        log_info(f"从checkpoint恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if 'global_step' not in checkpoint:
            raise ValueError("当前脚本只支持包含 global_step 的 step-only checkpoint")
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = int(checkpoint['global_step']) + 1
        best_margin = checkpoint.get('best_margin', float('-inf'))
        last_train_metrics['loss'] = float(checkpoint.get('train_loss', 0.0))

        if use_moco and moco_queues is not None and 'moco_queue_state' in checkpoint:
            for queue, state in zip(moco_queues, checkpoint['moco_queue_state']):
                queue.load_state_dict(state)
            log_info("MoCo队列状态已恢复（per-scene）")

    if start_step > total_steps:
        log_info(f"checkpoint 已到达 total_steps: start_step={start_step}, total_steps={total_steps}")

    train_sampler.set_start_step(start_step)

    def build_checkpoint_data(global_step: int, train_metrics: dict, val_metrics: Optional[dict] = None) -> dict:
        checkpoint_data = {
            'global_step': global_step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_metrics.get('loss', 0.0),
            'best_margin': best_margin,
            'config': supcon_config,
        }
        if val_metrics is not None:
            checkpoint_data['val_loss'] = val_metrics['loss']
        if use_moco and moco_queues is not None:
            checkpoint_data['moco_queue_state'] = [queue.state_dict() for queue in moco_queues]
        if is_main and use_wandb and wandb is not None and wandb.run is not None:
            checkpoint_data['wandb_run_id'] = wandb.run.id
        return checkpoint_data

    def save_current_checkpoint(global_step: int, train_metrics: dict, val_metrics: Optional[dict] = None):
        if not is_main:
            return
        checkpoint_data = build_checkpoint_data(global_step, train_metrics, val_metrics)
        torch.save(checkpoint_data, checkpoint_dir / "current_model.pth")

    def save_archive_checkpoint(global_step: int, train_metrics: dict, val_metrics: Optional[dict] = None):
        if not is_main:
            return
        checkpoint_data = build_checkpoint_data(global_step, train_metrics, val_metrics)
        torch.save(checkpoint_data, checkpoint_dir / f"checkpoint_step_{global_step}.pth")

    log_info("开始 step-only 训练...")
    log_interval = int(training_config.get('log_interval', 50))
    eval_interval = int(training_config.get('eval_interval', 1000))
    current_model_interval_steps = int(training_config.get('current_model_interval_steps', 500))
    save_interval_steps = int(training_config.get('save_interval_steps', 1000))
    recent_losses = deque(maxlen=max(1, log_interval))
    scene_step_counts = defaultdict(int)
    train_iter = iter(train_dataloader)
    pbar = tqdm(range(start_step, total_steps + 1), desc="Training", disable=not is_main)

    for global_step in pbar:
        batch = next(train_iter)
        scene_idx = int(batch['scene_idx'][0].item())
        scene_name = train_scene_names[scene_idx]
        scene_step_counts[scene_idx] += 1

        scene_queue = moco_queues[scene_idx] if moco_queues is not None else None
        train_metrics = train_one_batch(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            global_step=global_step,
            use_moco=use_moco,
            moco_queue=scene_queue,
        )
        last_train_metrics = train_metrics

        scheduler.step()
        recent_losses.append(train_metrics['loss'])
        window_loss = sum(recent_losses) / len(recent_losses)
        stage_gate_weights = get_stage_gate_weights(raw_model, use_moco)

        if is_main:
            pbar.set_postfix({
                'loss': f"{train_metrics['loss']:.4f}",
                'scene': scene_idx,
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })

        should_log = global_step == start_step or global_step % log_interval == 0

        val_metrics = None
        val_metrics_per_scene = []
        should_eval = args.use_eval and val_dataloaders and eval_interval > 0 and global_step % eval_interval == 0
        if should_eval:
            for val_scene_name, val_dl in zip(val_scene_names, val_dataloaders):
                metrics = validate(
                    raw_model,
                    val_dl,
                    criterion,
                    device,
                    use_moco=use_moco,
                    loss_temperature=loss_config['supcon']['temperature'],
                    embedding_source=embedding_source,
                    show_progress=False,
                )
                val_metrics_per_scene.append((val_scene_name, metrics))
            if is_main:
                val_metrics = {
                    'loss': sum(m['loss'] for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'margin_pos_sim': sum(m.get('margin_pos_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'margin_neg_sim': sum(m.get('margin_neg_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'margin': sum(m.get('margin', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'knn_accuracy': sum(m.get('knn_accuracy', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'projection_margin_pos_sim': sum(m.get('projection_margin_pos_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'projection_margin_neg_sim': sum(m.get('projection_margin_neg_sim', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'projection_margin': sum(m.get('projection_margin', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                    'projection_knn_accuracy': sum(m.get('projection_knn_accuracy', 0) for _, m in val_metrics_per_scene) / len(val_metrics_per_scene),
                }
                best_margin = max(best_margin, val_metrics.get('margin', float('-inf')))
                log_info(
                    f"Step {global_step} val: loss={val_metrics['loss']:.4f}, "
                    f"Margin={val_metrics.get('margin', 0):.4f}, kNN={val_metrics.get('knn_accuracy', 0):.4f}, "
                    f"ProjMargin={val_metrics.get('projection_margin', 0):.4f}, "
                    f"ProjKNN={val_metrics.get('projection_knn_accuracy', 0):.4f}"
                )

        should_wandb_log = should_log or val_metrics is not None
        if (
            should_wandb_log
            and is_main
            and use_wandb
            and wandb is not None
        ):
            log_dict = {
                'global_step': global_step,
                'train_loss': train_metrics['loss'],
                'train_window_loss': window_loss,
                'train_contrastive_loss': train_metrics.get('contrastive_loss', 0.0),
                'learning_rate': scheduler.get_last_lr()[0],
                'scene_idx': scene_idx,
                'scene_steps/current': scene_step_counts[scene_idx],
            }
            if use_moco:
                if 'pos_loss' in train_metrics:
                    log_dict['train_pos_loss'] = train_metrics['pos_loss']
                if 'neg_loss' in train_metrics:
                    log_dict['train_neg_loss'] = train_metrics['neg_loss']
                if moco_queues is not None:
                    log_dict[f'moco_queue_full/{scene_name}'] = float(moco_queues[scene_idx].is_full())
            for i, count in scene_step_counts.items():
                log_dict[f'scene_steps/{train_scene_names[i]}'] = count
            if stage_gate_weights is not None:
                for key, value in stage_gate_weights.items():
                    log_dict[f'stage_gate/{key}'] = value
            if val_metrics is not None:
                log_dict['val_loss'] = val_metrics['loss']
                log_dict['val_margin_pos_sim'] = val_metrics.get('margin_pos_sim', 0)
                log_dict['val_margin_neg_sim'] = val_metrics.get('margin_neg_sim', 0)
                log_dict['val_margin'] = val_metrics.get('margin', 0)
                log_dict['val_knn_accuracy'] = val_metrics.get('knn_accuracy', 0)
                log_dict['val_projection_margin_pos_sim'] = val_metrics.get('projection_margin_pos_sim', 0)
                log_dict['val_projection_margin_neg_sim'] = val_metrics.get('projection_margin_neg_sim', 0)
                log_dict['val_projection_margin'] = val_metrics.get('projection_margin', 0)
                log_dict['val_projection_knn_accuracy'] = val_metrics.get('projection_knn_accuracy', 0)
            wandb.log(log_dict, step=global_step)

        should_save_current = (
            current_model_interval_steps > 0
            and global_step % current_model_interval_steps == 0
        )
        should_save_archive = (
            save_interval_steps > 0
            and global_step % save_interval_steps == 0
        )
        if should_save_current or should_save_archive:
            if is_ddp:
                dist.barrier()
            if should_save_current:
                save_current_checkpoint(global_step, train_metrics, val_metrics)
            if should_save_archive:
                save_archive_checkpoint(global_step, train_metrics, val_metrics)
            if is_ddp:
                dist.barrier()

    if is_ddp:
        dist.barrier()
    if is_main:
        if total_steps >= start_step:
            save_current_checkpoint(total_steps, last_train_metrics)
            save_archive_checkpoint(total_steps, last_train_metrics)
        final_model_path = checkpoint_dir / "final_model.pth"
        torch.save({'model_state_dict': raw_model.state_dict()}, final_model_path)
        log_info(f"最终模型已保存: {final_model_path}")
    if is_ddp:
        dist.barrier()

    log_info("训练完成！")
    if is_main and use_wandb and wandb is not None:
        wandb.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
