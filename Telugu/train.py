"""
Telugu Training Script (Phase 5)
FastSpeech 2 on IndicTTS Telugu Dataset with:
- Native PyTorch FP16 AMP (mixed precision) for NVIDIA RTX 4050 6GB
- Multi-speaker support (telugu_female = 0, telugu_male = 1)
- Gradient accumulation (effective batch size = 32)
- Gradient clipping (1.0)
- Validation evaluation & loss tracking
- Checkpoint saving (latest & best models)
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, "scratch", "mpl")
os.makedirs(os.path.join(BASE_DIR, "scratch", "mpl"), exist_ok=True)
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "reference_fastspeech2"))

from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from dataset import Dataset
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config_path="Telugu/config.yaml", total_steps=3000, save_step=500, eval_step=250, resume=False):
    with open(os.path.join(BASE_DIR, config_path), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    preprocess_config = {
        "dataset": config["dataset"],
        "path": config["path"],
        "preprocessing": config["preprocessing"]
    }
    model_config = config["model"]
    train_config = config["train"]
    
    # Dataset and DataLoaders
    train_dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    val_dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    # Model
    model = FastSpeech2(preprocess_config, model_config).to(device)
    optimizer = ScheduledOptim(model, train_config, model_config, current_step=0)
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    
    # AMP Scaler for 6GB VRAM optimization
    scaler = torch.amp.GradScaler('cuda', enabled=train_config.get("mixed_precision", True))
    
    ckpt_dir = os.path.join(BASE_DIR, train_config["path"]["ckpt_path"])
    log_dir = os.path.join(BASE_DIR, train_config["path"]["log_path"])
    res_dir = os.path.join(BASE_DIR, "Results", "Telugu")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    
    step = 1
    epoch = 1
    best_val_loss = float("inf")
    
    train_loss_history = []
    val_loss_history = []
    step_history = []
    
    if resume:
        latest_ckpt = os.path.join(ckpt_dir, "latest_model.pth.tar")
        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer._optimizer.load_state_dict(ckpt["optimizer"])
            step = ckpt.get("step", 0) + 1
            if "val_loss" in ckpt:
                best_val_loss = ckpt["val_loss"]
            history_file = os.path.join(res_dir, "loss_history.json")
            if os.path.exists(history_file):
                with open(history_file, "r") as f:
                    old_hist = yaml.safe_load(f)
                    if old_hist:
                        step_history = old_hist.get("steps", [])
                        train_loss_history = old_hist.get("train_loss", [])
                        val_loss_history = old_hist.get("val_loss", [])
            print(f"--> Resumed training from Step {step-1}")
            
    print(f"\n==========================================")
    print(f" Starting FastSpeech 2 Training: Telugu ")
    print(f" Device: {torch.cuda.get_device_name(0)} (CUDA: {torch.cuda.is_available()})")
    print(f" Target Steps: {total_steps} | Batch Size: {batch_size} (Acc: {grad_acc_step})")
    print(f"==========================================\n")
    
    model.train()
    pbar = tqdm(total=total_steps, initial=step-1, desc="Telugu Training")
    
    while step <= total_steps:
        for batchs in train_loader:
            for batch in batchs:
                batch = to_device(batch, device)
                
                with torch.amp.autocast('cuda', enabled=train_config.get("mixed_precision", True)):
                    output = model(*(batch[2:]))
                    losses = loss_fn(batch, output)
                    total_loss = losses[0] / grad_acc_step
                    
                scaler.scale(total_loss).backward()
                
                if step % grad_acc_step == 0:
                    scaler.unscale_(optimizer._optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    scaler.step(optimizer._optimizer)
                    scaler.update()
                    optimizer._update_learning_rate()
                    optimizer.zero_grad()
                    
                train_loss_val = float(losses[0].item())
                pbar.set_postfix({"Loss": f"{train_loss_val:.4f}", "Mel": f"{losses[1].item():.4f}"})
                
                # Validation loop
                if step % eval_step == 0 or step == total_steps:
                    model.eval()
                    val_losses = []
                    v_count = 0
                    max_v_batches = 10
                    with torch.no_grad():
                        for v_batchs in val_loader:
                            for v_batch in v_batchs:
                                v_batch = to_device(v_batch, device)
                                with torch.amp.autocast('cuda', enabled=train_config.get("mixed_precision", True)):
                                    v_out = model(*(v_batch[2:]))
                                    v_l = loss_fn(v_batch, v_out)
                                    val_losses.append(float(v_l[0].item()))
                                v_count += 1
                                if v_count >= max_v_batches:
                                    break
                            if v_count >= max_v_batches:
                                break
                                    
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    val_loss_history.append(avg_val_loss)
                    train_loss_history.append(train_loss_val)
                    step_history.append(step)
                    
                    pbar.write(f"\n[Step {step}] Train Loss: {train_loss_val:.4f} | Val Loss: {avg_val_loss:.4f}")
                    
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                            "step": step,
                            "val_loss": best_val_loss
                        }, os.path.join(ckpt_dir, "best_model.pth.tar"))
                        pbar.write(f"--> Saved New Best Model (Val Loss: {best_val_loss:.4f})")
                        
                    model.train()
                    
                # Save regular checkpoint
                if step % save_step == 0 or step == total_steps:
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                        "step": step
                    }, os.path.join(ckpt_dir, f"checkpoint_{step}.pth.tar"))
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                        "step": step
                    }, os.path.join(ckpt_dir, "latest_model.pth.tar"))
                    
                step += 1
                pbar.update(1)
                if step > total_steps:
                    break
            if step > total_steps:
                break
        epoch += 1
        
    pbar.close()
    
    history = {
        "steps": step_history,
        "train_loss": train_loss_history,
        "val_loss": val_loss_history
    }
    with open(os.path.join(res_dir, "loss_history.json"), "w", encoding="utf-8") as f:
        yaml.safe_dump(history, f)
        
    plt.figure(figsize=(8, 4))
    plt.plot(step_history, train_loss_history, label="Training Loss", color="forestgreen", lw=2)
    plt.title("FastSpeech 2 Telugu - Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "training_loss.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 4))
    plt.plot(step_history, val_loss_history, label="Validation Loss", color="darkorange", lw=2)
    plt.title("FastSpeech 2 Telugu - Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "validation_loss.png"), dpi=150)
    plt.close()
    
    print("\nTraining completed successfully! Graphs saved in Results/Telugu/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval_step", type=int, default=250)
    parser.add_argument("--save_step", type=int, default=500)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()
    main(total_steps=args.steps, eval_step=args.eval_step, save_step=args.save_step, resume=args.resume)

