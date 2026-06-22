"""Entraîneur de diffusion conditionnelle sat→carte (style Palette).

- v-prediction + Min-SNR-γ (cf. scheduler.py)
- BF16 (Blackwell) sans GradScaler, channels_last, torch.compile
- EMA des poids pour l'échantillonnage (essentiel en diffusion)
- Métriques de qualité FID/LPIPS/SSIM sur échantillons DDIM (générateur EMA)
- Checkpointing local dédié (flow.save_model est spécifique aux GAN)
"""

import os
import copy
import time
import glob
import tempfile
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from sat2plan.logic.configuration.config import Model_Configuration, Global_Configuration
from sat2plan.logic.models.diffusion.unet import ConditionalUNet
from sat2plan.logic.models.diffusion.scheduler import GaussianDiffusion
from sat2plan.logic.loss.metrics import ImageQualityMetrics
from sat2plan.logic.preproc.dataset import Satellite2Map_Data


def _unwrap(model):
    """Récupère le module nu derrière torch.compile / DDP."""
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "module"):
        model = model.module
    return model


class EMA:
    """Moyenne mobile exponentielle des poids, pour l'échantillonnage."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema = copy.deepcopy(_unwrap(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = _unwrap(model)
        for ep, p in zip(self.ema.parameters(), src.parameters()):
            ep.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for eb, b in zip(self.ema.buffers(), src.buffers()):
            eb.copy_(b)


class DiffusionTrainer:
    # Échantillonnage / éval
    SAMPLE_STEPS = 50          # pas DDIM pour l'échantillonnage
    EVAL_SAMPLE_BATCHES = 4    # nb de batches val échantillonnés pour les métriques

    def __init__(self, rank, world_size):
        try:
            self.temp_dir = os.path.join(os.getcwd(), "tmp")
            os.makedirs(self.temp_dir, exist_ok=True)
            os.environ["TMPDIR"] = self.temp_dir
            tempfile.tempdir = self.temp_dir

            self.G_CFG = Global_Configuration()
            self.rank = rank
            self.world_size = world_size
            self.train_dir = f"{self.G_CFG.train_dir}/{self.G_CFG.data_bucket}"
            self.val_dir = f"{self.G_CFG.val_dir}/{self.G_CFG.data_bucket}"
            self.image_size = self.G_CFG.image_size
            # La diffusion 256px est gourmande: on plafonne le batch pour éviter l'OOM.
            self.batch_size = min(self.G_CFG.batch_size, 16)
            self.n_epochs = self.G_CFG.n_epochs
            self.num_workers = self.G_CFG.num_workers
            self.load_model = self.G_CFG.load_model
            self.save_model_bool = self.G_CFG.save_model

            self.lr = 1e-4  # lr saine pour un UNet de diffusion (l'ancien 1e-3 était trop élevé)
            self.M_CFG = Model_Configuration()
            self.beta1, self.beta2 = 0.9, 0.999

            self.setup_device()
            if self.cuda:
                torch.cuda.set_device(self.rank)

            self.dataloading()
            self.create_model()
            self.train()

        except Exception as e:
            print(f"Error in process {rank}: {str(e)}")
            import traceback
            traceback.print_exc()
            if hasattr(self, "cleanup"):
                self.cleanup()
            raise

    def setup_device(self):
        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            raise RuntimeError("CUDA requis mais indisponible.")
        self.device = torch.device(f"cuda:{self.rank}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if self.world_size > 1:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size,
                                    timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_per_process_memory_fraction(0.95, self.rank)
        print(f"GPU {self.rank}: {torch.cuda.get_device_name(self.rank)}")

    def cleanup(self):
        try:
            if self.cuda and self.world_size > 1:
                dist.barrier()
                dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: cleanup: {e}")

    def dataloading(self):
        os.makedirs("images", exist_ok=True)
        self.train_dataset = Satellite2Map_Data(root=self.train_dir, image_size=self.image_size)
        self.val_dataset = Satellite2Map_Data(root=self.val_dir, image_size=self.image_size)

        if self.cuda and self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        else:
            train_sampler = val_sampler = None

        kwargs = dict(batch_size=self.batch_size, num_workers=self.num_workers,
                      pin_memory=True, persistent_workers=self.num_workers > 0,
                      prefetch_factor=2 if self.num_workers > 0 else None, drop_last=True)
        self.train_dl = DataLoader(self.train_dataset, shuffle=(train_sampler is None),
                                   sampler=train_sampler, **kwargs)
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, sampler=val_sampler, **kwargs)
        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | batch={self.batch_size}")

    def create_model(self):
        self.model = ConditionalUNet(
            in_channels=3, cond_channels=3, out_channels=3,
            base=64, ch_mult=(1, 2, 2, 4, 4), num_res_blocks=2,
            attn_resolutions=(32, 16), time_dim=256, image_size=self.image_size,
        ).to(self.device, memory_format=torch.channels_last)

        self.diffusion = GaussianDiffusion(
            timesteps=1000, schedule="cosine", prediction_type="v",
            min_snr_gamma=5.0, device=self.device,
        )

        total = sum(p.numel() for p in self.model.parameters())
        print(f"Diffusion UNet — {total/1e6:.1f}M params | v-pred + Min-SNR | {self.diffusion.timesteps} steps")

        # EMA AVANT compile/DDP (copie du module nu).
        self.ema = EMA(self.model, decay=0.9999)

        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("Modèle compilé (torch.compile)")
            except Exception as e:
                print(f"Warning: compile échoué: {e}")

        if self.cuda and self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.rank], output_device=self.rank)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                           betas=(self.beta1, self.beta2), weight_decay=0.01)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, epochs=self.n_epochs,
            steps_per_epoch=len(self.train_dl), pct_start=0.05,
            div_factor=10, final_div_factor=100)

        self.metrics = ImageQualityMetrics(self.device) if self.rank == 0 else None
        self.last_metrics = {}

        self.starting_epoch = 0
        self.best_score = float("inf")
        self.patience, self.patience_counter = 20, 0
        if self.load_model:
            self._load_latest()

    # ---------------- entraînement ----------------
    def train(self):
        try:
            if self.rank == 0:
                os.makedirs("save/checkpoints", exist_ok=True)
                os.makedirs("images", exist_ok=True)

            batch_times = []
            for epoch in range(self.starting_epoch, self.n_epochs):
                if self.world_size > 1:
                    self.train_dl.sampler.set_epoch(epoch)
                epoch_start, total_images, epoch_loss, num_batches = time.time(), 0, 0.0, 0

                self.model.train()
                for idx, (cond, target, _) in enumerate(self.train_dl):
                    t0 = time.time()
                    num_batches += 1
                    bs = cond.size(0)
                    total_images += bs
                    cond = cond.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                    target = target.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)

                    t = torch.randint(0, self.diffusion.timesteps, (bs,), device=self.device).long()

                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = self.diffusion.p_losses(self.model, target, t, cond)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.ema.update(self.model)

                    epoch_loss += loss.item()

                    batch_times.append(time.time() - t0)
                    if len(batch_times) > 50:
                        batch_times.pop(0)
                    ips = bs / (sum(batch_times) / len(batch_times))

                    if self.rank == 0 and idx % 10 == 0:
                        print(f"[Epoch {epoch+1}/{self.n_epochs}] [Batch {idx+1}/{len(self.train_dl)}] "
                              f"[loss {loss.item():.4f}] [{ips:.1f} img/s] "
                              f"[lr {self.optimizer.param_groups[0]['lr']:.2e}]")
                        if idx % 100 == 0:
                            self._save_samples(cond, target, epoch, idx)

                if self.world_size > 1:
                    lt = torch.tensor([epoch_loss], device=self.device)
                    dist.all_reduce(lt, op=dist.ReduceOp.SUM)
                    epoch_loss = lt[0].item() / self.world_size

                if self.rank == 0:
                    avg_loss = epoch_loss / num_batches
                    thru = total_images / (time.time() - epoch_start)
                    print(f"Epoch {epoch+1} terminé — loss {avg_loss:.4f} — {thru:.1f} img/s")

                    val_loss = self._validate()
                    m = self.last_metrics
                    if "lpips" in m:
                        score, score_name = m["lpips"], "LPIPS"
                    else:
                        score, score_name = val_loss, "val_loss"
                    metric_str = "  ".join(f"{k.upper()}: {v:.4f}" for k, v in m.items()) or "—"
                    print(f"📐 Qualité: {metric_str} | denoise val: {val_loss:.4f} | "
                          f"score({score_name}): {score:.4f} (best {self.best_score:.4f})")

                    if score < self.best_score:
                        self.best_score = score
                        self.patience_counter = 0
                        if self.save_model_bool:
                            self._save_ckpt(f"-best", epoch)
                    else:
                        self.patience_counter += 1

                    if epoch % 10 == 0 and self.save_model_bool:
                        self._save_ckpt(f"-{epoch}", epoch)

                    if self.patience_counter >= self.patience:
                        print(f"Early stopping après {epoch+1} epochs")
                        break

            if self.rank == 0 and self.save_model_bool:
                self._save_ckpt("-final", self.n_epochs)
        finally:
            self.cleanup()

    # ---------------- validation ----------------
    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        if self.metrics is not None:
            self.metrics.reset()

        sum_loss, num = 0.0, 0
        for idx, (cond, target, _) in enumerate(self.val_dl):
            cond = cond.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
            target = target.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
            t = torch.randint(0, self.diffusion.timesteps, (cond.size(0),), device=self.device).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.diffusion.p_losses(self.model, target, t, cond)
            sum_loss += loss.item()
            num += 1

            # Métriques de qualité sur quelques batches échantillonnés (générateur EMA).
            if self.metrics is not None and idx < self.EVAL_SAMPLE_BATCHES:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake = self.diffusion.ddim_sample(
                        self.ema.ema, cond, cond.shape, self.device, steps=self.SAMPLE_STEPS)
                self.metrics.update(fake, target)

        self.last_metrics = self.metrics.compute() if self.metrics is not None else {}
        self.model.train()
        return sum_loss / max(num, 1)

    # ---------------- I/O ----------------
    @torch.no_grad()
    def _save_samples(self, cond, target, epoch, idx):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            samples = self.diffusion.ddim_sample(
                self.ema.ema, cond[:4], (min(4, cond.size(0)), 3, self.image_size, self.image_size),
                self.device, steps=self.SAMPLE_STEPS)
        grid = torch.cat((cond[:4], samples, target[:4]), dim=2).float()
        save_image(grid, f"images/diff-{epoch}-{idx}.png", nrow=4, normalize=True)

    def _save_ckpt(self, suffix, epoch):
        path = f"save/checkpoints/diffusion{suffix}.pt"
        tmp = path + ".tmp"
        torch.save({
            "model_state_dict": _unwrap(self.model).state_dict(),
            "ema_state_dict": self.ema.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }, tmp)
        os.replace(tmp, path)
        print(f"💾 Checkpoint: {path}")

    def _load_latest(self):
        paths = glob.glob("save/checkpoints/diffusion*.pt")
        if not paths:
            print("Aucun checkpoint diffusion trouvé, départ de zéro.")
            return
        path = max(paths, key=os.path.getmtime)
        ckpt = torch.load(path, map_location=self.device)
        _unwrap(self.model).load_state_dict(ckpt["model_state_dict"])
        self.ema.ema.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.starting_epoch = ckpt.get("epoch", 0)
        print(f"Checkpoint chargé: {path} (epoch {self.starting_epoch})")
