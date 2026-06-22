import os
import tempfile

# Create and set up temporary directory before any other imports
temp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(temp_dir, exist_ok=True)
# Ensure the directory is accessible and writable
os.chmod(temp_dir, 0o755)
os.environ['TMPDIR'] = temp_dir
os.environ['TEMP'] = temp_dir  # For compatibility
tempfile.tempdir = temp_dir

# Also set multiprocessing temp dir
import multiprocessing
multiprocessing.current_process()._config['tempdir'] = temp_dir

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sat2plan.logic.configuration.config import Model_Configuration, Global_Configuration
from torch.autograd import Variable
from torch import autograd
import pandas as pd
import datetime
from sat2plan.logic.models.ucvgan.model_building import Generator, Discriminator
from sat2plan.logic.loss.loss import r1_penalty, VGGPerceptualLoss, feature_matching_loss
from sat2plan.logic.loss.metrics import ImageQualityMetrics
from sat2plan.scripts.flow import save_results, save_model, load_model
from sat2plan.logic.preproc.dataset import Satellite2Map_Data
import shutil

class UCVGan(nn.Module):
    def __init__(self, rank, world_size):
        super(UCVGan, self).__init__()
        try:
            # Use the already created temporary directory and ensure it still exists
            self.temp_dir = temp_dir
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir, exist_ok=True)
                os.chmod(self.temp_dir, 0o755)
            
            # Import des paramètres globaux
            self.G_CFG = Global_Configuration()
            self.n_cpu = self.G_CFG.n_cpu
            self.rank = rank
            self.world_size = world_size
            self.train_dir = f"{self.G_CFG.train_dir}/{self.G_CFG.data_bucket}"
            self.val_dir = f"{self.G_CFG.val_dir}/{self.G_CFG.data_bucket}"
            self.image_size = self.G_CFG.image_size
            self.batch_size = self.G_CFG.batch_size
            self.n_epochs = self.G_CFG.n_epochs
            self.sample_interval = self.G_CFG.sample_interval
            self.num_workers = self.G_CFG.num_workers
            self.l1_lambda = 100.0  # Augmenté pour donner plus d'importance à la reconstruction
            self.lambda_gp = 10.0  # Augmenté pour une meilleure régularisation
            self.load_model = self.G_CFG.load_model
            self.save_model_bool = self.G_CFG.save_model
            self.checkpoint_disc = self.G_CFG.checkpoint_disc
            self.checkpoint_gen = self.G_CFG.checkpoint_gen

            # Import des hyperparamètres du modèle
            self.M_CFG = Model_Configuration()
            self.learning_rate_D = self.M_CFG.learning_rate_D
            self.learning_rate_G = self.M_CFG.learning_rate_G
            self.beta1 = self.M_CFG.beta1
            self.beta2 = self.M_CFG.beta2

            # Warmup parameters (avant setup_device pour éviter les erreurs d'initialisation)
            self.warmup_epochs = 10
            self.warmup_factor = 0.05

            # Setup device et distributed
            self.setup_device()
            if self.cuda:
                torch.cuda.set_device(self.rank)
                
            # Loading Data
            self.dataloading()

            # Création des models, optimizers, losses
            self.create_models()

            # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
            if self.cuda:
                torch.backends.cudnn.benchmark = True
                
            self.train()
            
        except Exception as e:
            print(f"Error in process {rank}: {str(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            # Make sure to cleanup even if initialization fails
            if hasattr(self, 'cleanup'):
                self.cleanup()
            raise  # Re-raise the exception after cleanup

    def setup_device(self):
        try:
            # Diagnostic PyTorch/CUDA
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Compiled with CUDA: {torch.backends.cudnn.enabled}")
            
            self.cuda = torch.cuda.is_available()
            if not self.cuda:
                raise RuntimeError("CUDA is required but not available. Please check your PyTorch installation.")
            
            print(f"CUDA is available - Using GPU {self.rank}")
            self.device = torch.device(f'cuda:{self.rank}')
            
            # Force CUDA initialization and set device
            torch.cuda.init()
            torch.cuda.set_device(self.rank)
            
            # Force some tensor operations to ensure CUDA is initialized
            dummy_tensor = torch.ones(1, device=self.device)
            dummy_tensor = dummy_tensor * 2
            del dummy_tensor
            
            # Configuration CUDA optimisée pour H100
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            
            # Enable optimized SDPA backends when available (PyTorch >= 2.0)
            try:
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
                if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'enable_math_sdp'):
                    torch.backends.cuda.enable_math_sdp(True)
            except Exception as _e:
                print(f"Warning: could not enable SDPA backends: {_e}")
            
            # Enable channels_last memory format for better performance
            torch.backends.cudnn.benchmark_limit = 10
            
            # Configuration du processus distribué
            if self.world_size > 1:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                dist.init_process_group(
                    "nccl", 
                    rank=self.rank, 
                    world_size=self.world_size,
                    timeout=datetime.timedelta(minutes=30)
                )
            
            # Pré-allocation de la mémoire CUDA avec une stratégie plus agressive
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(self.rank).total_memory
            reserved_memory = int(total_memory * 0.95)
            torch.cuda.set_per_process_memory_fraction(0.95, self.rank)
            
            print(f"GPU {self.rank}: Reserved {reserved_memory/1024**3:.1f}GB of VRAM")
            print(f"CUDA Device: {torch.cuda.get_device_name(self.rank)}")
            print(f"CUDA Capability: {torch.cuda.get_device_capability(self.rank)}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device properties: {torch.cuda.get_device_properties(self.rank)}")
            
            # Verify CUDA is working
            test_tensor = torch.ones(2, 2, device=self.device, dtype=torch.float32)
            print(f"Test tensor device: {test_tensor.device}")
            
        except Exception as e:
            print(f"Error in setup_device for process {self.rank}: {str(e)}")
            raise

    def cleanup(self):
        try:
            if self.cuda and self.world_size > 1:
                dist.barrier()  # Ensure all processes reach this point
                dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        finally:
            # Clean up temporary directory
            # Only rank 0 attempts to remove the shared temp dir to avoid races
            if getattr(self, 'rank', 0) == 0 and hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"Warning: Could not remove temporary directory: {e}")

    def __del__(self):
        self.cleanup()

    # Load datasets from train/val directories
    def dataloading(self):
        os.makedirs("images", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        # Create datasets
        self.train_dataset = Satellite2Map_Data(root=self.train_dir, image_size=self.image_size)
        self.val_dataset = Satellite2Map_Data(root=self.val_dir, image_size=self.image_size)

        # Create samplers for distributed training
        if self.cuda and self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        # Optimized DataLoader configuration for H100
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'pin_memory_device': f'cuda:{self.rank}' if self.cuda else '',
            'persistent_workers': True,
            'prefetch_factor': 2,
            'drop_last': True
        }

        # Create dataloaders with optimized settings
        self.train_dl = DataLoader(
            self.train_dataset,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            **dataloader_kwargs
        )

        self.val_dl = DataLoader(
            self.val_dataset,
            shuffle=False,
            sampler=val_sampler,
            **dataloader_kwargs
        )

        print(f"Train Data Loaded - {len(self.train_dataset)} images")
        print(f"Validation Data Loaded - {len(self.val_dataset)} images")
        print(f"DataLoader workers: {self.num_workers}, prefetch factor: 2")

        return

    # Create models, optimizers ans losses

    def create_models(self):
        # Initialize models and force them to GPU
        self.netD = Discriminator(in_channels=3)
        self.netG = Generator(in_channels=3)
        
        # Explicitly move models to GPU and verify
        self.netD = self.netD.to(self.device)
        self.netG = self.netG.to(self.device)
        print(f"Generator device: {next(self.netG.parameters()).device}")
        print(f"Discriminator device: {next(self.netD.parameters()).device}")
        
        self.starting_epoch = 0

        # ---- Objectif: GAN conditionnel hinge + R1 + reconstruction ----
        # D et G mis à jour 1:1 (n_critic=1). La hinge loss + R1 est stable et
        # ne nécessite ni label smoothing ni gradient penalty WGAN.
        self.n_critic = 1
        self.adv_weight = 1.0      # poids du terme adversarial (hinge)
        self.l1_lambda = 50.0      # reconstruction pixel L1
        self.perc_lambda = 10.0    # perceptual VGG (contours nets)
        self.fm_lambda = 10.0      # feature matching (stabilité + détail)
        self.r1_gamma = 10.0       # force de la régularisation R1
        self.d_reg_interval = 16   # R1 "lazy" (StyleGAN2) tous les 16 pas
        self.max_grad_norm = 1.0   # clipping raisonnable (au lieu de 0.2)

        # EMA du générateur — UNIQUEMENT pour l'échantillonnage / sauvegarde,
        # jamais pour entraîner le discriminateur.
        self.beta_smoothing = 0.999
        self.generator_ema = Generator(in_channels=3).to(self.device)
        self.generator_ema.load_state_dict(self.netG.state_dict())
        for param in self.generator_ema.parameters():
            param.requires_grad = False

        # Compile only the generator (PyTorch 2.0+) BEFORE DDP wrapping.
        # Le discriminateur reste en eager: la régularisation R1 fait un
        # double-backward à travers D, ce qui peut casser sous inductor.
        if hasattr(torch, 'compile'):
            try:
                print("Compiling generator with torch.compile()...")
                # Mode par défaut: compile rapide et kernels fiables. max-autotune
                # générait des kernels Inductor fautifs (illegal memory access) sur
                # ce générateur (240M params + checkpointing + channels_last + ViT).
                self.netG = torch.compile(self.netG)
                print("Generator successfully compiled")
            except Exception as e:
                print(f"Warning: Model compilation failed: {e}")
                print("Continuing without compilation")

        # Setup distributed training if using CUDA (after possible compilation)
        if self.cuda and self.world_size > 1:
            # Convert BatchNorm to SyncBatchNorm before DDP
            self.netG = nn.SyncBatchNorm.convert_sync_batchnorm(self.netG)
            self.netD = nn.SyncBatchNorm.convert_sync_batchnorm(self.netD)

            ddp_kwargs = {
                'device_ids': [self.rank],
                'output_device': self.rank,
                'find_unused_parameters': False,
                'gradient_as_bucket_view': True,
                'static_graph': True
            }
            self.netG = nn.parallel.DistributedDataParallel(self.netG, **ddp_kwargs)
            self.netD = nn.parallel.DistributedDataParallel(self.netD, **ddp_kwargs)
            print(f"Models wrapped in DistributedDataParallel on GPU {self.rank}")

        # Learning rates relevés au régime pix2pix standard. La config était à
        # 1e-5 (~20x trop faible, le générateur n'apprenait quasiment pas).
        self.lr_G = 2e-4
        self.lr_D = 2e-4
        self.OptimizerG = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.lr_G,
            betas=(self.beta1, self.beta2),
            fused=True
        )
        self.OptimizerD = torch.optim.Adam(
            self.netD.parameters(),
            lr=self.lr_D,
            betas=(self.beta1, self.beta2),
            fused=True  # fused Adam pour la perf sur GPU récents
        )

        # Scheduler cosine avec warmup amélioré
        import math
        
        def cosine_warmup_scheduler(epoch, warmup_epochs, total_epochs, min_lr=1e-6):
            """Cosine annealing avec warmup progressif"""
            if epoch < warmup_epochs:
                # Warmup progressif
                return self.warmup_factor + (1.0 - self.warmup_factor) * (epoch / warmup_epochs)
            else:
                # Cosine annealing après warmup
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                return max(cosine_factor, min_lr)
        
        self.schedulerD = torch.optim.lr_scheduler.LambdaLR(
            self.OptimizerD, 
            lambda epoch: cosine_warmup_scheduler(epoch, self.warmup_epochs, self.n_epochs)
        )
        self.schedulerG = torch.optim.lr_scheduler.LambdaLR(
            self.OptimizerG, 
            lambda epoch: cosine_warmup_scheduler(epoch, self.warmup_epochs, self.n_epochs)
        )
        
        # Load model and optimizer states if requested
        if self.load_model:
            try:
                model_and_optimizer, epoch = load_model()
                self.netG.load_state_dict(model_and_optimizer['gen_state_dict'])
                self.netD.load_state_dict(model_and_optimizer['disc_state_dict'])
                self.OptimizerG.load_state_dict(model_and_optimizer['gen_opt_optimizer_state_dict'])
                self.OptimizerD.load_state_dict(model_and_optimizer['gen_disc_optimizer_state_dict'])
                self.starting_epoch = epoch
                print(f"Successfully loaded model from epoch {epoch}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting from scratch")
                self.starting_epoch = 0

        # Losses. On utilise BF16 (Blackwell) via autocast: pas de GradScaler
        # nécessaire (le bf16 a la même plage dynamique que le fp32).
        self.L1_Loss = nn.L1Loss().to(self.device)
        self.perceptual = VGGPerceptualLoss().to(self.device)
        self.perceptual.eval()

        # Métriques de qualité (FID/LPIPS/SSIM) — uniquement rank 0 (validation).
        self.metrics = ImageQualityMetrics(self.device) if self.rank == 0 else None
        
        # Initialize loss history lists
        self.Gen_loss = []
        self.Dis_loss = []
        self.val_Dis_loss = []
        self.val_Gen_loss = []
        self.val_Gen_fake_loss = []
        self.val_Gen_L1_loss = []
        self.val_D_real_loss = []
        self.val_D_fake_loss = []

        # Early stopping parameters
        self.best_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        self.eps = 1e-8

        return

    # Train & save models
    def train(self):
        try:
            # Setup directories and logging
            if self.rank == 0:
                os.makedirs("save", exist_ok=True)
                os.makedirs("save/loss", exist_ok=True)
                os.makedirs("save/checkpoints", exist_ok=True)
                os.makedirs("images", exist_ok=True)
                params_json = open("params.json", mode="w", encoding='UTF-8')
                
                # Log model parameters
                pytorch_total_params_G = sum(p.numel() for p in self.netG.parameters() if p.requires_grad)
                pytorch_total_params_D = sum(p.numel() for p in self.netD.parameters() if p.requires_grad)
                print("Total params in Generator:", pytorch_total_params_G)
                print("Total params in Discriminator:", pytorch_total_params_D)

            # Pour le calcul du throughput
            import time
            batch_times = []
            
            for epoch in range(self.starting_epoch, self.n_epochs):

                if self.world_size > 1:
                    self.train_dl.sampler.set_epoch(epoch)
                
                epoch_start_time = time.time()
                total_images = 0
                epoch_g_loss = 0
                epoch_d_loss = 0
                num_batches = 0
                
                # Training phase
                self.netG.train()
                self.netD.train()
                
                for idx, (x, y, to_save) in enumerate(self.train_dl):
                    batch_start_time = time.time()
                    num_batches += 1
                    current_batch_size = x.size(0)
                    
                    # Move data to appropriate device
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    total_images += current_batch_size

                    autocast = lambda: torch.autocast('cuda', dtype=torch.bfloat16)

                    ############## Train Discriminator (hinge + R1) ##############
                    for p in self.netD.parameters():
                        p.requires_grad = True
                    self.OptimizerD.zero_grad(set_to_none=True)

                    # Faux générés par le générateur COURANT (jamais l'EMA), détachés.
                    with torch.no_grad(), autocast():
                        y_fake = self.netG(x)
                    y_fake = y_fake.detach()

                    with autocast():
                        D_real = self.netD(x, y)
                        D_fake = self.netD(x, y_fake)
                        # Hinge loss
                        D_loss = (torch.relu(1.0 - D_real).mean()
                                  + torch.relu(1.0 + D_fake).mean())

                    D_loss.backward()

                    # Régularisation R1 "lazy" (tous les d_reg_interval pas), en fp32
                    r1 = torch.zeros((), device=self.device)
                    if idx % self.d_reg_interval == 0:
                        y_real = y.detach().requires_grad_(True)
                        D_real_r1 = self.netD(x, y_real)
                        r1 = r1_penalty(D_real_r1, y_real)
                        (self.r1_gamma * 0.5 * r1 * self.d_reg_interval).backward()

                    torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=self.max_grad_norm)
                    self.OptimizerD.step()

                    ############## Train Generator ##############
                    for p in self.netD.parameters():
                        p.requires_grad = False
                    self.OptimizerG.zero_grad(set_to_none=True)

                    with autocast():
                        y_fake = self.netG(x)
                        D_fake, feats_fake = self.netD(x, y_fake, return_features=True)
                        with torch.no_grad():
                            _, feats_real = self.netD(x, y, return_features=True)

                        G_adv = -D_fake.mean()                       # hinge generator
                        L1 = self.L1_Loss(y_fake, y)
                        perc = self.perceptual(y_fake, y)
                        fm = feature_matching_loss(feats_fake, feats_real)
                        G_loss = (self.adv_weight * G_adv
                                  + self.l1_lambda * L1
                                  + self.perc_lambda * perc
                                  + self.fm_lambda * fm)

                    G_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.max_grad_norm)
                    self.OptimizerG.step()

                    # Mise à jour de l'EMA (sert uniquement à l'échantillonnage)
                    with torch.no_grad():
                        for ema_param, current_param in zip(self.generator_ema.parameters(), self.netG.parameters()):
                            ema_param.data.mul_(self.beta_smoothing).add_(
                                current_param.data, alpha=(1 - self.beta_smoothing)
                            )

                    # Accumulate losses
                    epoch_d_loss += D_loss.item()
                    epoch_g_loss += G_loss.item()

                    # Calculate and log throughput
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)
                    if len(batch_times) > 50:
                        batch_times.pop(0)
                    avg_time = sum(batch_times) / len(batch_times)
                    images_per_sec = current_batch_size / avg_time

                    if self.rank == 0 and idx % 10 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D: hinge %.3f | r1 %.3f] [G: adv %.3f | L1 %.3f | perc %.3f | fm %.3f] [%.1f img/s]"
                            % (epoch+1, self.n_epochs, idx+1, len(self.train_dl),
                               D_loss.item(), r1.item(), G_adv.item(), L1.item(), perc.item(), fm.item(),
                               images_per_sec))

                        if idx % 50 == 0:
                            with torch.no_grad(), autocast():
                                y_sample = self.generator_ema(x[:4])
                                concatenated_images = torch.cat((x[:4], y_sample, y[:4]), dim=2)
                            save_image(concatenated_images.float(), f"images/{str(epoch) + '-' + str(idx)}.png", nrow=3, normalize=True)

                # Synchronize losses across GPUs
                if self.world_size > 1:
                    loss_tensor = torch.tensor([epoch_d_loss, epoch_g_loss], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    epoch_d_loss = loss_tensor[0].item() / self.world_size
                    epoch_g_loss = loss_tensor[1].item() / self.world_size

                if self.rank == 0:
                    # Calculate average epoch losses
                    avg_epoch_d_loss = epoch_d_loss / num_batches
                    avg_epoch_g_loss = epoch_g_loss / num_batches

                    # Calculate epoch throughput
                    epoch_time = time.time() - epoch_start_time
                    epoch_throughput = total_images / epoch_time
                    print(f"Epoch {epoch+1} completed. Average throughput: {epoch_throughput:.2f} images/s")

                    # Optional: write detailed per-batch losses; currently disabled to avoid heavy I/O
                    # If needed, collect and write aggregated metrics only

                    # Validation and model saving
                    print("-- Validation Test --")
                    self.validation()
                    m = self.last_metrics
                    # Sélection du modèle: LPIPS (perceptuel appris, le mieux corrélé
                    # à la qualité perçue) si dispo, sinon repli sur L1 + perceptual VGG.
                    if 'lpips' in m:
                        val_score = m['lpips']
                        score_name = "LPIPS"
                    else:
                        val_score = self.val_Gen_L1_loss[-1] + self.val_Gen_fake_loss[-1]
                        score_name = "L1+perc"
                    metric_str = "  ".join(
                        f"{k.upper()}: {v:.4f}" for k, v in m.items()
                    ) or "—"
                    print(f"\n{'='*60}")
                    print(f"EPOCH {epoch+1}/{self.n_epochs} VALIDATION SUMMARY")
                    print(f"{'='*60}")
                    print(f"📐 Qualité: {metric_str}")
                    print(f"📊 L1: {self.val_Gen_L1_loss[-1]:.4f}  |  Perceptual: {self.val_Gen_fake_loss[-1]:.4f}  |  D hinge: {self.val_Dis_loss[-1]:.4f}")
                    print(f"🏅 Score sélection ({score_name}): {val_score:.4f}  (best: {self.best_loss:.4f})")
                    print(f"🎯 LR D/G: {self.OptimizerD.param_groups[0]['lr']:.2e} / {self.OptimizerG.param_groups[0]['lr']:.2e}")
                    print(f"⚡ Performance: {epoch_throughput:.2f} images/sec")
                    print(f"{'='*60}\n")

                    # Early stopping check
                    if val_score < self.best_loss:
                        self.best_loss = val_score
                        self.patience_counter = 0
                        if self.save_model_bool:
                            save_model(
                                models={'gen': self.netG, 'disc': self.netD},
                                optimizers={'gen_opt': self.OptimizerG, 'gen_disc': self.OptimizerD},
                                suffix=f"-best-{epoch}"
                            )
                    else:
                        self.patience_counter += 1

                    if epoch % 10 == 0 and self.save_model_bool:
                        save_model(
                            models={'gen': self.netG, 'disc': self.netD},
                            optimizers={'gen_opt': self.OptimizerG, 'gen_disc': self.OptimizerD},
                            suffix=f"-{epoch}"
                        )
                        save_results(params=self.M_CFG, metrics=dict(
                            Gen_loss=avg_epoch_g_loss,
                            Dis_loss=avg_epoch_d_loss,
                            Val_Gen_loss=self.val_Gen_loss[-1],
                            Val_Dis_loss=self.val_Dis_loss[-1],
                            **self.last_metrics
                        ))

                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                # Update learning rates at the end of the epoch
                self.schedulerD.step()
                self.schedulerG.step()

            if self.rank == 0:
                # Final save
                save_model(
                    models={'gen': self.netG, 'disc': self.netD},
                    optimizers={'gen_opt': self.OptimizerG, 'gen_disc': self.OptimizerD},
                    suffix=f"-final"
                )
                save_results(params=self.M_CFG, metrics=dict(
                    Gen_loss=avg_epoch_g_loss,
                    Dis_loss=avg_epoch_d_loss,
                    Val_Gen_loss=self.val_Gen_loss[-1],
                    Val_Dis_loss=self.val_Dis_loss[-1],
                    **getattr(self, 'last_metrics', {})
                ))
                params_json.close()

        except Exception as e:
            print(f"Error during training: {e}")
            raise
        finally:
            self.cleanup()

    # Test du modèle sur le set de validation
    def validation(self):
        # Passage en mode eval
        self.netG.eval()
        self.netD.eval()

        sum_D_loss = 0.0      # hinge D (monitoring)
        sum_L1 = 0.0          # qualité: reconstruction
        sum_perc = 0.0        # qualité: perceptuel VGG
        num_batches = 0

        if self.metrics is not None:
            self.metrics.reset()

        with torch.no_grad():
            for idx, (x, y, _) in enumerate(self.val_dl):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                # Forward sous bf16; on échantillonne avec l'EMA (le modèle servi)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    y_fake = self.generator_ema(x)
                    D_real = self.netD(x, y)
                    D_fake = self.netD(x, y_fake)
                    D_loss = (torch.relu(1.0 - D_real).mean()
                              + torch.relu(1.0 + D_fake).mean())
                    L1 = self.L1_Loss(y_fake, y)
                    perc = self.perceptual(y_fake, y)

                # Métriques en fp32, hors autocast (réseaux Inception/VGG internes)
                if self.metrics is not None:
                    self.metrics.update(y_fake, y)

                sum_D_loss += D_loss.item()
                sum_L1 += L1.item()
                sum_perc += perc.item()
                num_batches += 1

        avg_D_loss = sum_D_loss / num_batches
        avg_L1 = sum_L1 / num_batches
        avg_perc = sum_perc / num_batches

        self.last_metrics = self.metrics.compute() if self.metrics is not None else {}

        # Stocker les résultats. On réutilise les listes existantes:
        #   val_Gen_L1_loss  -> L1 brute (qualité)
        #   val_Gen_fake_loss -> perceptual (qualité)
        #   val_Dis_loss     -> hinge D (monitoring)
        self.val_Dis_loss.append(avg_D_loss)
        self.val_Gen_loss.append(avg_L1 + avg_perc)
        self.val_Gen_fake_loss.append(avg_perc)
        self.val_Gen_L1_loss.append(avg_L1)
        self.val_D_real_loss.append(avg_D_loss)
        self.val_D_fake_loss.append(avg_D_loss)

        # Retour en mode train
        self.netG.train()
        self.netD.train()
