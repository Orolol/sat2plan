import os
import tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sat2plan.logic.configuration.config import Model_Configuration, Global_Configuration
from sat2plan.logic.models.diffusion.unet import ConditionalUNet
from sat2plan.logic.models.diffusion.scheduler import DDPMScheduler
from sat2plan.scripts.flow import save_results, save_model, load_model
from sat2plan.logic.preproc.dataset import Satellite2Map_Data
import datetime
import pandas as pd

class DiffusionTrainer:
    def __init__(self, rank, world_size):
        try:
            # Create and set up temporary directory
            self.temp_dir = os.path.join(os.getcwd(), 'tmp')
            os.makedirs(self.temp_dir, exist_ok=True)
            os.environ['TMPDIR'] = self.temp_dir
            tempfile.tempdir = self.temp_dir

            # Import global parameters
            self.G_CFG = Global_Configuration()
            self.rank = rank
            self.world_size = world_size
            self.train_dir = f"{self.G_CFG.train_dir}/{self.G_CFG.data_bucket}"
            self.val_dir = f"{self.G_CFG.val_dir}/{self.G_CFG.data_bucket}"
            self.image_size = self.G_CFG.image_size
            self.batch_size = self.G_CFG.batch_size
            self.n_epochs = self.G_CFG.n_epochs
            self.sample_interval = self.G_CFG.sample_interval
            self.num_workers = self.G_CFG.num_workers
            self.load_model = self.G_CFG.load_model
            self.save_model_bool = self.G_CFG.save_model

            # Import model hyperparameters
            self.M_CFG = Model_Configuration()
            self.learning_rate = self.M_CFG.learning_rate_ViT
            self.beta1 = self.M_CFG.beta1
            self.beta2 = self.M_CFG.beta2

            # Setup device and distributed training
            self.setup_device()
            if self.cuda:
                torch.cuda.set_device(self.rank)

            # Load datasets
            self.dataloading()

            # Create model, optimizer, and scheduler
            self.create_model()

            if self.cuda:
                torch.backends.cudnn.benchmark = True

            if self.world_size > 1:
                torch.autograd.set_detect_anomaly(True)

            self.train()

        except Exception as e:
            print(f"Error in process {rank}: {str(e)}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            if hasattr(self, 'cleanup'):
                self.cleanup()
            raise

    def setup_device(self):
        try:
            self.cuda = torch.cuda.is_available()
            if self.cuda:
                print(f"CUDA is available - Using GPU {self.rank}")
                self.device = torch.device(f'cuda:{self.rank}')
                
                # CUDA configuration for performance
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                if self.world_size > 1:
                    os.environ['MASTER_ADDR'] = 'localhost'
                    os.environ['MASTER_PORT'] = '12355'
                    dist.init_process_group(
                        "nccl", 
                        rank=self.rank, 
                        world_size=self.world_size,
                        timeout=datetime.timedelta(minutes=30)
                    )
                
                # Pre-allocate CUDA memory
                torch.cuda.empty_cache()
                total_memory = torch.cuda.get_device_properties(self.rank).total_memory
                reserved_memory = int(total_memory * 0.95)
                torch.cuda.set_per_process_memory_fraction(0.95, self.rank)
                
                print(f"GPU {self.rank}: Reserved {reserved_memory/1024**3:.1f}GB of VRAM")
            else:
                print("CUDA not available - Using CPU")
                self.device = torch.device("cpu")
        except Exception as e:
            print(f"Error in setup_device for process {self.rank}: {str(e)}")
            raise

    def cleanup(self):
        try:
            if self.cuda and self.world_size > 1:
                dist.barrier()
                dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        finally:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"Warning: Could not remove temporary directory: {e}")

    def dataloading(self):
        os.makedirs("images", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        self.train_dataset = Satellite2Map_Data(root=self.train_dir)
        self.val_dataset = Satellite2Map_Data(root=self.val_dir)

        if self.cuda and self.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
        else:
            train_sampler = None
            val_sampler = None

        self.train_dl = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        self.val_dl = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=True
        )

        print(f"Train Data Loaded - {len(self.train_dataset)} images")
        print(f"Validation Data Loaded - {len(self.val_dataset)} images")

    def create_model(self):
        # Initialize model
        self.model = ConditionalUNet(
            in_channels=3,
            out_channels=3,
            time_dim=256,
            context_dim=768
        ).to(self.device)

        # Initialize diffusion scheduler
        self.scheduler = DDPMScheduler(
            timesteps=1000,
            schedule="cosine",  # Using cosine schedule for better results
            device=self.device
        )

        # Calculate and display total parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel Statistics:")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Diffusion steps: {self.scheduler.timesteps}\n")

        # Setup distributed training if using CUDA
        if self.cuda and self.world_size > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.rank], output_device=self.rank)
            print(f"Model wrapped in DistributedDataParallel on GPU {self.rank}")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            epochs=self.n_epochs,
            steps_per_epoch=len(self.train_dl),
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000,
        )

        # Load model if requested
        self.starting_epoch = 0
        if self.load_model:
            try:
                model_and_optimizer, epoch = load_model()
                self.model.load_state_dict(model_and_optimizer['model_state_dict'])
                self.optimizer.load_state_dict(model_and_optimizer['optimizer_state_dict'])
                self.starting_epoch = epoch
                print(f"Successfully loaded model from epoch {epoch}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting from scratch")

        # Initialize loss history
        self.train_losses = []
        self.val_losses = []

        # Early stopping parameters
        self.best_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):
        try:
            if self.rank == 0:
                os.makedirs("save", exist_ok=True)
                os.makedirs("save/loss", exist_ok=True)
                os.makedirs("save/checkpoints", exist_ok=True)
                os.makedirs("images", exist_ok=True)

            # For throughput calculation
            import time
            batch_times = []

            for epoch in range(self.starting_epoch, self.n_epochs):
                if self.world_size > 1:
                    self.train_dl.sampler.set_epoch(epoch)

                epoch_start_time = time.time()
                total_images = 0
                epoch_loss = 0
                num_batches = 0

                # Training phase
                self.model.train()
                
                for idx, (context, target, _) in enumerate(self.train_dl):
                    batch_start_time = time.time()
                    num_batches += 1
                    current_batch_size = context.size(0)
                    
                    context = context.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    total_images += current_batch_size

                    # Sample random timesteps
                    t = torch.randint(0, self.scheduler.timesteps, (current_batch_size,),
                                    device=self.device).long()

                    self.optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast():
                        loss = self.scheduler.p_losses(self.model, target, t, context)

                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.lr_scheduler.step()

                    epoch_loss += loss.item()

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
                            f"[Epoch {epoch+1}/{self.n_epochs}] "
                            f"[Batch {idx+1}/{len(self.train_dl)}] "
                            f"[Loss: {loss.item():.4f}] "
                            f"[{images_per_sec:.2f} img/s] "
                            f"[lr: {self.optimizer.param_groups[0]['lr']:.2e}]"
                        )

                        if idx % 100 == 0:
                            # Generate samples
                            self.model.eval()
                            with torch.no_grad():
                                samples = self.scheduler.ddim_sample(
                                    self.model,
                                    context[:4],
                                    (4, 3, self.image_size, self.image_size),
                                    self.device
                                )
                                img_grid = torch.cat((context[:4], samples, target[:4]), dim=2)
                                save_image(img_grid, f"images/{epoch}-{idx}.png", nrow=1, normalize=True)
                            self.model.train()

                # Synchronize losses across GPUs
                if self.world_size > 1:
                    dist.all_reduce(torch.tensor([epoch_loss], device=self.device))
                    epoch_loss /= self.world_size

                if self.rank == 0:
                    avg_epoch_loss = epoch_loss / num_batches
                    self.train_losses.append(avg_epoch_loss)

                    # Calculate epoch throughput
                    epoch_time = time.time() - epoch_start_time
                    epoch_throughput = total_images / epoch_time
                    print(f"Epoch {epoch+1} completed. Average throughput: {epoch_throughput:.2f} images/s")

                    # Validation
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)
                    print(f"Validation Loss: {val_loss:.4f}")

                    # Save model if it's the best so far
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.patience_counter = 0
                        if self.save_model_bool:
                            save_model(
                                models={'model': self.model},
                                optimizers={'optimizer': self.optimizer},
                                suffix=f"-best-{epoch}"
                            )
                    else:
                        self.patience_counter += 1

                    # Regular checkpointing
                    if epoch % 10 == 0 and self.save_model_bool:
                        save_model(
                            models={'model': self.model},
                            optimizers={'optimizer': self.optimizer},
                            suffix=f"-{epoch}"
                        )
                        save_results(
                            params=self.M_CFG,
                            metrics={'train_loss': avg_epoch_loss, 'val_loss': val_loss}
                        )

                    # Early stopping check
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            if self.rank == 0:
                # Final save
                save_model(
                    models={'model': self.model},
                    optimizers={'optimizer': self.optimizer},
                    suffix="-final"
                )

        except Exception as e:
            print(f"Error during training: {e}")
            raise
        finally:
            self.cleanup()

    def validate(self):
        self.model.eval()
        val_loss = 0
        num_batches = 0

        with torch.no_grad():
            for context, target, _ in self.val_dl:
                context = context.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # Sample random timesteps
                t = torch.randint(0, self.scheduler.timesteps, (context.size(0),),
                                device=self.device).long()

                with torch.cuda.amp.autocast():
                    loss = self.scheduler.p_losses(self.model, target, t, context)

                val_loss += loss.item()
                num_batches += 1

                # Clean up GPU memory
                if num_batches % 2 == 0:
                    torch.cuda.empty_cache()

        self.model.train()
        return val_loss / num_batches 