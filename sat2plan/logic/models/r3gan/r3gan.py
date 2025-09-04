import os
import tempfile

# Create and set up temporary directory before any other imports
temp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(temp_dir, exist_ok=True)
os.environ['TMPDIR'] = temp_dir
tempfile.tempdir = temp_dir

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
from sat2plan.logic.models.r3gan.model_building import Generator, Discriminator
# Removed GradientPenalty import, R1/R2 will be calculated directly
from sat2plan.scripts.flow import save_results, save_model, load_model
from sat2plan.logic.preproc.dataset import Satellite2Map_Data
import shutil

class R3Gan():
    def __init__(self, rank, world_size):
        try:
            # Use the already created temporary directory
            self.temp_dir = temp_dir
            
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
            self.warmup_epochs = 5
            self.warmup_factor = 0.1

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
                
            # Pour le debug des opérations inplace
            if self.world_size > 1:
                torch.autograd.set_detect_anomaly(True)

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
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            
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
            test_tensor = torch.cuda.FloatTensor(2, 2).fill_(1.0)
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
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"Warning: Could not remove temporary directory: {e}")

    def __del__(self):
        self.cleanup()

    def get_cached_tensor(self, shape, dtype=torch.float32):
        """Récupère un tenseur du cache ou en crée un nouveau"""
        key = (shape, dtype)
        if key not in self.tensor_cache:
            self.tensor_cache[key] = torch.empty(shape, dtype=dtype, device=self.device)
        return self.tensor_cache[key]

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
        self.netD = Discriminator(in_channels=3, base_features=64, num_blocks_per_stage=2, groups=16, expansion_ratio=4)
        self.netG = Generator(in_channels=3, base_features=64, num_blocks_per_stage=2, groups=16, expansion_ratio=4)
        
        # Explicitly move models to GPU and verify
        self.netD = self.netD.to(self.device)
        self.netG = self.netG.to(self.device)
        print(f"Generator device: {next(self.netG.parameters()).device}")
        print(f"Discriminator device: {next(self.netD.parameters()).device}")
        
        self.starting_epoch = 0
        
        # Paramètres d'équilibrage
        # R3GAN parameters (n_critic, g_factor, max_grad_norm removed)
        self.l1_lambda = 10.0  # Keep L1 for image-to-image
        self.lambda_gp = 10.0 # Placeholder, will use separate R1/R2 gamma later
        self.r1_gamma = 10.0 # Default R1 gamma, can be tuned
        self.r2_gamma = 0.0 # Default R2 gamma (often R1 is sufficient, but R3GAN uses both)
        
        # Gradient smoothing
        self.beta_smoothing = 0.999
        self.generator_ema = Generator(in_channels=3, base_features=64, num_blocks_per_stage=2, groups=16, expansion_ratio=4).to(self.device)
        self.generator_ema.load_state_dict(self.netG.state_dict())
        for param in self.generator_ema.parameters():
            param.requires_grad = False

        # Setup distributed training if using CUDA
        if self.cuda and self.world_size > 1:
            # Convert BatchNorm to SyncBatchNorm before DDP
            self.netG = nn.SyncBatchNorm.convert_sync_batchnorm(self.netG)
            self.netD = nn.SyncBatchNorm.convert_sync_batchnorm(self.netD)
            
            # Wrap models in DistributedDataParallel with specific H100 settings
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

        # Initialize optimizers with full learning rate
        self.OptimizerD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=self.learning_rate_D, # Use full LR initially, scheduler handles decay
            betas=(0.0, 0.99), # R3GAN uses beta1=0
            fused=True
        )
        self.OptimizerG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=self.learning_rate_G, # Use full LR initially, scheduler handles decay
            betas=(0.0, 0.99), # R3GAN uses beta1=0
            fused=True
        )

        # Custom learning rate scheduler avec un minimum pour éviter les instabilités
        total_epochs = self.n_epochs
        constant_epochs = total_epochs // 2
        min_lr = 1e-6
        
        def lr_lambda(epoch):
            if epoch < constant_epochs:
                return 1.0
            else:
                decay = 1.0 - (epoch - constant_epochs) / (total_epochs - constant_epochs)
                return max(decay, min_lr)
        
        self.schedulerD = torch.optim.lr_scheduler.LambdaLR(self.OptimizerD, lr_lambda)
        self.schedulerG = torch.optim.lr_scheduler.LambdaLR(self.OptimizerG, lr_lambda)

        # Compile models if using PyTorch 2.0+
        if hasattr(torch, 'compile'):
            try:
                print("Compiling models with torch.compile()...")
                # Use inductor backend for H100
                compile_config = {
                    "mode": "default",
                    # "backend": "inductor",
                    # "fullgraph": False,  # Changed to False to avoid CUDA graph issues
                    # "dynamic": True,  # Changed to True for more flexible execution
                    # "options": {
                    #     "max_autotune": True,
                    #     "epilogue_fusion": True,
                    #     "max_fusion_size": 4,
                    # }
                }
                self.netG = torch.compile(self.netG, **compile_config)
                # self.netD = torch.compile(self.netD, **compile_config) # Disable compilation for Discriminator due to channel error
                print("Generator compiled, Discriminator remains uncompiled.")
                print("Models successfully compiled")
            except Exception as e:
                print(f"Warning: Model compilation failed: {e}")
                print("Continuing without compilation")

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

        # Initialize losses and metrics tracking
        self.scaler = torch.amp.GradScaler(enabled=True)  # Enable mixed precision
        # self.BCE_Loss = nn.BCEWithLogitsLoss().to(self.device) # Removed, RpGAN implemented in train loop
        self.L1_Loss = nn.L1Loss().to(self.device)
        
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
        # self.eps = 1e-8 # Removed, not needed for RpGAN

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

            # gradient_penalty = GradientPenalty(self.batch_size, self.lambda_gp, device=self.device) # Removed for R1/R2
            loss = []

            # Pour le calcul du throughput
            import time
            batch_times = []
            
            for epoch in range(self.starting_epoch, self.n_epochs):
                # Update learning rates using schedulers
                self.schedulerD.step()
                self.schedulerG.step()

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

                    # Pre-compute fake images for both D and G updates
                    with torch.amp.autocast(device_type='cuda'):
                        torch.compiler.cudagraph_mark_step_begin()  # Mark CUDA graph step
                        y_fake = self.netG(x)
                        y_fake = y_fake.clone()  # Clone to prevent overwriting
                        # EMA generator removed for R3GAN baseline simplicity

                    ############## Train Discriminator ##############
                    ############## Train Discriminator ##############
                    # R3GAN trains D every step
                    for p in self.netD.parameters():
                        p.requires_grad = True
                    self.OptimizerD.zero_grad(set_to_none=True)

                    # Enable gradients for real data for R1 penalty
                    y.requires_grad = True

                    with torch.amp.autocast(device_type='cuda'):
                        torch.compiler.cudagraph_mark_step_begin()  # Mark CUDA graph step

                        # Real forward pass
                        D_real_logits = self.netD(x, y)

                        # Fake forward pass (detach y_fake to avoid G gradients)
                        # print(f"Shape of x before D_fake call: {x.shape}") # DEBUG removed
                        # print(f"Shape of y_fake.detach() before D_fake call: {y_fake.detach().shape}") # DEBUG removed
                        D_fake_logits = self.netD(x, y_fake.detach())

                        # RpGAN Loss for Discriminator (minimize f(D_real - D_fake))
                        # f(t) = -log(1 + exp(-t)) = softplus(-t)
                        # Calculated inside autocast
                        D_loss_rpgan = torch.nn.functional.softplus(D_fake_logits - D_real_logits).mean()

                        # R1 Gradient Penalty will be calculated outside autocast

                        # Partial Discriminator Loss (inside autocast - only RpGAN loss now)
                        D_loss_partial = D_loss_rpgan
# Removed redundant D_loss_partial assignment

                    # --- Exit Autocast Block ---

                    # --- Calculate Penalties Outside Autocast ---

                    # R1 Gradient Penalty (on real data)
                    grad_penalty_r1 = torch.tensor(0.0, device=self.device)
                    if self.r1_gamma > 0:
                        # y requires grad was set before autocast block
                        # D_real_logits was calculated inside autocast, need to use it
                        # Ensure y is float32 if D expects that
                        try:
                            grad_real = autograd.grad(
                                outputs=D_real_logits.sum(), # Use logits from autocast, but calculate grad outside
                                inputs=y,                    # y requires grad
                                create_graph=True,           # Still need graph for D backward
                                allow_unused=False
                            )[0]
                            # Calculate penalty in full precision
                            grad_penalty_r1 = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                            grad_penalty_r1 = (self.r1_gamma / 2) * grad_penalty_r1
                        except RuntimeError as e:
                            if "element 0 of tensors does not require grad" in str(e) or "One of the differentiated Tensors" in str(e):
                                print("WARNING: R1 Penalty calculation failed with 'does not require grad'. Setting R1 penalty to 0 for this step.")
                                print(f"y requires_grad: {y.requires_grad}")
                                grad_penalty_r1 = torch.tensor(0.0, device=self.device)
                            elif "expected scalar type float" in str(e):
                                print("WARNING: R1 Penalty calculation failed due to dtype mismatch. Setting R1 penalty to 0 for this step.")
                                grad_penalty_r1 = torch.tensor(0.0, device=self.device)
                            else:
                                print(f"WARNING: R1 Penalty calculation failed with unexpected error: {e}. Setting R1 penalty to 0.")
                                grad_penalty_r1 = torch.tensor(0.0, device=self.device)


                    # R2 Gradient Penalty (on fake data)
                    grad_penalty_r2 = torch.tensor(0.0, device=self.device) # Initialize on correct device
                    if self.r2_gamma > 0:
                        # We need y_fake from the autocast block (already exists)
                        y_fake_for_r2 = y_fake.detach().clone() # Detach and clone outside autocast
                        y_fake_for_r2.requires_grad = True

                        # Run Discriminator forward pass outside autocast for R2
                        # Ensure inputs (x, y_fake_for_r2) are float32 if D expects that
                        # x is already on device, y_fake_for_r2 is created on device
                        D_fake_logits_for_r2 = self.netD(x, y_fake_for_r2) # Run in full precision

                        # Calculate gradient without autocast and without scaler
                        # Use try-except to catch the specific error if it persists
                        try:
                            grad_fake = autograd.grad(
                                outputs=D_fake_logits_for_r2.sum(), # No scaler here
                                inputs=y_fake_for_r2,
                                create_graph=True, # Still need graph for D backward pass
                                allow_unused=False # Ensure input is used
                            )[0]
                            grad_penalty_r2 = (grad_fake.view(grad_fake.size(0), -1).norm(2, dim=1) ** 2).mean()
                            grad_penalty_r2 = (self.r2_gamma / 2) * grad_penalty_r2
                        except RuntimeError as e:
                            if "One of the differentiated Tensors does not require grad" in str(e):
                                print("WARNING: R2 Penalty calculation failed again with 'does not require grad'. Setting R2 penalty to 0 for this step.")
                                print(f"y_fake_for_r2 requires_grad: {y_fake_for_r2.requires_grad}")
                                # Potentially inspect D_fake_logits_for_r2.grad_fn here if needed
                                grad_penalty_r2 = torch.tensor(0.0, device=self.device)
                            else:
                                raise e # Re-raise other runtime errors

                    # Final Discriminator Loss (Combine partial loss from autocast with R1 and R2 penalties)
                    # Ensure all components are float32 before adding
                    D_loss = D_loss_partial.float() + grad_penalty_r1.float() + grad_penalty_r2.float()

                    # Backward and Optimize (outside autocast, using the final D_loss)
                    self.scaler.scale(D_loss).backward() # Scale the combined loss
                    # self.scaler.unscale_(self.OptimizerD) # Unscale before clipping if needed
                    # torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=self.max_grad_norm) # Clipping removed
                    self.scaler.step(self.OptimizerD)
                    # Scaler update is done after G step

                    # Disable discriminator gradients for generator update
                    for p in self.netD.parameters():
                        p.requires_grad = False
                    # y.requires_grad = False # Already done after R1 calc

                    ############## Train Generator ##############
                    self.OptimizerG.zero_grad(set_to_none=True)

                    with torch.amp.autocast(device_type='cuda'):
                        torch.compiler.cudagraph_mark_step_begin()  # Mark CUDA graph step

                        # Need D outputs for G loss calculation
                        # Re-calculate D outputs (gradients are disabled for D)
                        D_real_logits_g = self.netD(x, y) # No grad needed on y here
                        D_fake_logits_g = self.netD(x, y_fake) # y_fake requires grad from G

                        # RpGAN Loss for Generator (minimize f(D_real - D_fake))
                        # f(t) = -log(1 + exp(-t)) = softplus(-t)
                        # Minimize softplus(D_fake - D_real)
                        G_loss_rpgan = torch.nn.functional.softplus(D_fake_logits_g - D_real_logits_g).mean()

                        # L1 Loss (reconstruction)
                        L1 = self.L1_Loss(y_fake, y) * self.l1_lambda

                        # Total Generator Loss
                        G_loss = G_loss_rpgan + L1

                        # Adaptive scaling removed
                        # Gradient clipping removed

                    # Backward and Optimize
                    self.scaler.scale(G_loss).backward()
                    # self.scaler.unscale_(self.OptimizerG) # Unscale before clipping if needed
                    # torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.max_grad_norm) # Clipping removed
                    self.scaler.step(self.OptimizerG)
                    self.scaler.update() # Update scaler once after both D and G steps

                    # Accumulate losses
                    epoch_d_loss += D_loss.item() # Use the calculated D_loss
                    epoch_g_loss += G_loss.item()

                    # Calculate and log throughput
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)
                    if len(batch_times) > 50:
                        batch_times.pop(0)
                    avg_time = sum(batch_times) / len(batch_times)
                    images_per_sec = current_batch_size / avg_time

                    if self.rank == 0 and idx % 1 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %.3f (RpGAN %.3f + R1 %.3f + R2 %.3f)] [G loss: %.3f (RpGAN %.3f + L1 %.3f)] [%.2f img/s] [lr D: %e] [lr G: %e]"
                            % (epoch+1, self.n_epochs, idx+1, len(self.train_dl),
                               D_loss.item(), D_loss_rpgan.item(), grad_penalty_r1.item(), grad_penalty_r2.item(),
                               G_loss.item(), G_loss_rpgan.item(), L1.item(), images_per_sec,
                               self.OptimizerD.param_groups[0]['lr'], self.OptimizerG.param_groups[0]['lr']))
                        
                        if idx % 50 == 0:  # Changé de 10 à 100 pour réduire le nombre d'images sauvegardées
                            with torch.no_grad():
                                with torch.amp.autocast('cuda' if self.cuda else 'cpu'):
                                    concatenated_images = torch.cat((x[:4], y_fake[:4], y[:4]), dim=2)
                                save_image(concatenated_images, f"images/{str(epoch) + '-' + str(idx)}.png", nrow=3, normalize=True)

                # Synchronize losses across GPUs
                if self.world_size > 1:
                    dist.all_reduce(torch.tensor([epoch_d_loss, epoch_g_loss], device=self.device))
                    epoch_d_loss /= self.world_size
                    epoch_g_loss /= self.world_size

                if self.rank == 0:
                    # Calculate average epoch losses
                    avg_epoch_d_loss = epoch_d_loss / num_batches
                    avg_epoch_g_loss = epoch_g_loss / num_batches

                    # Calculate epoch throughput
                    epoch_time = time.time() - epoch_start_time
                    epoch_throughput = total_images / epoch_time
                    print(f"Epoch {epoch+1} completed. Average throughput: {epoch_throughput:.2f} images/s")

                    if epoch % 5 == 0:
                        loss_df = pd.DataFrame(loss, columns=["epoch", "batch", "loss_g", "loss_d"])
                        loss_df.to_csv("save/loss/loss.csv", mode="a", header=(epoch == 0))

                    # Validation and model saving
                    print("-- Validation Test --")
                    self.validation()
                    val_loss = self.val_Gen_loss[-1] + self.val_Dis_loss[-1]
                    print(f"Epoch : {epoch+1}/{self.n_epochs}")
                    print(f"Validation Discriminator Loss : {self.val_Dis_loss[-1]:.3f} = Real: {self.val_D_real_loss[-1]:.3f} + Fake: {self.val_D_fake_loss[-1]:.3f}")
                    print(f"Validation Generator Loss : {self.val_Gen_loss[-1]:.3f} = Adv: {self.val_Gen_fake_loss[-1]:.3f} + L1: {self.val_Gen_L1_loss[-1]:.3f}")
                    print("------------------------")

                    # Early stopping check
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
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
                            Val_Dis_loss=self.val_Dis_loss[-1]
                        ))

                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

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
                    Val_Dis_loss=self.val_Dis_loss[-1]
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

        sum_D_loss = 0
        sum_G_loss = 0
        sum_G_rpgan_loss = 0 # Renamed from G_fake_loss
        sum_G_L1_loss = 0
        sum_D_rpgan_loss = 0 # Renamed from D_real/fake_loss
        # sum_D_real_loss = 0 # Removed
        # sum_D_fake_loss = 0 # Removed
        num_batches = 0

        with torch.no_grad(), torch.amp.autocast('cuda' if self.cuda else 'cpu'):
            for idx, (x, y, _) in enumerate(self.val_dl):
                # Move to device and free memory from previous batch
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                # Generator forward pass
                y_fake = self.netG(x)
                
                # Discriminator RpGAN loss
                D_real_logits = self.netD(x, y)
                D_fake_logits = self.netD(x, y_fake) # No detach needed in eval mode
                D_loss_rpgan = torch.nn.functional.softplus(D_fake_logits - D_real_logits).mean()
                # R1/R2 penalties are not typically calculated during validation
                D_loss = D_loss_rpgan

                # Generator RpGAN loss
                # Use the same logits calculated for D loss
                G_loss_rpgan = torch.nn.functional.softplus(D_fake_logits - D_real_logits).mean() # G minimizes softplus(D_fake - D_real)
                G_L1 = self.L1_Loss(y_fake, y) * self.l1_lambda
                G_loss = G_loss_rpgan + G_L1
                
                # Accumulate batch losses
                sum_D_loss += D_loss.item()
                sum_G_loss += G_loss.item()
                sum_G_rpgan_loss += G_loss_rpgan.item()
                sum_G_L1_loss += G_L1.item()
                sum_D_rpgan_loss += D_loss_rpgan.item()
                # sum_D_real_loss += D_real_loss.item() # Removed
                # sum_D_fake_loss += D_fake_loss.item() # Removed
                num_batches += 1
                
                # Explicitement libérer la mémoire
                del y_fake, D_real_logits, D_fake_logits, D_loss_rpgan, G_loss_rpgan, G_L1, D_loss, G_loss
                if idx % 2 == 0:  # Périodiquement forcer le garbage collector
                    torch.cuda.empty_cache()

        # Calculer les moyennes
        avg_D_loss = sum_D_loss / num_batches
        avg_G_loss = sum_G_loss / num_batches
        avg_G_rpgan_loss = sum_G_rpgan_loss / num_batches
        avg_G_L1_loss = sum_G_L1_loss / num_batches
        avg_D_rpgan_loss = sum_D_rpgan_loss / num_batches
        # avg_D_real_loss = sum_D_real_loss / num_batches # Removed
        # avg_D_fake_loss = sum_D_fake_loss / num_batches # Removed
        
        # Stocker les résultats
        self.val_Dis_loss.append(avg_D_loss)
        self.val_Gen_loss.append(avg_G_loss)
        self.val_Gen_fake_loss.append(avg_G_rpgan_loss) # Storing RpGAN loss here
        self.val_Gen_L1_loss.append(avg_G_L1_loss)
        # self.val_D_real_loss.append(avg_D_real_loss) # Removed
        # self.val_D_fake_loss.append(avg_D_fake_loss) # Removed
        # Store total D loss instead of real/fake breakdown
        self.val_D_real_loss.append(avg_D_rpgan_loss) # Reusing this list to store D RpGAN loss
        self.val_D_fake_loss.append(0.0) # Placeholder for fake loss list
        
        # Retour en mode train
        self.netG.train()
        self.netD.train()
        
        # Forcer le nettoyage de la mémoire à la fin
        torch.cuda.empty_cache()
