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
from sat2plan.logic.models.ucvgan.model_building import Generator, Discriminator
from sat2plan.logic.loss.loss import GradientPenalty
from sat2plan.scripts.flow import save_results, save_model, load_model
from sat2plan.logic.preproc.dataset import Satellite2Map_Data
import shutil

class UCVGan():
    def __init__(self, rank, world_size):
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
        self.l1_lambda = self.G_CFG.l1_lambda
        self.lambda_gp = self.G_CFG.lambda_gp
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

    def setup_device(self):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            print(f"CUDA is available - Using GPU {self.rank}")
            self.device = torch.device(f'cuda:{self.rank}')
            
            # Configuration CUDA pour les performances
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Configuration du processus distribué
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(
                "nccl", 
                rank=self.rank, 
                world_size=self.world_size,
                timeout=datetime.timedelta(minutes=30)
            )
            
            # Pré-allocation de la mémoire CUDA
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(self.rank).total_memory
            reserved_memory = int(total_memory * 0.95)  # Réserve 95% de la mémoire disponible
            torch.cuda.set_per_process_memory_fraction(0.95, self.rank)
            
            print(f"GPU {self.rank}: Reserved {reserved_memory/1024**3:.1f}GB of VRAM")
        else:
            print("CUDA not available - Using CPU")
            self.device = torch.device("cpu")

    def cleanup(self):
        if self.cuda and self.world_size > 1:
            dist.destroy_process_group()
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
        self.train_dataset = Satellite2Map_Data(root=self.train_dir)
        self.val_dataset = Satellite2Map_Data(root=self.val_dir)

        # Create samplers for distributed training
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

        # Create dataloaders
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

        return

    # Create models, optimizers ans losses

    def create_models(self):
        # Initialize models
        self.netD = Discriminator(in_channels=3).to(self.device)
        self.netG = Generator(in_channels=3).to(self.device)
        self.starting_epoch = 0

        # Setup distributed training if using CUDA
        if self.cuda and self.world_size > 1:
            # Convert BatchNorm to SyncBatchNorm before DDP
            self.netG = nn.SyncBatchNorm.convert_sync_batchnorm(self.netG)
            self.netD = nn.SyncBatchNorm.convert_sync_batchnorm(self.netD)
            
            # Wrap models in DistributedDataParallel
            self.netG = nn.parallel.DistributedDataParallel(
                self.netG, device_ids=[self.rank], output_device=self.rank)
            self.netD = nn.parallel.DistributedDataParallel(
                self.netD, device_ids=[self.rank], output_device=self.rank)
            print(f"Models wrapped in DistributedDataParallel on GPU {self.rank}")

        # Initialize optimizers
        self.OptimizerD = torch.optim.Adam(
            self.netD.parameters(), lr=self.learning_rate_D , betas=(self.beta1, self.beta2))
        self.OptimizerG = torch.optim.Adam(
            self.netG.parameters(), lr=self.learning_rate_G , betas=(self.beta1, self.beta2))

        # Initialize learning rate schedulers with warm restarts
        self.schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.OptimizerD, T_0=10, T_mult=2, eta_min=1e-6
        )
        self.schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.OptimizerG, T_0=10, T_mult=2, eta_min=1e-6
        )

        # Warmup parameters
        self.warmup_epochs = 5
        self.warmup_factor = 0.1

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
        self.scaler = torch.amp.GradScaler()  # For mixed precision training
        self.BCE_Loss = nn.BCEWithLogitsLoss()
        self.L1_Loss = nn.L1Loss()
        
        # Initialize loss history lists
        self.Gen_loss = []
        self.Dis_loss = []
        self.val_Dis_loss = []
        self.val_Gen_loss = []
        self.val_Gen_fake_loss = []
        self.val_Gen_L1_loss = []

        # Early stopping parameters
        self.best_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0

        # Add small epsilon for numerical stability
        self.eps = 1e-8

        return

    # Train & save models
    def train(self):
        try:
            # Setup directories and logging
            if self.rank == 0:  # Seulement le processus principal crée les dossiers
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

            gradient_penalty = GradientPenalty(self.batch_size, self.lambda_gp, device=self.device)
            loss = []

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

                    ############## Train Discriminator ##############
                    self.OptimizerD.zero_grad(set_to_none=True)
                    
                    with torch.amp.autocast(device_type='cuda' if self.cuda else 'cpu'):
                        y_fake = self.netG(x)
                        D_real = self.netD(x, y)
                        D_fake = self.netD(x, y_fake.detach())
                        
                        real_label = torch.ones_like(D_real) * 0.9  # Label smoothing
                        fake_label = torch.zeros_like(D_fake) + 0.1  # Label smoothing
                        
                        D_real_loss = self.BCE_Loss(D_real + self.eps, real_label)
                        D_fake_loss = self.BCE_Loss(D_fake + self.eps, fake_label)
                        D_loss = (D_fake_loss + D_real_loss) / 2
                        gp = gradient_penalty(self.netD, y.detach(), y_fake.detach(), x)
                        D_loss_W = D_loss + gp
     
                    self.scaler.scale(D_loss_W).backward()
                    # Add gradient clipping
                    self.scaler.unscale_(self.OptimizerD)
                    torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
                    self.scaler.step(self.OptimizerD)

                    ############## Train Generator ##############
                    self.OptimizerG.zero_grad(set_to_none=True)

                    with torch.amp.autocast(device_type='cuda' if self.cuda else 'cpu'):
                        D_fake = self.netD(x, y_fake)
                        G_fake_loss = self.BCE_Loss(D_fake, torch.ones_like(D_fake))
                        L1 = self.L1_Loss(y_fake, y) * self.l1_lambda
                        G_loss = G_fake_loss + L1

                    self.scaler.scale(G_loss).backward()
                    # Add gradient clipping
                    self.scaler.unscale_(self.OptimizerG)
                    torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
                    self.scaler.step(self.OptimizerG)
                    self.scaler.update()

                    # Accumulate losses
                    epoch_d_loss += D_loss_W.item()
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
                            "[Epoch %d/%d] [Batch %d/%d] [D: disc %.3f | gp %.3f] [G: adv %.3f | L1 %.3f] [%.2f img/s] [lr D: %e] [lr G: %e]"
                            % (epoch+1, self.n_epochs, idx+1, len(self.train_dl), 
                               D_loss.item(), gp.item(), G_fake_loss.item(), L1.item(), images_per_sec,
                               self.OptimizerD.param_groups[0]['lr'], self.OptimizerG.param_groups[0]['lr']))
                        
                        if idx % 100 == 0:  # Changé de 10 à 100 pour réduire le nombre d'images sauvegardées
                            with torch.no_grad():
                                with torch.amp.autocast(device_type='cuda' if self.cuda else 'cpu'):
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

                    # Update learning rates with warmup
                    if epoch < self.warmup_epochs:
                        warmup_factor = self.warmup_factor + (1 - self.warmup_factor) * (epoch / self.warmup_epochs)
                        for param_group in self.OptimizerD.param_groups:
                            param_group['lr'] = self.learning_rate_D * warmup_factor * 0.1
                        for param_group in self.OptimizerG.param_groups:
                            param_group['lr'] = self.learning_rate_G * warmup_factor * 0.1
                    else:
                        self.schedulerD.step((epoch - self.warmup_epochs) + idx / len(self.train_dl))
                        self.schedulerG.step((epoch - self.warmup_epochs) + idx / len(self.train_dl))

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

        finally:
            self.cleanup()

    # Test du modèle sur le set de validation
    def validation(self):
        # Passage en mode eval
        self.netG.eval()
        self.netD.eval()

        sum_D_loss = 0
        sum_G_loss = 0
        sum_G_fake_loss = 0
        sum_G_L1_loss = 0
        num_batches = 0

        with torch.no_grad(), torch.amp.autocast(device_type='cuda' if self.cuda else 'cpu'):
            for idx, (x, y, _) in enumerate(self.val_dl):
                # Move to device and free memory from previous batch
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                # Generator forward pass
                y_fake = self.netG(x)
                
                # Discriminator losses
                D_real = self.netD(x, y)
                D_fake = self.netD(x, y_fake)
                
                D_real_loss = self.BCE_Loss(D_real, torch.ones_like(D_real))
                D_fake_loss = self.BCE_Loss(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
                
                # Generator losses
                G_fake_loss = self.BCE_Loss(D_fake, torch.ones_like(D_fake))
                G_L1 = self.L1_Loss(y_fake, y) * self.l1_lambda
                G_loss = G_fake_loss + G_L1
                
                # Accumulate batch losses
                sum_D_loss += D_loss.item()
                sum_G_loss += G_loss.item()
                sum_G_fake_loss += G_fake_loss.item()
                sum_G_L1_loss += G_L1.item()
                num_batches += 1
                
                # Explicitement libérer la mémoire
                del y_fake, D_real, D_fake, D_loss, G_loss, G_L1
                if idx % 2 == 0:  # Périodiquement forcer le garbage collector
                    torch.cuda.empty_cache()

        # Calculer les moyennes
        avg_D_loss = sum_D_loss / num_batches
        avg_G_loss = sum_G_loss / num_batches
        avg_G_fake_loss = sum_G_fake_loss / num_batches
        avg_G_L1_loss = sum_G_L1_loss / num_batches
        
        # Stocker les résultats
        self.val_Dis_loss.append(avg_D_loss)
        self.val_Gen_loss.append(avg_G_loss)
        self.val_Gen_fake_loss.append(avg_G_fake_loss)
        self.val_Gen_L1_loss.append(avg_G_L1_loss)
        
        # Retour en mode train
        self.netG.train()
        self.netD.train()
        
        # Forcer le nettoyage de la mémoire à la fin
        torch.cuda.empty_cache()
