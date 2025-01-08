import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from sat2plan.logic.configuration.config import Model_Configuration, Global_Configuration
from torch.autograd import Variable
from torch import autograd
import pandas as pd
from sat2plan.logic.models.ucvgan.model_building import Generator, Discriminator
from sat2plan.logic.loss.loss import GradientPenalty
from sat2plan.scripts.flow import save_results, save_model, load_model
from sat2plan.logic.preproc.dataset import Satellite2Map_Data

class UCVGan():
    def __init__(self, rank, world_size):
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

        # Setup device
        self.setup_device()

        # Loading Data
        self.dataloading()

        # Création des models, optimizers, losses
        self.create_models()

        # If True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
        if self.cuda:
            torch.backends.cudnn.benchmark = True

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
            
            # Pré-allocation de la mémoire CUDA
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(self.rank).total_memory
            reserved_memory = int(total_memory * 0.95)  # Réserve 95% de la mémoire disponible
            torch.cuda.set_per_process_memory_fraction(0.95, self.rank)
            
            # Création d'un cache de tenseurs pour réutilisation
            self.tensor_cache = {}
            
            # Configuration du processus distribué
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(
                "nccl", rank=self.rank, world_size=self.world_size)
            
            print(f"GPU {self.rank}: Reserved {reserved_memory/1024**3:.1f}GB of VRAM")
        else:
            print("CUDA not available - Using CPU")
            self.device = torch.device("cpu")

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
            self.netG = nn.parallel.DistributedDataParallel(
                self.netG, device_ids=[self.rank], output_device=self.rank)
            self.netD = nn.parallel.DistributedDataParallel(
                self.netD, device_ids=[self.rank], output_device=self.rank)
            print(f"Models wrapped in DistributedDataParallel on GPU {self.rank}")

        # Initialize optimizers
        self.OptimizerD = torch.optim.Adam(
            self.netD.parameters(), lr=self.learning_rate_D, betas=(self.beta1, self.beta2))
        self.OptimizerG = torch.optim.Adam(
            self.netG.parameters(), lr=self.learning_rate_G, betas=(self.beta1, self.beta2))

        # Initialize learning rate schedulers
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.OptimizerD, mode='min', factor=0.5, patience=5, verbose=True)
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.OptimizerG, mode='min', factor=0.5, patience=5, verbose=True)

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

        return

    # Train & save models
    def train(self):
        # Setup directories and logging
        os.makedirs("save", exist_ok=True)
        os.makedirs("save/loss", exist_ok=True)
        os.makedirs("save/checkpoints", exist_ok=True)
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
                current_batch_size = self.batch_size
                
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
                    
                    # Créer les labels avec la bonne dimension
                    real_label = torch.ones_like(D_real, device=self.device)
                    fake_label = torch.zeros_like(D_fake, device=self.device)
                    
                    D_real_loss = self.BCE_Loss(D_real, real_label)
                    D_fake_loss = self.BCE_Loss(D_fake, fake_label)
                    D_loss = (D_fake_loss + D_real_loss) / 2
                    gp = gradient_penalty(self.netD, y.detach(), y_fake.detach(), x)
                    D_loss_W = D_loss + gp
 
                self.scaler.scale(D_loss_W).backward()
                self.scaler.step(self.OptimizerD)

                ############## Train Generator ##############
                self.OptimizerG.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type='cuda' if self.cuda else 'cpu'):
                    D_fake = self.netD(x, y_fake)
                    G_fake_loss = self.BCE_Loss(D_fake, torch.ones_like(D_fake))
                    L1 = self.L1_Loss(y_fake, y) * self.l1_lambda
                    G_loss = G_fake_loss + L1

                self.scaler.scale(G_loss).backward()
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

                if idx % 10 == 0:  # Réduit la fréquence des logs
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [%.2f img/s]"
                        % (epoch+1, self.n_epochs, idx+1, len(self.train_dl), 
                           D_loss_W.item(), G_loss.item(), images_per_sec))
                    # Save images moins fréquemment
                    if idx % 100 == 0:
                        with torch.no_grad():
                            with torch.amp.autocast(device_type='cuda' if self.cuda else 'cpu'):
                                concatenated_images = torch.cat((x[:4], y_fake[:4], y[:4]), dim=2)  # Réduit le nombre d'images
                            save_image(concatenated_images, f"images/{str(epoch) + '-' + str(idx)}.png", nrow=3, normalize=True)

            # Calculate average epoch losses
            avg_epoch_d_loss = epoch_d_loss / num_batches
            avg_epoch_g_loss = epoch_g_loss / num_batches

            # Calculate epoch throughput
            epoch_time = time.time() - epoch_start_time
            epoch_throughput = total_images / epoch_time
            print(f"Epoch {epoch+1} completed. Average throughput: {epoch_throughput:.2f} images/s")

            # Save loss history moins fréquemment
            if epoch % 5 == 0:
                loss_df = pd.DataFrame(loss, columns=["epoch", "batch", "loss_g", "loss_d"])
                loss_df.to_csv("save/loss/loss.csv", mode="a", header=(epoch == 0))

            # Validation and model saving
            print("-- Validation Test --")
            self.validation()
            val_loss = self.val_Gen_loss[-1] + self.val_Dis_loss[-1]
            print(f"Epoch : {epoch+1}/{self.n_epochs} :")
            print(f"Validation Discriminator Loss : {self.val_Dis_loss[-1]}")
            print(f"Validation Generator Loss : {self.val_Gen_loss[-1]} : {self.val_Gen_fake_loss[-1]} + {self.val_Gen_L1_loss[-1]}")
            print("------------------------")

            # Update learning rate schedulers
            self.schedulerD.step(self.val_Dis_loss[-1])
            self.schedulerG.step(self.val_Gen_loss[-1])

            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                if self.save_model_bool:
                    save_model(
                        {"gen": self.netG, "disc": self.netD},
                        {"gen_opt": self.OptimizerG, "gen_disc": self.OptimizerD},
                        suffix=f"-best"
                    )
            else:
                self.patience_counter += 1

            # Regular checkpoint saving (moins fréquent)
            if epoch % 20 == 0 and self.save_model_bool:
                save_model(
                    {"gen": self.netG, "disc": self.netD},
                    {"gen_opt": self.OptimizerG, "gen_disc": self.OptimizerD},
                    suffix=f"-{epoch}"
                )
                save_results(params=self.M_CFG, metrics=dict(
                    Gen_loss=avg_epoch_g_loss,
                    Dis_loss=avg_epoch_d_loss,
                    Val_Gen_loss=self.val_Gen_loss[-1],
                    Val_Dis_loss=self.val_Dis_loss[-1]
                ))

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Final save
        save_model(
            {"gen": self.netG, "disc": self.netD},
            {"gen_opt": self.OptimizerG, "gen_disc": self.OptimizerD},
            suffix=f"-final"
        )
        save_results(params=self.M_CFG, metrics=dict(
            Gen_loss=avg_epoch_g_loss,
            Dis_loss=avg_epoch_d_loss,
            Val_Gen_loss=self.val_Gen_loss[-1],
            Val_Dis_loss=self.val_Dis_loss[-1]
        ))
        params_json.close()

        return

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
