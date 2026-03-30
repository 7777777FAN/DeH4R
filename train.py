
from dataset import DeH4R_Dataset, DeH4R_DataModule
from model import DeH4R
from utils import load_config, collate_fn

from argparse import ArgumentParser
from torch.utils.data import DataLoader

import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

import os 


parser = ArgumentParser()
parser.add_argument('--config', default='./config/DeH4R.yml')
parser.add_argument('--dev_run', default=False, action='store_true')
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--ckpt', default=None)
parser.add_argument('--run_id', default=None)
parser.add_argument('--n_gpus', type=int, default=2)



if '__main__' == __name__:
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.dev_run = args.dev_run
    
    # Initialize only on rank 0
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        wandb.init(
            project="DeH4R_spacenet" if cfg.DATASET=='spacenet' else "DeH4R",
            config=cfg,
            mode='disabled' if args.dev_run else 'online',
            # mode='disabled' if args.dev_run else 'offline',
            resume='must' if args.resume else None,
            id=args.run_id if args.resume else None,
        )
        
    wandb_logger = WandbLogger(project="DeH4R", config=cfg)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True 
    
    torch.set_float32_matmul_precision('medium') 
    
    
    model = DeH4R(cfg=cfg)
    # print(model)
    # assert False
    
    train_ds = DeH4R_Dataset(cfg=cfg, is_train=True, dev_run=args.dev_run)
    val_ds   = DeH4R_Dataset(cfg=cfg, is_train=False, dev_run=args.dev_run)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        # prefetch_factor=8,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    checkpoint_callback = ModelCheckpoint(every_n_epochs=1, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    find_unused_parameters = True if (not cfg.MASK_DECODER.USE_HIGH_RES_FEAT) and (not cfg.MASK_DECODER.USE_HIGH_RES_FEAT) else False
    trainer = pl.Trainer(
        max_epochs=cfg.EPOCH,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger,
        fast_dev_run=len(train_ds) if args.dev_run else False,
        precision=32,
        devices=list(range(args.n_gpus)),
        strategy=DDPStrategy()
        # strategy=DDPStrategy(find_unused_parameters=find_unused_parameters)
    )
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader, 
        # datamodule=datamodule,
        ckpt_path=args.ckpt if args.resume else None
    )
    
    train_ds.close()
    val_ds.close()
