import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot

import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment, create_lr_scheduler
from tnp_crps.utils.lightning_utils import LitWrapper, LogPerformanceCallback
from tnp_crps.utils.np_functions import np_loss_fn, np_pred_fn

def get_project_root() -> str:
    return os.environ.get(
        "TNP_CRPS_ROOT",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    )


def main():
    experiment = initialize_experiment()
    
    # added for local checkpoints and naming 
    project_root = get_project_root()
    run_group = getattr(experiment.misc, "run_group", None)
    run_id = getattr(experiment.misc, "run_id", None) or os.environ.get("TNP_RUN_ID", "default-run")

    # W&B run/display name
    run_name = f"{experiment.misc.name}-{run_id}"

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    scheduler = create_lr_scheduler(optimiser, experiment, gen_train)
    epochs = experiment.params.epochs

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        batch_size=None,
        num_workers=experiment.misc.num_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_workers > 0 else False,
        pin_memory=getattr(experiment.misc, "pin_memory", True),
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_val_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        pin_memory=getattr(experiment.misc, "pin_memory", True),
    )

    def plot_fn(model, batches, name):
        plot(
            model=model,
            batches=batches,
            num_fig=min(5, len(batches)),
            name=name,
            pred_fn=np_pred_fn,
        )

    # new logic for resuming from local or W&B checkpoint
    if experiment.misc.resume_from_checkpoint is not None:
        resume_ref = experiment.misc.resume_from_checkpoint

        if os.path.exists(resume_ref):
            ckpt_file = resume_ref
        else:
            api = wandb.Api()
            artifact = api.artifact(resume_ref)
            artifact_dir = artifact.download()
            ckpt_file = os.path.join(artifact_dir, "model.ckpt")
    else:
        ckpt_file = None
        
    lit_model = LitWrapper(
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        loss_fn=np_loss_fn,
        pred_fn=np_pred_fn,
        plot_fn=plot_fn,
        plot_interval=experiment.misc.plot_interval,
        )

    checkpoint_parts = [
        project_root,
        "checkpoints",
    ]

    if run_group is not None:
        checkpoint_parts.append(str(run_group))

    checkpoint_parts.extend(
        [
            experiment.misc.project,
            experiment.misc.name,
            run_id,
        ]
    )

    checkpoint_dir = os.path.join(*checkpoint_parts)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # add local checkpointing independent of W&B + W&B
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:04d}",
        every_n_epochs=experiment.misc.checkpoint_interval,
        save_last=True,
        save_top_k=-1,
    )

    callbacks = [checkpoint_callback]

    if scheduler is not None and experiment.misc.logging:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))

    if experiment.misc.logging:
        logger = pl.loggers.WandbLogger(
            project=experiment.misc.project,
            entity=os.environ.get("WANDB_ENTITY", None),
            name=run_name,
            group=str(run_group) if run_group is not None else None,
            config=OmegaConf.to_container(experiment.config),
            log_model="all",
            save_dir=os.path.join(project_root, "logs"),
        )
        performance_callback = LogPerformanceCallback()
        callbacks.append(performance_callback)
    else:
        logger = False


    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=(
            experiment.misc.log_interval if not experiment.misc.logging else None
        ),
        devices="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=(experiment.misc.check_val_every_n_epoch),
        gradient_clip_val=experiment.misc.gradient_clip_val,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_file,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
