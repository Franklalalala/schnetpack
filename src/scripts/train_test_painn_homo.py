import os
from schnetpack.datasets import thuEMol
import schnetpack as spk
import schnetpack.transform as trn

import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

##################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def file_2_npy(file_path: str):
    value_list = []
    with open(file_path, 'r') as f_r:
        for a_line in f_r.readlines():
            a_line.strip()
            value_list.append(float(a_line))
    return np.array(value_list)


def plot_rel(label: str, prediction: str, flag: str):
    label = file_2_npy(label)
    prediction = file_2_npy(prediction)
    mae_e = np.mean(np.abs(prediction - label)).round(3)

    fig = plt.figure()
    plt.plot(label, label, label="Reference Line")
    r, _ = stats.pearsonr(label, prediction)
    r = round(r, 3)
    sns.regplot(x=prediction, y=label, marker='+', label='Distribution Dots', truncate=False,
                line_kws={'label': 'Regression Line'})

    plt.title(f'{flag} Correlation factor: {r}\n'
              f'{flag} MAE error(eV): {mae_e}')
    plt.xlabel('Prediction (ev)')
    plt.ylabel('Label (ev)')
    plt.legend(loc='best')
    plt.savefig(f'{flag}_rel_e.png', dpi=400)


def rename_file(flag):
    os.rename(src='pred', dst=f'{flag}_pred')
    os.rename(src='target', dst=f'{flag}_target')


##################################################################################################


if __name__ == "__main__":
    ##################################################################################################
    workbase = r'./run'
    thuEMolData = thuEMol(
        datapath='/root/data/rdkit_7_27/train.db',
        # datapath='/data/test_schnetpack/data0722/train.db',
        batch_size=100,
        num_train=33540,
        num_val=3354,
        num_test=0,
        load_properties=[thuEMol.homo],
        transforms=[
            trn.ASENeighborList(cutoff=20.),
            trn.CastTo32()
        ],
        pin_memory=False,
        num_workers=2
    )
    thuEMolData.prepare_data()
    thuEMolData.setup()
    ##################################################################################################
    cutoff = 20
    n_atom_basis = 128

    pairwise_distance = spk.atomistic.PairwiseDistances()  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)

    schnet = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff)
    )
    pred_U0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key=thuEMol.homo)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_U0],
        postprocessors=[trn.CastTo64()]
    )

    output_U0 = spk.task.ModelOutput(
        name=thuEMol.homo,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            "MAE": torchmetrics.MeanAbsoluteError()
        }
    )

    task = spk.task.SinglePropertyClrTask(
        model=nnpot,
        outputs=[output_U0],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 5e-4},
        predict_property=thuEMol.homo,
        scheduler_args={
            'base_lr': 1e-5,
            'max_lr': 5e-4,
            'step_size_up': 10,
            'step_size_down': 40,
            'lr_mode': "exp_range",
        }
    )
    ##################################################################################################
    logger = pl_loggers.TensorBoardLogger(save_dir=workbase)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(workbase, "best_val_inf"),
            filename='{epoch:08d}-{val_loss:.6f}',
            save_top_k=3,
            save_last=True,
            monitor="val_loss",
        )
    ]
    ##################################################################################################
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator='gpu',
        devices=[0],
        logger=logger,
        default_root_dir=workbase,
        max_epochs=2000,  # for testing, we restrict the number of epochs
    )

    trainer.fit(task, datamodule=thuEMolData)

    #################################################################################################
    if os.path.exists(r'split.npz'):
        os.remove(r'split.npz')
        os.remove(r'splitting.lock')
    if os.path.exists(r'pred'):
        os.remove(r'pred')
        os.remove(r'target')
    ##################################################################################################
    ##################################################################################################
    db_path = r'/root/data/rdkit_7_27/ood_test.db'
    ##################################################################################################
    iid_test_Data = thuEMol(
        datapath=db_path,
        # datapath='/data/test_schnetpack/data0722/train.db',
        batch_size=100,
        num_train=1,
        num_val=1,
        num_test=4447,
        load_properties=[thuEMol.homo],
        transforms=[
            trn.ASENeighborList(cutoff=20.),
            trn.CastTo32()
        ],
        pin_memory=False,
        num_workers=2
    )
    iid_test_Data.prepare_data()
    iid_test_Data.setup()
    ##################################################################################################
    trainer.test(model=task, datamodule=iid_test_Data, ckpt_path='best')
    plot_rel(label='target', prediction='pred', flag='ood_be')
    rename_file(flag='ood_be')
    ##################################################################################################
    ##################################################################################################
    if os.path.exists(r'split.npz'):
        os.remove(r'split.npz')
        os.remove(r'splitting.lock')
    if os.path.exists(r'pred'):
        os.remove(r'pred')
        os.remove(r'target')
    ##################################################################################################
    ##################################################################################################
    db_path = r'/root/data/rdkit_7_27/iid_test.db'
    ##################################################################################################
    iid_test_Data = thuEMol(
        datapath=db_path,
        # datapath='/data/test_schnetpack/data0722/train.db',
        batch_size=100,
        num_train=1,
        num_val=1,
        num_test=4600,
        load_properties=[thuEMol.homo],
        transforms=[
            trn.ASENeighborList(cutoff=20.),
            trn.CastTo32()
        ],
        pin_memory=False,
        num_workers=2
    )
    iid_test_Data.prepare_data()
    iid_test_Data.setup()
    ##################################################################################################
    trainer.test(model=task, datamodule=iid_test_Data, ckpt_path='best')
    plot_rel(label='target', prediction='pred', flag='iid_be')
    rename_file(flag='iid_be')
