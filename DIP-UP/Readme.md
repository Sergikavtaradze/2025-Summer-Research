# DIP-UP: Deep Image Prior for Unwrapping Phase

- This study provides an enhancement mechanism for pre-trained 3D CNN models in MRI phase unwrapping.
- The enhancement is based on Deep Image Prior (DIP), which is established by two restrictions (Laplacian loss and TV loss)
- The pipeline combines two steps: 1) model training and 2) DIP-modification.

## <span id="head1">Checkpoints and Test Data </span>
- Checkpoints, including the pre-trained models PHU-NET3D and PhaseNet3D, are available at:
- https://www.dropbox.com/scl/fo/r4qv54fsdznxxmqgdcwn0/AMCMT9M-hvgmZ276-hcgaV8?rlkey=di0g1drz4whpz308rn84cxnx9&st=yw6d5qfz&dl=0
- Testing labels including one simulation (10ms TE, with noisy, sigma = 0.1) and one InVivo (5.8ma TE), available at:
- https://www.dropbox.com/scl/fo/iz98whja62v5ih6idg60g/AEyEcemOzOd6yY1A0n2sldM?rlkey=zol5wkvxn4onu6xi5mf55w3en&st=54svy08l&dl=0

## <span id="head3">Usage </span>

- Train: Train_PhaseNet3D.py 

    - root: The root directory of trainig dataset.
    - file_path: The sepcific path of each trainig data. 

    - depth: depth of Unet for QSM reconstruction.
    - recon_base:  base channel number of Unet for QSM reconstruction.
