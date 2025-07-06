import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
import time
from torch.nn.functional import pad
import sys
import math
import nibabel as nib
import math

sys.path.append('/clusterdata/uqxzhu14/Script_QSMUnet/xQSM_Python/YangsCodeNew/Unet_Backup/')

from Unet_2Chan_9Class import *
from Train_PHUNET3D import *

if __name__ == '__main__':
    # with torch.no_grad():
    FileNo = 1  # file identifier
    # Set the loading path of pre-trained PHU-NET3D
    SourceDir = '/scratch/itee/xy_BFR/PhaseUnwrapping/recon/ChrisResults/'

    ReconType = 'PHU-NET3D/'  # Update
    # Set the storage path

    SaveDir_NIFTI = SourceDir + 'NewBrain/InVivo/recons/NIFTI/' + ReconType
    No_subs = 1  # total number of subjects to be reconstructed

    ModelFoder = SourceDir + 'SomethingGood/'
    ModelName = 'PHUNET'
    LoadModel = ModelFoder + ModelName + '.pth'

    Inferencetype = 'InVivo'
    # Inference type include 'Simulation' and 'InVivo'

    print('DIP type: MixLossDIP with Laplacian loss and TV-Masked Loss')
    print('InVivo, Echo1, variable learning rate from 10e-6 to 10% decay in 10 epoch, New Softmax * 10000')

    print('Network Loading')
    ## load pre-trained network
    PHUnet = Unet_2Chan_9Class(4)
    # 9 Classee for 10ms TE. Calculate the class before loading the pre-trained models
    PHUnet = nn.DataParallel(PHUnet)
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PHUnet.load_state_dict(torch.load(LoadModel, map_location=torch.device('cpu')))  # ZXY
    # PHUnet.load_state_dict(torch.load(LoadModel))  # ZXY
    PHUnet = PHUnet.to(device)  # Model load to CPU?
    PHUnet.eval()

    # Set the initial learning rate here. You can choose learning rate decay below in the looping area.
    optimizer1 = optim.RMSprop(PHUnet.parameters(), lr=1e-6)
    # optimizer1 = optim.RMSprop(PHUnet.parameters(), lr=1e-4)

    # T1 = time.clock()
    # Loop test. if there's only one image ready for test, then the No_subs = 1

    for index in range(FileNo, FileNo + No_subs):
        # Upload the raw phase image here, in .mat format
        input_img_files = SourceDir + 'NewBrain/InVivo/wph/' + ("wph%s_144.mat" % index)  # XYZ
        # Upload the Laplacian image here, in.mat format
        input_lap_files = SourceDir + 'NewBrain/InVivo/wph_lap/' + ('wph_lap%s_144.mat' % index)  # XYZ

        print('Loading Test Data No.%s' % index)
        matTest = scio.loadmat(input_img_files, verify_compressed_data_integrity=False)
        matLap = scio.loadmat(input_lap_files, verify_compressed_data_integrity=False)

        image = matTest['wph']  # XYZ
        lap = matLap['wph_lap']  # XYZ

        image = np.array(image)
        image = np.real(image)
        image = torch.from_numpy(image)
        image = image.float()

        img_lap = np.array(lap)
        img_lap = np.real(img_lap)
        img_lap = torch.from_numpy(img_lap)
        img_lap = img_lap.float()

        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        img_lap = torch.unsqueeze(img_lap, 0)
        img_lap = torch.unsqueeze(img_lap, 0)

        image = image.to(device)
        img_lap = img_lap.to(device)

        Features = torch.cat([image, img_lap], dim=1)  # ZX edit

        tissue_mask = torch.zeros_like(image)
        tissue_mask[image != 0] = 1

        shift_base = 5

        # for params in optimizer1.param_groups:
        #     params['lr'] *= 0.9

        # with torch.no_grad():
        time_start = time.time()
        for inner in range(2001):

            recons = PHUnet(Features)  # Check line No.22

            # print('Recon Initialized')

            # recons = recons.to('cpu')

            recons_softmax = torch.softmax(recons*10000, dim=1)
            Idx = torch.arange(0, 9, device=device)

            Idx2 = Idx[None, :, None, None, None]
            recon_distri = Idx2 * recons_softmax

            recon_count = recon_distri.sum(1).unsqueeze(1)
            recon_count = recon_count - shift_base
            tissue_mask = tissue_mask.squeeze()
            recon_count = recon_count * tissue_mask

            recon_uwph = recon_count * 2 * math.pi + image

            tissue_mask = torch.unsqueeze(tissue_mask, 0)
            tissue_mask = torch.unsqueeze(tissue_mask, 0)

            loss1 = TVLoss(recon_uwph, tissue_mask)
            loss2, Diff = Laploss(image, recon_uwph)
            # loss, Diff = MskedResidueLoss(image, recon_uwph)
            loss = loss1 + loss2

            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            time_end = time.time()

            # You may comment the learning decay here.
            # vlr meaning variable learning rate
            # clr meaning constant learning rate
            if inner % 10 == 0:
                for params in optimizer1.param_groups:
                    params['lr'] *= 0.9

            if inner % 10 == 0:
                print('Outside: Epoch : %d, Loss1: %f, Loss2: %f, lr1: %f,  used time: %.2f s' %
                # print('Outside: Epoch : %d, Loss1: %f, lr1: %f,  used time: %d s' %
                      (inner, loss1, loss2, optimizer1.param_groups[0]['lr'], time_end - time_start))

                recon_count_save = torch.squeeze(recon_count)
                recon_count_save = torch.round(recon_count_save)
                recon_save = recon_count_save.float()
                recon_save = recon_save.cpu().detach().numpy()

                # InferenceName = Inferencetype
                # Set the testing Echo number and Mark the test feature
                InferenceName = Inferencetype + ('_Echo%s' % index)
                # path_NII = SaveDir_NIFTI + ModelName + '_' + InferenceName + ('_%s_Iter.nii' % inner)
                path_NII = SaveDir_NIFTI + ('Echo_%s_vlr/' % index) + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_vlrNS10K.nii' % inner)
                # path_NII = SaveDir_NIFTI + ('Echo_%s_clr/' % index) + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_vlr.nii' % inner)
                nib.save(nib.Nifti1Image(recon_save, np.eye(4)), path_NII)

                recon_uwph_save = torch.squeeze(recon_uwph)
                image_squeeze = torch.squeeze(image)
                recon_uwph_save = recon_count_save * 2 * math.pi + image_squeeze
                recon_uwph_save = recon_uwph_save.float()
                recon_uwph_save = recon_uwph_save.cpu().detach().numpy()

                # path_UWP = SaveDir_NIFTI + 'UWP/UWP_' + ModelName + '_' + InferenceName + ('_SGTV_%s_Iter.nii' % inner)
                path_UWP = SaveDir_NIFTI + ('Echo_%s_vlrNS/' % index) + 'UWP/UWP_' + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_vlrNS10K.nii' % inner)
                nib.save(nib.Nifti1Image(recon_uwph_save, np.eye(4)), path_UWP)

                Diff = Diff.squeeze()
                Diff = Diff.float()
                Diff = Diff.cpu().detach().numpy()

                # path_Diff = SaveDir_NIFTI + ModelName + '_' + InferenceName + ('_TV_%s_Diff.nii' % inner)
                path_Diff = SaveDir_NIFTI + ('Echo_%s_vlrNS/' % index) + 'Diff/Diff_' + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_vlrNS10K.nii' % inner)
                nib.save(nib.Nifti1Image(Diff, np.eye(4)), path_Diff)

    print('Reconstruction Ends, Going to the storge path')