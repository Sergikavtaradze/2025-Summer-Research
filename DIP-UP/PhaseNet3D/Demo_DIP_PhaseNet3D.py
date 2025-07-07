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
from torch.optim.lr_scheduler import StepLR

sys.path.append('/clusterdata/uqxzhu14/Script_QSMUnet/xQSM_Python/YangsCodeNew/Unet_Backup/')

from Unet_1Chan_9Class import *
from Train_PhaseNet3D import *

if __name__ == '__main__':

    #########  Section 1: Inference Data and Pre-trained Network Upload #############
    # with torch.no_grad():
    FileNo = 1  # file identifier
    SourceDir = '/scratch/itee/xy_BFR/PhaseUnwrapping/recon/ChrisResults/'

    ReconType = 'PhaseNetVer2_3D/'  

    SaveDir_NIFTI = SourceDir + 'NewBrain/InVivo/recons/NIFTI/' + ReconType
    No_subs = 1  # total number of subjects to be reconstructed

    ModelFoder = SourceDir + 'SomethingGood/'
    ModelName = 'PhaseNetVer2_3D'
    LoadModel = ModelFoder + ModelName + '.pth'

    Inferencetype = 'InVivo'

    print('DIP type: MixLossDIP with Laplacian loss and TV-Masked Loss')
    print('InVivo,  Echo1, Noise sigma = 0.1, constent learning rate with 10e-6')

    print('Network Loading')
    ## load pre-trained network
    PHUnet = Unet_1Chan_9Class(4)
    PHUnet = nn.DataParallel(PHUnet)
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PHUnet.load_state_dict(torch.load(LoadModel, map_location=torch.device('cpu')))

    PHUnet = PHUnet.to(device)
    PHUnet.eval()

    optimizer1 = optim.RMSprop(PHUnet.parameters(), lr=1e-6)

    # T1 = time.clock()

    for index in range(FileNo, FileNo + No_subs):
        input_img_files = SourceDir + 'NewBrain/InVivo/wph/' + ("wph%s_144.mat" % index)  # XYZ

        print('Loading Test Data No.%s' % index)
        matTest = scio.loadmat(input_img_files, verify_compressed_data_integrity=False)
        image = matTest['wph']

        image = np.array(image)
        image = np.real(image)
        image = torch.from_numpy(image)
        image = image.float()

        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        image = image.to(device)
        tissue_mask = torch.zeros_like(image)
        tissue_mask[image != 0] = 1

        shift_base = 5

        # We illustrate the DIP-mode of 'clr' here. If you need to try 'vlr' mode, you may cancel the comment of learning decay here.
        # vlr meaning variable learning rate
        # clr meaning constant learning rate
        
        # for params in optimizer1.param_groups:
        #     params['lr'] *= 0.9

        #########  Section 2: Deep Image Prior to enhance the Initial output #############
        # with torch.no_grad():
        time_start = time.time()
        for inner in range(2001):

            recons = PHUnet(image)  # Check line No.22
            # recons = PHUnet(Features)  # Check line No.22

            # print('Recon Initialized')

            # recons = recons.to('cpu')

            recons_softmax = torch.softmax(recons, dim=1)
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
            loss = loss1 + loss2

            loss.backward()
            optimizer1.step()
            # schedular.step()
            optimizer1.zero_grad()

            time_end = time.time()

            # if inner % 10 == 0:
            #     for params in optimizer1.param_groups:
            #         params['lr'] *= 0.9

            if inner % 10 == 0:
                print('Outside: Epoch : %d, Loss1: %f, Loss2: %f, lr1: %f,  used time: %.2f s' %
                # print('Outside: Epoch : %d, Loss1: %f, lr1: %f,  used time: %d s' %
                      (inner, loss1, loss2, optimizer1.param_groups[0]['lr'], time_end - time_start))

                recon_count_save = torch.squeeze(recon_count)
                recon_count_save = torch.round(recon_count_save)
                recon_save = recon_count_save.float()
                recon_save = recon_save.cpu().detach().numpy()

                InferenceName = Inferencetype + ('_Echo_%s' % index)
                path_NII = SaveDir_NIFTI + ('Echo_%s_clr/' % index) + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_clr.nii' % inner)
                nib.save(nib.Nifti1Image(recon_save, np.eye(4)), path_NII)

                recon_uwph_save = torch.squeeze(recon_uwph)
                image_squeeze = torch.squeeze(image)
                recon_uwph_save = recon_count_save * 2 * math.pi + image_squeeze
                recon_uwph_save = recon_uwph_save.float()
                recon_uwph_save = recon_uwph_save.cpu().detach().numpy()

                path_UWP = SaveDir_NIFTI + ('Echo_%s_clr/' % index) + 'UWP/UWP_' + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_clr.nii' % inner)
                nib.save(nib.Nifti1Image(recon_uwph_save, np.eye(4)), path_UWP)

                recons_softmax_cfd = torch.softmax(recons, dim=1)
                recons_max = torch.max(recons_softmax_cfd, dim=1)[0]
                tissue_mask_SQ = torch.squeeze(tissue_mask)
                recons_max = torch.squeeze(recons_max)
                recons_max = recons_max * tissue_mask_SQ
                recons_max = recons_max.float()
                cfd_map = recons_max.cpu().detach().numpy()
                path_CFD = SaveDir_NIFTI + ('Echo_%s_clr/' % index) + 'CFD/CFD_' + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_vlr.nii' % inner)
                nib.save(nib.Nifti1Image(cfd_map, np.eye(4)), path_CFD)

    print('Reconstruction Ends, Going to the storge path')

