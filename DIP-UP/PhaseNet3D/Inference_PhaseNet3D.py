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
    # with torch.no_grad():
    FileNo = 1  # file identifier
    # SourceDir = '/home/Student/s4564445/mrf/inference_con/code/multi/XYDIPtemp/'
    # SourceDir = 'D:/QSMData/PhaseUnwrapping/NetworksTemp/Python/NewSets/DIPtemp/'
    SourceDir = '/scratch/itee/xy_BFR/PhaseUnwrapping/recon/ChrisResults/'

    ReconType = 'PhaseNetVer2_3D/'  # Update
    # SaveDir_MAT = SourceDir + 'NewBrain/InVivo/recons/MAT/'

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
    PHUnet.load_state_dict(torch.load(LoadModel, map_location=torch.device('cpu')))  # ZXY
    # PHUnet.load_state_dict(torch.load(LoadModel))  # ZXY
    PHUnet = PHUnet.to(device)  # Model load to CPU?
    PHUnet.eval()

    optimizer1 = optim.RMSprop(PHUnet.parameters(), lr=1e-6)
    # optimizer1 = optim.RMSprop(PHUnet.parameters(), lr=1e-4)
    # schedular = StepLR(optimizer1, step_size=50, gamma=0.9)

    # T1 = time.clock()

    for index in range(FileNo, FileNo + No_subs):
        # input_img_files = SourceDir + 'wph_noisy_10ms_010.mat'  # XYZ
        input_img_files = SourceDir + 'NewBrain/InVivo/wph/' + ("wph%s_144.mat" % index)  # XYZ

        print('Loading Test Data No.%s' % index)
        matTest = scio.loadmat(input_img_files, verify_compressed_data_integrity=False)
        # matLap = scio.loadmat(input_lap_files, verify_compressed_data_integrity=False)

        image = matTest['wph']  # XYZ
        # lap = matLap['wph_lap']  # XYZ

        image = np.array(image)
        image = np.real(image)
        image = torch.from_numpy(image)
        image = image.float()

        # img_lap = np.array(lap)
        # img_lap = np.real(img_lap)
        # img_lap = torch.from_numpy(img_lap)
        # img_lap = img_lap.float()

        image = torch.unsqueeze(image, 0)
        image = torch.unsqueeze(image, 0)

        # img_lap = torch.unsqueeze(img_lap, 0)
        # img_lap = torch.unsqueeze(img_lap, 0)

        image = image.to(device)
        # img_lap = img_lap.to(device)

        # Features = torch.cat([image, img_lap], dim=1)  # ZX edit

        tissue_mask = torch.zeros_like(image)
        tissue_mask[image != 0] = 1

        shift_base = 5

        # for params in optimizer1.param_groups:
        #     params['lr'] *= 0.9

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
            # loss, Diff = MskedResidueLoss(image, recon_uwph)
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

                # InferenceName = Inferencetype
                InferenceName = Inferencetype + ('_Echo_%s' % index)
                # path_NII = SaveDir_NIFTI + ModelName + '_' + InferenceName + ('_%s_Iter.nii' % inner)
                path_NII = SaveDir_NIFTI + ('Echo_%s_clr/' % index) + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_clr.nii' % inner)
                nib.save(nib.Nifti1Image(recon_save, np.eye(4)), path_NII)

                recon_uwph_save = torch.squeeze(recon_uwph)
                image_squeeze = torch.squeeze(image)
                recon_uwph_save = recon_count_save * 2 * math.pi + image_squeeze
                recon_uwph_save = recon_uwph_save.float()
                recon_uwph_save = recon_uwph_save.cpu().detach().numpy()

                # path_UWP = SaveDir_NIFTI + 'UWP/UWP_' + ModelName + '_' + InferenceName + ('_SGTV_%s_Iter.nii' % inner)
                path_UWP = SaveDir_NIFTI + ('Echo_%s_clr/' % index) + 'UWP/UWP_' + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_clr.nii' % inner)
                nib.save(nib.Nifti1Image(recon_uwph_save, np.eye(4)), path_UWP)

                # Diff = Diff.squeeze()
                # Diff = Diff.float()
                # Diff = Diff.cpu().detach().numpy()
                #
                recons_softmax_cfd = torch.softmax(recons, dim=1)
                recons_max = torch.max(recons_softmax_cfd, dim=1)[0]
                tissue_mask_SQ = torch.squeeze(tissue_mask)
                recons_max = torch.squeeze(recons_max)
                recons_max = recons_max * tissue_mask_SQ
                recons_max = recons_max.float()
                cfd_map = recons_max.cpu().detach().numpy()

                # # path_Diff = SaveDir_NIFTI + ModelName + '_' + InferenceName + ('_TV_%s_Diff.nii' % inner)
                path_CFD = SaveDir_NIFTI + ('Echo_%s_clr/' % index) + 'CFD/CFD_' + ModelName + '_' + InferenceName + ('_%s_Iter_MixDIP_vlr.nii' % inner)
                nib.save(nib.Nifti1Image(cfd_map, np.eye(4)), path_CFD)





        # print('Saving UWP results')
        # path = SaveDir_NIFTI + 'UWP_' + ModelName + '_' + InferenceName + ('_%s.nii' % idx)
        # nib.save(nib.Nifti1Image(recon_uwph.cpu().numpy(), np.eye(4)), path)


    # T2 = time.clock()
    # RunningTime = T2 - T1
    # print('Time:%s s' % RunningTime)

    print('Reconstruction Ends, Going to the storge path')

    # recons = recons.numpy()
    #
    # # Save .mat file for future confidence map?
    # print('Saving .mat results')
    # path = SaveDir_MAT + ModelName + '_' + InferenceName + ('_%s.mat' % idx)
    # scio.savemat(path, {'recons': recons})
    #
    # # Save .nii file for future confidence map?
    # recons = np.array(recons)
    # recons = np.real(recons)
    # recons = torch.from_numpy(recons)
    # # recons = np.array(recons.cpu().detach())
    #
    # recons = torch.max(recons, dim=1)[1].float()
    # recons_prob = torch.max(recons, dim=1)[0].float()
    # recons_prob = torch.squeeze(recons_prob)
    # recons_count = torch.argmax(recons, dim=1)
    # recons_count = torch.squeeze(recons_count)

    # x = recons_prob

    # recons_count = torch.argmax(recons, dim=1).float().unsqueeze(1)
    # index = recons_count
    # one_hot = torch.zeros_like(x)
    # one_hot = torch.unsqueeze(one_hot, dim=0)
    # one_hot.scatter_(1, index.unsqueeze(0), 1)
    # one_hot = one_hot * index.unsqueeze(0)
    # one_hot = one_hot.detach() + x - x.detach()
    # recons.require_grad = True
    # recons = torch.max(recons, dim=1)[1].unsqueeze(1)
    # recons = torch.squeeze(recons, dim=0)
    # recons = torch.squeeze(recons, dim=0)
    # recons = recons.float()

    # print('Saving Count results')
    # path = SaveDir_
    # NIFTI + ModelName + '_' + InferenceName + ('_%s.nii' % idx)
    # nib.save(nib.Nifti1Image(recons.cpu().numpy(), np.eye(4)), path)

    # base = 5
    # recon_un = recons_count - base