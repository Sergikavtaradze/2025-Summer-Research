################### train AutoBCS framework #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from Unet_1Chan_9Class import *
from TrainingDataLoad_ResidueLoss_1Chan import *
import math
from torch.nn.functional import pad
import numpy as np

ratio = 0.0
threshold = math.pi*(1+ratio)

#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):
    # DATA_DIRECTORY = 'D:/QSMData/PhaseUnwrapping/10msTE/patches/'
    DATA_DIRECTORY = '/scratch/itee/xy_BFR/PhaseUnwrapping/patches/SingleClass/10msTE'
    # DATA_LIST_PATH = 'C:/Users/s4513947/Downloads/Python/test_IDs_28800.txt'
    DATA_LIST_PATH = '/scratch/itee/xy_BFR/PhaseUnwrapping/patches/test_IDs_28800.txt'

    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=False, drop_last = True)
    return trainloader

#########  Section 2: Loss Functions Design #############
def Gradient(image):

    diff_x = torch.diff(image, dim=2)
    diff_y = torch.diff(image, dim=3)
    diff_z = torch.diff(image, dim=4)

    '''
    diff_x_pad = pad(diff_x ** 2, [0, 0, 0, 0, 0, 1])
    diff_y_pad = pad(diff_y ** 2, [0, 0, 0, 1])
    diff_z_pad = pad(diff_z ** 2, [0, 1])
    '''

    diff_x_pad = pad(diff_x, [0, 0, 0, 0, 0, 1])
    diff_y_pad = pad(diff_y, [0, 0, 0, 1])
    diff_z_pad = pad(diff_z, [0, 1])

    gradient = torch.sqrt((diff_x_pad ** 2 + diff_y_pad ** 2 + diff_z_pad ** 2)/3)

    return gradient


def MskedResidueLoss(input, recon_unimg):

    # L2 = nn.MSELoss()
    Res = nn.L1Loss()

    grad_input = Gradient(input)  # diff is the masked edge of 1-D gradients
    grad_recon = Gradient(recon_unimg)

    grad_mask = torch.zeros_like(grad_input)
    grad_mask[grad_input < threshold] = 1

    tissue_mask = torch.zeros_like(grad_input)
    tissue_mask[grad_input != 0] = 1

    grad_input = grad_input * grad_mask
    grad_recon = grad_recon * grad_mask
    grad_recon = grad_recon * tissue_mask

    # numvox = np.count_nonzero(Msk.cpu())
    # How to load into CPU first, Answer: Change (Msk) to (Msk.cpu())

    resloss = Res(grad_input, grad_recon)

    Diff = grad_input - grad_recon

    return resloss, Diff


def LapLacian(img, device):
    SourceDir = '/home/Student/s4564445/mrf/inference_con/code/multi/XYDIPtemp/'
    # SourceDir = 'D:/QSMData/PhaseUnwrapping/NetworksTemp/Python/NewSets/DIPtemp/'
    # SourceDir = '/scratch/itee/xy_BFR/PhaseUnwrapping/recon/ChrisResults/'
    load_dker = SourceDir + 'dker.mat'
    dker = scio.loadmat(load_dker)['dker']  # Replace
    dker = nn.Parameter(torch.from_numpy(dker).unsqueeze(0).unsqueeze(0).float(), requires_grad=False).to(device)

    return F.conv3d(img, weight=dker, stride=1, padding=1)


def Laploss(input, recon_unimg, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    Res = nn.L1Loss()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cos_input = torch.cos(input)
    sin_input = torch.sin(input)

    CosImage_lAP = LapLacian(cos_input, device)
    SinImage_lAP = LapLacian(sin_input, device)

    # CosImage_lAP = CosImage_lAP.to(device)
    # SinImage_lAP = SinImage_lAP.to(device)

    UWPlabel_LAP = cos_input * SinImage_lAP - sin_input * CosImage_lAP
    Recon_LAP = LapLacian(recon_unimg, device)

    Diff = Recon_LAP - UWPlabel_LAP

    Laploss = Res(Recon_LAP, UWPlabel_LAP)

    return Laploss, Diff

def TVLoss(image, mask):

    diff_x = torch.diff(image, dim=2)
    diff_y = torch.diff(image, dim=3)
    diff_z = torch.diff(image, dim=4)

    diff_mask_x = torch.diff(mask, dim=2)
    diff_mask_y = torch.diff(mask, dim=3)
    diff_mask_z = torch.diff(mask, dim=4)

    # diff_x_pad = pad(diff_x ** 2, [0, 0, 0, 0, 0, 1])
    # diff_y_pad = pad(diff_y ** 2, [0, 0, 0, 1])
    # diff_z_pad = pad(diff_z ** 2, [0, 1])

    diff_x_pad = pad(diff_x, [0, 0, 0, 0, 0, 1])
    diff_y_pad = pad(diff_y, [0, 0, 0, 1])
    diff_z_pad = pad(diff_z, [0, 1])

    diff_maskx_pad = pad(diff_mask_x, [0, 0, 0, 0, 0, 1])
    diff_masky_pad = pad(diff_mask_y, [0, 0, 0, 1])
    diff_maskz_pad = pad(diff_mask_z, [0, 1])

    loss_3D = torch.abs((1 - diff_maskx_pad) * diff_x_pad) + torch.abs((1 - diff_masky_pad)*diff_y_pad) + torch.abs((1 - diff_maskz_pad) * diff_z_pad)

    true_loss = torch.sum(loss_3D)

    shape_image = image.shape
    dimx = shape_image[2]
    dimy = shape_image[3]
    dimz = shape_image[4]

    loss = true_loss/(dimx*dimy*dimz)

    # gradient = torch.sqrt((diff_x_pad ** 2 + diff_y_pad ** 2 + diff_z_pad ** 2)/3)

    return loss

#########  Section 3: Save the network, iteratively by the epoch #############
def SaveNet(PhaseNet, epo, enSave = False):
    print('save results')
    #### save the pre
    if enSave:
        pass
    else:
        ModelFoder = '/scratch/itee/xy_BFR/PhaseUnwrapping/recon/ChrisResults/Networks/'
        ModelName = 'PhaseNet3D.pth'
        LoadModel = ModelFoder + ModelName
        torch.save(PhaseNet.state_dict(), LoadModel)
        # If you need to iteratively save the trained networks, cancel the comment below:
        # torch.save(PhaseNet.state_dict(), './PHUNet_CEL1GradMSKResidueLoss_2Chan_BaseShift.pth')
        # torch.save(PhaseNet.state_dict(), ("/PHUNet_%EPO.pth" % epo))

#########  Section 4: Start training #############
def TrainNet(PhaseNet, LR = 0.001, Batchsize = 24, Epoches = 45, useGPU = True):
    print('Unet')
    print('DataLoader setting begins')
    trainloader = DataLoad(Batchsize)
    print('Dataloader settting end')

    print('Training Begins')
    criterion1 = nn.CrossEntropyLoss()  # XYZ 27062022
    criterion2 = nn.L1Loss()
    # criterion2 = nn.MSELoss(reduction='sum')

    # optimizer1 = optim.Adam(PhaseNet.parameters())
    optimizer1 = optim.RMSprop(PhaseNet.parameters(), lr=LR)

    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [50, 80], gamma = 0.1)
    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            PhaseNet = nn.DataParallel(PhaseNet)
            PhaseNet.to(device)
            torch.random.manual_seed(5)

            for epoch in range(1, Epoches + 1):

                '''
                if epoch % 20 == 0:
                    SaveNet(PhaseNet, epoch, enSave = False)
                '''

                acc_loss = 0.0
                for i, data in enumerate(trainloader):

                    Imagess, labels, name = data
                    Imagess = Imagess.to(device)

                    Labels = labels.to(device)
                    Labels = Labels.to(device=device, dtype=torch.int64)

                    # print('Dim of Input')
                    # print(Imagess.size())

                    # print('Dim of Label')
                    # print(Labels.size())

                    ## zero the gradient buffers
                    optimizer1.zero_grad()
                    ## forward:
                    Predictions = PhaseNet(Imagess)
                    Predictions = torch.squeeze(Predictions, 1)
                    # print('Dim of Prediction')
                    # print(Predictions.size())

                    Max_Prediction = torch.max(Predictions, dim=1)[1].unsqueeze(1)
                    Prediction_OriBase = Max_Prediction - 5  # Base RE-Shifting Ver.2
                    # print('Dim of Max_Prediction')
                    # print(Max_Prediction.size())

                    # print('Dim of Max_Imagess')
                    # print(Max_Imagess.size())

                    Recon_UWP_Shift = 2 * Max_Prediction * math.pi + Imagess  # [24,1,64,64,64]
                    Recon_UWP_Orig = 2 * Prediction_OriBase * math.pi + Imagess  # [24,1,64,64,64]

                    Label_Unsq = torch.unsqueeze(Labels, 1)  # [24,1,64,64,64]
                    Label_UWPs = 2 * Label_Unsq * math.pi + Imagess
                    # print('Dim of Label_Unsq')
                    # print(Label_Unsq.size())

                    ## loss
                    loss1 = criterion1(Predictions, Labels)
                    loss2 = MskedResidueLoss(Imagess, Recon_UWP_Orig)  # XYZ edit ## Breakpoint2
                    loss3 = criterion2(Recon_UWP_Shift, Label_UWPs)

                    # loss2 = loss2.mean()
                    lamda2 = 1
                    lamda3 = 1
                    loss = loss1 + lamda2 * loss2 + lamda3 * loss3
                    ## backward
                    loss.backward()
                    # loss1.backward()
                    ##
                    optimizer1.step()

                    optimizer1.zero_grad()
                    ## print statistical information
                    ## print every 20 mini-batch size
                    if i % 5 == 0:
                        # acc_loss1 = loss1.item()
                        acc_loss1 = loss1.item()
                        acc_loss2 = loss2.item()
                        acc_loss3 = loss3.item()
                        time_end = time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, Loss2: %f, Loss3: %f, lr1: %f,  used time: %d s' %
                            (epoch, i + 1, acc_loss1, acc_loss2, acc_loss3, optimizer1.param_groups[0]['lr'], time_end - time_start))
                scheduler1.step()
        else:
            pass
            print('No Cuda Device!')
            quit()
    print('Training Ends')
    SaveNet(PhaseNet, Epoches)

if __name__ == '__main__':
    ## load laplacian operator; 

    PhaseNet = Unet_1Chan_9Class(4)
    PhaseNet.apply(weights_init)
    PhaseNet.train()

    print(PhaseNet.state_dict)
    print(get_parameter_number(PhaseNet))

    ## train network
    # TrainNet(PhaseNet, LR = 0.001, Batchsize = 12, Epoches = 45 , useGPU = True)
    TrainNet(PhaseNet, LR = 0.0001, Batchsize = 24, Epoches = 45 , useGPU = True)
