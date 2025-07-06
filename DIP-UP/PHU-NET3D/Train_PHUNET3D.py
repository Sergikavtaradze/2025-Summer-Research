################### train AutoBCS framework #####################
#########  Network Training #################### 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import time
from Unet_2Chan_9Class import *
from TrainingDataLoad_ResidueLoss_2Chan import *
import math
from torch.nn.functional import pad
import numpy as np

ratio = 0.0
threshold = math.pi*(1+ratio)

#########  Section 1: DataSet Load #############
def DataLoad(Batch_size):

    DATA_DIRECTORY = '/scratch/itee/xy_BFR/PhaseUnwrapping/patches/SingleClass/10msTE'
    DATA_LIST_PATH = '/scratch/itee/xy_BFR/PhaseUnwrapping/patches/test_IDs_28800.txt'
    # Switch the 'DATA_DIRECTORY' and 'DATA_LIST_PATH' by your own path

    dst = DataSet(DATA_DIRECTORY,DATA_LIST_PATH)
    print('dataLength: %d'%dst.__len__())
    trainloader = data.DataLoader(dst, batch_size = Batch_size, shuffle=False, drop_last = True)
    return trainloader

#########  Section 2: Loss functions design #############

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

#########  Section 3: Save the network #############
def SaveNet(PHU_Net, epo, enSave = False):
    print('save results')
    #### save the pre
    if enSave:
        pass
    else:
        ModelFoder = '/scratch/itee/xy_BFR/PhaseUnwrapping/recon/ChrisResults/Networks/'
        ModelName = 'PHUNET3D.pth'
        LoadModel = ModelFoder + ModelName
        torch.save(PHU_Net.state_dict(), LoadModel)
        # If you need to itertively save the training model by epoch, use the code below instead:
        # torch.save(PHU_Net.state_dict(), './PHUNet_CEL1GradMSKResidueLoss_2Chan_BaseShift.pth')
        # torch.save(PHU_Net.state_dict(), ("/PHUNet_%EPO.pth" % epo))

#########  Section 4: Model Trainig #############
def TrainNet(PHU_Net, LR = 0.001, Batchsize = 24, Epoches = 45, useGPU = True):
    print('Unet')
    print('DataLoader setting begins')
    trainloader = DataLoad(Batchsize)
    print('Dataloader settting end')

    print('Training Begins')
    criterion1 = nn.CrossEntropyLoss() 

    optimizer1 = optim.RMSprop(PHU_Net.parameters(), lr=LR)

    scheduler1 = LS.MultiStepLR(optimizer1, milestones = [50, 80], gamma = 0.1)
    ## start the timer. 
    time_start=time.time()
    if useGPU:
        if torch.cuda.is_available():
            print(torch.cuda.device_count(), "Available GPUs!")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            PHU_Net = nn.DataParallel(PHU_Net)
            PHU_Net.to(device)
            torch.random.manual_seed(5)

            for epoch in range(1, Epoches + 1):

                '''
                if epoch % 20 == 0:
                    SaveNet(PHU_Net, epoch, enSave = False)
                '''

                acc_loss = 0.0
                for i, data in enumerate(trainloader):

                    Imagess, Laps, labels, name = data
                    Imagess = Imagess.to(device)
                    Laps = Laps.to(device)

                    Labels = labels.to(device)
                    Labels = Labels.to(device=device, dtype=torch.int64)

                    # print('Dim of Input')
                    # print(Imagess.size())

                    # print('Dim of Label')
                    # print(Labels.size())

                    ## zero the gradient buffers
                    optimizer1.zero_grad()
                    ## forward:
                    Features = torch.cat([Imagess, Laps], dim=1)
                    Predictions = PHU_Net(Features)  # ZX edit
                    # Predictions = PHU_Net(Imagess)  # ZX edit  # [24,9,64,64,64]
                    Predictions = torch.squeeze(Predictions, 1)
                    # print('Dim of Prediction')
                    # print(Predictions.size())

                    ## loss
                    loss = criterion1(Predictions, Labels)

                    ## backward
                    loss.backward()
                    ##
                    optimizer1.step()

                    optimizer1.zero_grad()
                    ## print statistical information
                    ## print every 20 mini-batch size
                    if i % 5 == 0:
                        acc_loss = loss.item()
                        time_end = time.time()
                        print('Outside: Epoch : %d, batch: %d, Loss1: %f, lr1: %f,  used time: %d s' %
                            (epoch, i + 1, acc_loss, optimizer1.param_groups[0]['lr'], time_end - time_start))
                scheduler1.step()
        else:
            pass
            print('No Cuda Device!')
            quit()
    print('Training Ends')
    SaveNet(PHU_Net, Epoches)

if __name__ == '__main__':
    ## load laplacian operator; 

    PHU_Net = Unet_2Chan_9Class(4)
    PHU_Net.apply(weights_init)
    PHU_Net.train()

    print(PHU_Net.state_dict)
    print(get_parameter_number(PHU_Net))

    ## train network
    # TrainNet(PHU_Net, LR = 0.001, Batchsize = 12, Epoches = 45 , useGPU = True)
    TrainNet(PHU_Net, LR = 0.0001, Batchsize = 24, Epoches = 45 , useGPU = True)
