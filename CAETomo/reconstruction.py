import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as tkf
import tifffile
import time
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import linalg as LA
import torchkbnufft as tkbn

class CSET():
    def __init__(self, adr, angles, cuda_device=None):
        
        self.num_adr = len(adr)
        
        self.ts = []
        for i in range(self.num_adr):
            self.ts.append(tifffile.imread(adr[i]))
        
        if self.ts[0].shape[1] != self.ts[0].shape[2]:
            print("Warning: The width and height of tilt-series images must be the same")

        
        self.edge_length = self.ts[0].shape[1]
        
        self.ta = angles
        if self.ts[0].shape[0] != len(self.ta):
            print("Warning: the number of angles is wrong")

        self.n_tilt = len(self.ta)
        
        self.cuda_device = cuda_device
        print("cuda device: ", self.cuda_device)
        
        
    def NUFFT(self, pad_ratio=2.0, tilt_axis="horizontal"):
        
        self.pad_ratio = pad_ratio
                
        if self.pad_ratio < 1.0:
            print("the pad ratio must be larget than 1.0")
            return
        
        self.pad_length = int(self.edge_length * self.pad_ratio)
        self.tilt_axis = tilt_axis
        
        self.NUFFT_object()
        self.create_coord()
        
        
    def create_coord(self):
        
        r = np.linspace(-np.pi, np.pi, self.pad_length, endpoint=False)[np.newaxis, :].repeat(self.edge_length, axis=0).flatten()
        z = np.linspace(-np.pi, np.pi, self.edge_length, endpoint=False)[:, np.newaxis].repeat(self.pad_length)
        
        
        k_coordinates = []
        for angle in self.ta:
            tmp = np.zeros((self.edge_length * self.pad_length, 3))
            tmp[:, 0] = r * np.cos(angle)
            tmp[:, 1] = r * np.sin(angle)
            tmp[:, 2] = z
            k_coordinates.append(tmp)

        k_coordinates = np.asarray(k_coordinates).reshape(-1, 3)
        self.ktraj = k_coordinates.T
        self.ktraj = torch.tensor(self.ktraj).to(torch.float32)
        self.ktraj = self.ktraj.to(self.cuda_device)
        
        
    def NUFFT_object(self):

        im_size = (self.edge_length, self.edge_length, self.edge_length)
        grid_size = (self.pad_length, self.pad_length, self.pad_length)

        self.nufft_ob = tkbn.KbNufft(
                im_size=im_size,
                grid_size=grid_size,
            ).to(torch.complex64)

        self.nufft_ob = self.nufft_ob.to(self.cuda_device)
    
    
    def fourier_radial_sampling(self, ts):
        
        ts = ts/np.max(ts)
        
        ext_ts = np.zeros((self.n_tilt, self.pad_length, self.edge_length))

        if self.tilt_axis=="vertical":
            ext_ts[:, int((self.pad_length-self.edge_length)/2):-int(np.ceil((self.pad_length-self.edge_length)/2)), :] = np.rot90(ts, axes=(1, 2))
        else:
            ext_ts[:, int((self.pad_length-self.edge_length)/2):-int(np.ceil((self.pad_length-self.edge_length)/2)), :] = ts

        FT_ts = []
        for img in ext_ts:
            tmp = np.fft.ifftshift(img.astype(np.complex64))
            tmp = np.fft.fft2(tmp)
            tmp = np.fft.fftshift(tmp).T
            FT_ts.append(tmp)

        FT_ts = np.asarray(FT_ts)
        
        kdata = torch.tensor(FT_ts.flatten()).to(torch.complex64).unsqueeze(0).unsqueeze(0)
        kdata = kdata.to(self.cuda_device)
        
        return kdata

    def recontruct(self, n_iter=200, lmbd_l1=5E-4, lmbd_tv=3.0, l_rate=2E-5, save_result=True, save_adr="reconstructed_", verbose=True):
        
        self.reconstruction_result = []
        print(tabulate([
                        ["total number of iteration", n_iter], 
                        ["lambda_L1", lmbd_l1],
                        ["lambda_TV", lmbd_tv],
                        ["learning rate", l_rate],
                        ["save the result (tiff)", save_result],
                        ["prefix for save", save_adr],
                        ]))
        
        solution = np.random.randn(self.edge_length, self.edge_length, self.edge_length)
        solution = torch.tensor(solution).to(torch.complex64).unsqueeze(0).unsqueeze(0)
        solution = solution.cuda(self.cuda_device)
        solution.requires_grad_(requires_grad=True)
        nn.init.xavier_uniform_(solution)
        
        CS_optimizer = optim.SGD([solution], lr=l_rate)
        
        for i in range(self.num_adr):
            
            kdata = self.fourier_radial_sampling(self.ts[i])
            nn.init.xavier_uniform_(solution)
            
            for epoch in range(n_iter):

                fft_x = self.nufft_ob(solution, self.ktraj)
                main_loss = LA.norm((fft_x - kdata).squeeze(), 2)

                l1_reg = torch.sum(torch.abs(fft_x))
                l1_reg *= lmbd_l1

                gz, gy, gx = torch.gradient(torch.real(solution.squeeze()))
                tv_reg = LA.norm(gz.flatten(), 1) + LA.norm(gy.flatten(), 1) + LA.norm(gx.flatten(), 1)

                tv_reg *= lmbd_tv

                total_loss = main_loss + l1_reg + tv_reg

                CS_optimizer.zero_grad()
                total_loss.backward()
                CS_optimizer.step()

                solution.real.data.clamp_(min=0.0)

                if verbose and (epoch+1) % int(n_iter/5) == 0:
                    
                    fig, ax = plt.subplots(3, 6, figsize=(18, 9))
        
                    tmp_gz = np.real(gz.squeeze().data.cpu().numpy())
                    tmp_gy = np.real(gy.squeeze().data.cpu().numpy())
                    tmp_gx = np.real(gx.squeeze().data.cpu().numpy())
                    tmp_sol = np.real(solution.squeeze().data.cpu().numpy())
                    for j, ind in enumerate([int(self.edge_length/5), int(self.edge_length*2/5), int(self.edge_length*3/5)]):
                            ax[j][0].imshow(tmp_gz[:, ind, :], cmap="inferno")
                            ax[j][0].axis("off")
                            ax[j][1].imshow(tmp_sol[:, ind, :], cmap="inferno")
                            ax[j][1].axis("off")
                            ax[j][2].imshow(tmp_gy[:, :, ind], cmap="inferno")
                            ax[j][2].axis("off")
                            ax[j][3].imshow(tmp_sol[:, :, ind], cmap="inferno")
                            ax[j][3].axis("off")
                            ax[j][4].imshow(tmp_gx[ind, :, :], cmap="inferno")
                            ax[j][4].axis("off")
                            ax[j][5].imshow(tmp_sol[ind, :, :], cmap="inferno")
                            ax[j][5].axis("off")

                    fig.tight_layout()
                    plt.show()
                    
                    
                    print(tabulate([
                                    ["epoch", epoch+1], 
                                    ["total loss", total_loss.item()],
                                    ["-main loss", main_loss.item(), main_loss.item()*100/total_loss.item()],
                                    ["-L1 loss", l1_reg.item(), l1_reg.item()*100/total_loss.item()], 
                                    ["-TV reg.", tv_reg.item(), tv_reg.item()*100/total_loss.item()],
                                    ]))
            print("The %d optimization has been finished."%(i+1))

            restored_object = solution.squeeze().data.cpu().numpy()
            
            self.reconstruction_result.append(np.real(restored_object))
            
            if save_result:
                tifffile.imwrite(save_adr+"%02d.tif"%(i+1), np.real(restored_object).astype(np.float32))
