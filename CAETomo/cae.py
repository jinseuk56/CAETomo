import time
import hyperspy.api as hys
from drca import drca, reshape_coeff
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tabulate import tabulate

color_rep = ["black", "red", "green", "blue", "purple", "orange"]
cm_rep = ["Greys", "Reds", "Greens", "Blues", "Purples", "Oranges"]

class cae():
    def __init__(self, adr, dat_dim, dat_unit, 
                 cr_range=None, dat_scale=1, 
                 rescale=True, DM_file=False):
        self.load_data = drca(adr, dat_dim=dat_dim, 
                         dat_unit=dat_unit, cr_range=cr_range,
                         dat_scale=dat_scale, rescale=rescale,
                         DM_file=DM_file)
        
    def binning(self, bin_y, bin_x, str_y, str_x, offset=0, rescale_0to1=True):
        self.load_data.binning(bin_y, bin_x, str_y, str_x, offset, rescale_0to1)

    def make_input(self, min_val=0.0, max_normalize=True, rescale_0to1=False, log_scale=False, radial_flat=True, w_size=0, radial_range=None, final_dim=1):
        self.load_data.make_input(min_val, max_normalize, rescale_0to1, log_scale, radial_flat, w_size, radial_range, final_dim)

    def create_mini_batch(self, batch_size):
        self.mini_batches = [self.load_data.dataset_input[k:k+batch_size] for k in range(0, len(self.load_data.dataset_input), batch_size)]
        print("number of mini-batches", len(self.mini_batches))

    def create_autoencoder(self, encoder="CAE1D", decoder="linear",
                           num_comp=5, channels=[8, 16, 32, 5], 
                           kernels=[64, 32, 16, 7], pooling=[2, 2, 2, 2],
                           data_parallel=False, cuda_device=None):
        
        self.num_comp = num_comp
        self.channels = channels
        self.kernels = kernels
        self.pooling = pooling
        self.data_parallel = data_parallel
        self.cuda_device = cuda_device

        self.dat_dim = []
        tmp_dim = self.load_data.num_dim
        for i in range(len(self.kernels)):
            tmp_dim += (-self.kernels[i]+1)
            tmp_dim /= self.pooling[i]
            self.dat_dim.append(int(tmp_dim))
            
        self.kernels[-1] = self.dat_dim[-2] - self.pooling[-1] + 1
        self.dat_dim[-1] = 1

        print("dimensions of the encoded data", self.dat_dim)
        print("kernels", self.kernels)
        print("channels", self.channels)
        print("poolings", self.pooling)


        if encoder == "CAE1D":
            self.enc_model = CAE1D_encoder(self.load_data.num_dim, self.channels, self.kernels, self.pooling)

        if self.data_parallel:
            self.enc_model = nn.DataParallel(self.enc_model)

        self.enc_model.cuda(self.cuda_device)
        for p in self.enc_model.parameters():
            if p.requires_grad:
                print(p.data.shape)
        train_params = sum(p.numel() for p in self.enc_model.parameters() if p.requires_grad)
        print(train_params)
        print(self.enc_model)

        if decoder == "linear":
            self.dec_model = linFE_decoder(self.num_comp, self.load_data.num_dim)

        if self.data_parallel:
            self.dec_model = nn.DataParallel(self.dec_model)
        self.dec_model.cuda(self.cuda_device)

    def training(self, optimizer="ADAM", loss_fn="MSE", l_rate=0.001, n_epoch=100, orthogonal_initialize=True):
        
        self.params = list(self.enc_model.parameters()) + list(self.dec_model.parameters())

        if optimizer == "ADAM":
            self.optimizer = optim.Adam(self.params, lr=l_rate)

        if orthogonal_initialize:
            nn.init.orthogonal_(self.dec_model.decoder[0].weight)

        self.n_epoch = n_epoch

        start = time.time()

        self.ae_coeffs = []
        self.ae_bias = []
        self.losses = []
        for epoch in range(self.n_epoch):
            tmp_loss = 0
            for i, m_batch in enumerate(self.mini_batches):
                
                x = torch.from_numpy(self.mini_batches[i])
                x = x.to(torch.float32)
                x = x.to(self.cuda_device)
                x.requires_grad_(requires_grad=False)
                
                encoded = self.enc_model(x)
                decoded = self.dec_model(encoded)
                
                if loss_fn == "BCE":
                    main_loss = F.binary_cross_entropy(decoded, x, reduction="mean")
                    tmp_loss += main_loss.item()
                elif loss_fn == "MSE":
                    main_loss = F.mse_loss(decoded, x, reduction="mean")
                    tmp_loss += main_loss.item()
                
                self.optimizer.zero_grad()
                main_loss.backward()
                self.optimizer.step()
                
                if self.data_parallel:
                    self.dec_model.module.decoder[0].weight.data.clamp_(min=0.0)

                else:
                    self.dec_model.decoder[0].weight.data.clamp_(min=0.0)

                if epoch == n_epoch-1:
                    coeff_batch = encoded.data.cpu().numpy().tolist()
                    self.ae_coeffs.extend(coeff_batch)            
            
            self.losses.append(tmp_loss)
            
            if epoch == 0:
                print(torch.cuda.memory_summary(device=self.cuda_device))
            
            if (epoch+1) % int(n_epoch/10) == 0:
                print(tabulate([
                                ["epoch", epoch+1], 
                                ["loss", main_loss.item()],
                                ]))
                print("%.2f minutes have passed"%((time.time()-start)/60))
                
                fig, ax = plt.subplots(1, self.num_comp, figsize=(5*self.num_comp, 5))

                if self.data_parallel:
                    fc = self.dec_model.module.decoder[0].weight.data.cpu().numpy()

                else:
                    fc = self.dec_model.decoder[0].weight.data.cpu().numpy()

                for i in range(self.num_comp):
                    ax[i].plot(self.load_data.dat_dim_range, fc[:, i])
                fig.tight_layout()
                plt.show()

        print("The training has been finished.")

    def show_result(self, save_result=False, save_prefix=None):

        self.ae_coeffs = np.asarray(self.ae_coeffs)

        if self.data_parallel:
            self.ae_comp_vectors = self.dec_model.module.decoder[0].weight.data.cpu().numpy().T

        else:
            self.ae_comp_vectors = self.dec_model.decoder[0].weight.data.cpu().numpy().T

        print(self.ae_coeffs.shape)
        print(self.ae_comp_vectors.shape)

        coeffs = np.zeros_like(self.ae_coeffs)
        coeffs[self.load_data.ri] = self.ae_coeffs.copy()
        self.coeffs_reshape = reshape_coeff(coeffs, self.load_data.data_shape)

        peak_ind = np.argmax(self.ae_comp_vectors, axis=1)
        peak_pos = self.load_data.dat_dim_range[peak_ind]
        peak_order = np.argsort(peak_pos)

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for i in range(self.num_comp):
            ax.plot(self.load_data.dat_dim_range, self.ae_comp_vectors[i], "-", c=color_rep[np.where(peak_order==i)[0][0]], label="loading vector %d"%(i+1), linewidth=3)
        ax.set_xlabel("Energy Loss (eV)", fontsize=30)
        ax.set_ylabel("Intensity (arb. unit)", fontsize=30)
        ax.tick_params(axis="both", labelsize=30)
        fig.tight_layout()
        plt.show()

        for i in range(self.num_comp):
            fig, ax = plt.subplots(1, self.load_data.num_img, figsize=(10*self.load_data.num_img, 10))
            min_val = np.min(coeffs[:, i])
            max_val = np.max(coeffs[:, i])
            for j in range(self.load_data.num_img):
                tmp = ax[j].imshow(self.coeffs_reshape[j][:, :, i], 
                                    vmin=min_val, vmax=max_val, cmap=cm_rep[np.where(peak_order==i)[0][0]])
                ax[j].axis("off")
            fig.tight_layout()
        plt.show()

        if save_result:
            for i in range(self.num_comp):
                tilt_series = []
                for j in range(self.load_data.num_img):
                    tilt_series.append(self.coeffs_reshape[j][:, :, i].astype(np.float32))
                tilt_series = np.asarray(tilt_series)
                tifffile.imsave(save_prefix+"_SC_%02d_feature_maps.tif"%(np.where(peak_order==i)[0][0]), tilt_series)
            tifffile.imwrite(save_prefix+"spectral_component.tif", self.ae_comp_vectors)
            tifffile.imwrite(save_prefix+"feature_maps.tif", self.coeffs_reshape)
            print("Saving completed.")

class CAE1D_encoder(nn.Module):
    def __init__(self, input_size, channels, kernels, poolings):
        super(CAE1D_encoder, self).__init__()

        self.input_size = input_size

        enc_net = []
        enc_net.append(nn.Conv1d(1, channels[0], kernels[0], bias=True))
        enc_net.append(nn.BatchNorm1d(channels[0]))
        enc_net.append(nn.ReLU())
        if poolings[0] != 1:
            enc_net.append(nn.AvgPool1d(poolings[0]))
        for i in range(1, len(channels)):
            enc_net.append(nn.Conv1d(channels[i-1], channels[i], kernels[i], bias=True))
            enc_net.append(nn.BatchNorm1d(channels[i]))
            enc_net.append(nn.ReLU())
            if poolings[i] != 1:
                enc_net.append(nn.AvgPool1d(poolings[i]))

        enc_net.append(nn.Flatten())
        self.encoder = nn.Sequential(*enc_net)

    def forward(self, x):
        x = x.view(-1, 1, self.input_size)
        return self.encoder(x)


class linFE_decoder(nn.Module):
    def __init__(self, z_dim, in_dim):
        super(linFE_decoder, self).__init__()
        
        self.z_dim = z_dim
        self.in_dim = in_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.in_dim, bias=False),
            nn.Hardsigmoid(),
        )
        
    def forward(self, z):       
        return self.decoder(z)