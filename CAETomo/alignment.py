# Tilt Axis Alignment
# Jinseok Ryu (jinseuk56@gmail.com)
# 

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import ipywidgets as pyw
import matplotlib.gridspec as gridspec
import scipy.signal as signal
from skimage.registration import phase_cross_correlation
import cv2


class slice_viewer:
    def __init__(self, ax, X):
        self.ax = ax
        
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0
        self.threshold = 10

        self.ax.set_title("Slice No.%d, Percentile %d"%(self.ind+1, self.threshold))
        
        self.im = ax.imshow(kill_low_int(self.X[self.ind], self.threshold), cmap="inferno")
        self.ax.axis("off")
        self.update()

    def on_press(self, event):
        if event.key == "up":
            self.threshold += 1
        elif event.key == "down":
            self.threshold -= 1
        elif event.key == "up" or event.key == "right":
            self.ind = (self.ind + 1) % self.slices
        elif event.key == "down" or event.key == "left":
            self.ind = (self.ind - 1) % self.slices
        else:
            return
        self.update()

    def update(self):
        self.im.set_data(kill_low_int(self.X[self.ind], self.threshold))
        self.ax.set_title("Slice No.%d, Percentile %d"%(self.ind+1, self.threshold))
        self.im.axes.figure.canvas.draw()

        
        
class tilt_series_alignment():
    def __init__(self, ref_adr, angles, img_adr=None):
        
        self.data_original = tifffile.imread(ref_adr)
        self.data = self.data_original.copy()
        self.num_img = len(self.data)
        self.child_ = False
        self.imgs = []
        if img_adr:
            self.child_ = True
            for adr in img_adr:
                self.imgs.append(tifffile.imread(adr))
        
        self.angles = angles
        
    def prepare_alignment(self, pad=0):
        
        self.pad = pad
        self.pad_ = False
        
        if pad:
            self.pad_ = True
            self.n_row = self.data_original.shape[1]
            self.n_col = self.data_original.shape[2]
            pad_data = np.zeros((self.num_img, self.n_row+2*self.pad, self.n_col+2*self.pad))
            pad_data[:, self.pad:self.pad+self.n_row, self.pad:self.pad+self.n_col] = self.data_original.copy()
            self.data = pad_data.copy()
            
        self.fft_stack = []
        for slc in self.data:
            self.fft_stack.append(np.fft.fftshift(np.fft.fft2(slc)))
        self.fft_stack = np.asarray(self.fft_stack)

        self.modulus = np.abs(self.fft_stack)
        
    def filter_adjust(self):
        
        self.fig = plt.figure(figsize=(10, 5))
        G = gridspec.GridSpec(4, 8)
        self.ax1 = self.fig.add_subplot(G[:2, :2])
        self.ax2 = self.fig.add_subplot(G[:2, 2:4])
        self.ax3 = self.fig.add_subplot(G[:2, 4:6])
        self.ax4 = self.fig.add_subplot(G[2:, :2])
        self.ax5 = self.fig.add_subplot(G[2:, 2:4])
        self.ax6 = self.fig.add_subplot(G[2:, 4:6])
        self.ax7 = self.fig.add_subplot(G[:2, 6:8])
        self.ax8 = self.fig.add_subplot(G[2:, 6:8])
        
        st = {"description_width": "initial"}
        tilt_selector = pyw.IntSlider(min=1, max=len(self.modulus)-2, step=1, value=int(len(self.modulus)/2), description="tilt", style=st)
        x_pos = pyw.IntText(value=0, description="x position", style=st)
        y_pos = pyw.IntText(value=0, description="y position", style=st)
        width = pyw.IntText(value=self.data.shape[2], description="width", style=st)
        height = pyw.IntText(value=self.data.shape[1], description="height", style=st)
        low_cut = pyw.FloatText(value=int(self.data[0].shape[0]*0.001), description="low cut-off: ", style=st)
        high_cut = pyw.FloatText(value=int(self.data[0].shape[0]*0.5), description="high cut-off: ", style=st)
        stretching_x = pyw.Checkbox(value=False, description="x direction stretching")
        stretching_y = pyw.Checkbox(value=False, description="y direction stretching")
        hanning_window = pyw.Checkbox(value=False, description="activate hanning window")
        gauss_kernel = pyw.IntText(value=0, description="kernel size of gauss filter", style=st)
        gauss_sigma = pyw.FloatText(value=0.0, description="sigma of gauss filter", style=st)
        gauss_high = pyw.Checkbox(value=False, description="gauss high pass")
        laplacian_filter = pyw.IntText(value=0, description="kernel size of laplacian filter", style=st)
        sobel_filter = pyw.IntText(value=0, description="kernel size of sobel filter", style=st)
        scharr_filter = pyw.Checkbox(value=False, description="activate scharr filter")

        self.filter_widgets = pyw.interact(self.interact_widget, xp=x_pos, yp=y_pos, w=width, 
                                           h=height, snum=tilt_selector, lc=low_cut, hc=high_cut,
                                           stretch_x=stretching_x, stretch_y=stretching_y, hw=hanning_window,
                                           gas_k=gauss_kernel, gas_s=gauss_sigma,gas_h=gauss_high,
                                           lap_k=laplacian_filter, sob_k=sobel_filter, sch_a=scharr_filter)
        plt.show()
        
        
    def interact_widget(self, snum, xp, yp, w, h, lc, hc, 
                        stretch_x, stretch_y, hw, gas_k, gas_s, 
                        gas_h, lap_k, sob_k, sch_a):

        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        self.ax6.cla()
        self.ax7.cla()
        self.ax8.cla()

        self.ax1.imshow(self.data[snum-1], cmap="gray")
        self.ax2.imshow(self.data[snum], cmap="gray")
        self.ax3.imshow(self.data[snum+1], cmap="gray")

        bandpass = radial_indices(self.modulus.shape[1:], [lc, hc])

        filter_0 = np.abs(np.fft.ifft2(np.fft.fftshift(np.multiply(self.fft_stack[snum-1], bandpass))))
        filter_1 = np.abs(np.fft.ifft2(np.fft.fftshift(np.multiply(self.fft_stack[snum], bandpass))))
        filter_2 = np.abs(np.fft.ifft2(np.fft.fftshift(np.multiply(self.fft_stack[snum+1], bandpass))))

        filter_0 = filter_0[yp:(yp+h), xp:(xp+w)]
        filter_1 = filter_1[yp:(yp+h), xp:(xp+w)]
        filter_2 = filter_2[yp:(yp+h), xp:(xp+w)]

        if stretch_x:
            print("* x-direction stretch activated")
            filter_0 = cos_stretch(filter_0, self.angles[snum-1], orientation="x")
            filter_1 = cos_stretch(filter_1, self.angles[snum], orientation="x")        
            filter_2 = cos_stretch(filter_2, self.angles[snum+1], orientation="x")    

        if stretch_y:
            print("* y-direction stretch activated")
            filter_0 = cos_stretch(filter_0, self.angles[snum-1], orientation="y")
            filter_1 = cos_stretch(filter_1, self.angles[snum], orientation="y")        
            filter_2 = cos_stretch(filter_2, self.angles[snum+1], orientation="y")

        cr_shape = filter_1.shape

        if hw:
            print("* hanning window activated")
            han_win = hanning_2d(cr_shape)
            filter_0 = np.multiply(filter_0, han_win)
            filter_1 = np.multiply(filter_1, han_win)
            filter_2 = np.multiply(filter_2, han_win)


        if gas_k:
            print("* gaussian filter activated")
            if gas_h:
                print("** gaussian - high pass")
                filter_0 = filter_0 - cv2.GaussianBlur(filter_0, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                filter_1 = filter_1 - cv2.GaussianBlur(filter_1, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                filter_2 = filter_2 - cv2.GaussianBlur(filter_2, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)

            else:
                filter_0 = cv2.GaussianBlur(filter_0, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                filter_1 = cv2.GaussianBlur(filter_1, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                filter_2 = cv2.GaussianBlur(filter_2, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)

        if lap_k:
            print("* laplacian filter activated")
            filter_0 = cv2.Laplacian(filter_0, ddepth=-1, ksize=lap_k)
            filter_1 = cv2.Laplacian(filter_1, ddepth=-1, ksize=lap_k)
            filter_2 = cv2.Laplacian(filter_2, ddepth=-1, ksize=lap_k)


        cv_dt = cv2.CV_16U
        if sob_k:
            print("* sobel filter activated")
            filter_0 = sobel_xy(filter_0, sob_k, cv_dt)
            filter_1 = sobel_xy(filter_1, sob_k, cv_dt)
            filter_2 = sobel_xy(filter_2, sob_k, cv_dt)

        if sch_a:
            print("* scharr filter activated")
            filter_0 = scharr_xy(filter_0, cv_dt)
            filter_1 = scharr_xy(filter_1, cv_dt)
            filter_2 = scharr_xy(filter_2, cv_dt)


        self.ax4.imshow(filter_0, cmap="gray")
        self.ax5.imshow(filter_1, cmap="gray")
        self.ax6.imshow(filter_2, cmap="gray")

        uint_f0 = selective_normalize(filter_0.copy(), threshold=0, datatype="uint8")
        uint_f1 = selective_normalize(filter_1.copy(), threshold=0, datatype="uint8")
        uint_f2 = selective_normalize(filter_2.copy(), threshold=0, datatype="uint8")

        psh_1, _, _ = phase_cross_correlation(filter_1, filter_0, upsample_factor=100)
        psh_2, _, _ = phase_cross_correlation(filter_1, filter_2, upsample_factor=100)

        print("- left, phase cross correlation, shift")
        print(tuple(-psh_1))

        print("- right, phase cross correlation, shift")
        print(tuple(-psh_2))

        xcorr_l = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(filter_1) * np.fft.fft2(filter_0).conj()))
        xcorr_r = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(filter_1) * np.fft.fft2(filter_2).conj()))
        
        self.ax7.imshow(xcorr_l.real, cmap="gray")
        self.ax8.imshow(xcorr_r.real, cmap="gray")

        self.ax1.set_title("tilt series No.%d"%(snum))
        self.ax2.set_title("tilt series No.%d"%(snum+1))
        self.ax3.set_title("tilt series No.%d"%(snum+2))
        self.ax4.set_title("filtered")
        self.ax5.set_title("filtered")
        self.ax6.set_title("filtered")
        self.ax7.set_title("cross correlation (left)")
        self.ax8.set_title("cross correlation (right)")

        self.ax1.axis("off")
        self.ax2.axis("off")
        self.ax3.axis("off")
        self.ax4.axis("off")
        self.ax5.axis("off")
        self.ax6.axis("off")
        self.ax7.axis("off")
        self.ax8.axis("off")

        self.fig.canvas.draw()
        self.fig.tight_layout()
            

    def calculate_shift(self):
        
        widget_values = self.filter_widgets.widget.kwargs
        self.filter_widgets.widget.close_all()

        lp = widget_values["lc"]
        hp = widget_values["hc"]
        bpss = radial_indices(self.modulus.shape[1:], [lp, hp])

        yi = widget_values["yp"]
        xi = widget_values["xp"]
        hi = widget_values["h"]
        wi = widget_values["w"]
        print(yi, xi, hi, wi)
        
        stretch_x = widget_values["stretch_x"]
        stretch_y = widget_values["stretch_y"]
        print(stretch_x, stretch_y)
        
        hw = widget_values["hw"]
        print(hw)
        gas_k = widget_values["gas_k"]
        print(gas_k)
        gas_s = widget_values["gas_s"]
        print(gas_s)
        gas_h = widget_values["gas_h"]
        print(gas_h)
        lap_k = widget_values["lap_k"]
        print(lap_k)
        sob_k = widget_values["sob_k"]
        print(sob_k)
        sch_a = widget_values["sch_a"]
        print(sch_a)
        
        ref_num = int(self.num_img/2)
        psh_r = []
        psh_l = []

        for i in range(ref_num, self.num_img-1, 1):

            filter_1 = np.abs(np.fft.ifft2(np.fft.fftshift(np.multiply(self.fft_stack[i], bpss))))
            filter_2 = np.abs(np.fft.ifft2(np.fft.fftshift(np.multiply(self.fft_stack[i+1], bpss))))

            if stretch_x:
                filter_1 = cos_stretch(filter_1, self.angles[i], orientation="x")        
                filter_2 = cos_stretch(filter_2, self.angles[i+1], orientation="x")

            if stretch_y:
                filter_1 = cos_stretch(filter_1, self.angles[i], orientation="y")        
                filter_2 = cos_stretch(filter_2, self.angles[i+1], orientation="y")

            filter_1 = filter_1[yi:(yi+hi), xi:(xi+wi)]
            filter_2 = filter_2[yi:(yi+hi), xi:(xi+wi)]

            cr_shape = filter_1.shape

            if hw:
                han_win = hanning_2d(cr_shape)
                filter_1 = np.multiply(filter_1, han_win)
                filter_2 = np.multiply(filter_2, han_win)

            if gas_k:
                if gas_h:
                    filter_1 = filter_1 - cv2.GaussianBlur(filter_1, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                    filter_2 = filter_2 - cv2.GaussianBlur(filter_2, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)

                else:
                    filter_1 = cv2.GaussianBlur(filter_1, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                    filter_2 = cv2.GaussianBlur(filter_2, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)

            if lap_k:
                filter_1 = cv2.Laplacian(filter_1, ddepth=-1, ksize=lap_k)
                filter_2 = cv2.Laplacian(filter_2, ddepth=-1, ksize=lap_k)


            cv_dt = cv2.CV_16U
            if sob_k:
                filter_1 = sobel_xy(filter_1, sob_k, cv_dt)
                filter_2 = sobel_xy(filter_2, sob_k, cv_dt)

            if sch_a:
                filter_1 = scharr_xy(filter_1, cv_dt)
                filter_2 = scharr_xy(filter_2, cv_dt)

            psh_tmp, _, _ = phase_cross_correlation(filter_1, filter_2, upsample_factor=100)
            psh_r.append(psh_tmp)

        for i in range(ref_num, 0, -1):

            filter_1 = np.abs(np.fft.ifft2(np.fft.fftshift(np.multiply(self.fft_stack[i], bpss))))
            filter_2 = np.abs(np.fft.ifft2(np.fft.fftshift(np.multiply(self.fft_stack[i-1], bpss))))

            if stretch_x:
                filter_1 = cos_stretch(filter_1, self.angles[i], orientation="x")        
                filter_2 = cos_stretch(filter_2, self.angles[i+1], orientation="x")

            if stretch_y:
                filter_1 = cos_stretch(filter_1, self.angles[i], orientation="y")        
                filter_2 = cos_stretch(filter_2, self.angles[i+1], orientation="y")

            filter_1 = filter_1[yi:(yi+hi), xi:(xi+wi)]
            filter_2 = filter_2[yi:(yi+hi), xi:(xi+wi)]

            cr_shape = filter_1.shape

            if hw:
                han_win = hanning_2d(cr_shape)
                filter_1 = np.multiply(filter_1, han_win)
                filter_2 = np.multiply(filter_2, han_win)

            if gas_k:
                if gas_h:
                    filter_1 = filter_1 - cv2.GaussianBlur(filter_1, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                    filter_2 = filter_2 - cv2.GaussianBlur(filter_2, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)

                else:
                    filter_1 = cv2.GaussianBlur(filter_1, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)
                    filter_2 = cv2.GaussianBlur(filter_2, ksize=(gas_k, gas_k), sigmaX=gas_s, sigmaY=gas_s)

            if lap_k:
                filter_1 = cv2.Laplacian(filter_1, ddepth=-1, ksize=lap_k)
                filter_2 = cv2.Laplacian(filter_2, ddepth=-1, ksize=lap_k)


            cv_dt = cv2.CV_16U
            if sob_k:
                filter_1 = sobel_xy(filter_1, sob_k, cv_dt)
                filter_2 = sobel_xy(filter_2, sob_k, cv_dt)

            if sch_a:
                filter_1 = scharr_xy(filter_1, cv_dt)
                filter_2 = scharr_xy(filter_2, cv_dt)

            psh_tmp, _, _ = phase_cross_correlation(filter_1, filter_2, upsample_factor=100)
            psh_l.append(psh_tmp)
            
            
        psh_r_tmp = np.cumsum(psh_r, axis=0).tolist()
        psh_l_tmp = np.flip(np.cumsum(psh_l, axis=0), axis=0).tolist()
        psh = []
        psh.extend(psh_l_tmp)
        psh.append([0.0, 0.0])
        psh.extend(psh_r_tmp)
        self.psh = np.asarray(psh)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(self.psh[:, 0], self.psh[:, 1], c=np.arange(self.num_img), cmap="inferno")
        for i in range(self.num_img):
            ax.text(self.psh[i, 0], self.psh[i, 1], '%d'%(i+1))
        fig.suptitle("Calculated Shift")
        fig.tight_layout()
        plt.show()

    
    def apply_alignment(self):

        aligned_p = []
        for i, slc in enumerate(self.data):
            M = np.float32([[1, 0, self.psh[i][1]], [0, 1, self.psh[i][0]]])
            shifted = cv2.warpAffine(slc, M, slc.shape, borderMode=cv2.BORDER_REFLECT)
            aligned_p.append(shifted)
        self.aligned_p = np.asarray(aligned_p)

        if self.child_:
            for j in range(len(self.imgs)):
                data_tmp = self.imgs[j]
                if self.pad_:
                    pad_tmp = np.zeros((self.num_img, self.n_row+2*self.pad, self.n_col+2*self.pad))
                    pad_tmp[:, self.pad:self.pad+self.n_row, self.pad:self.pad+self.n_col] = data_tmp.copy()
                    data_tmp = pad_tmp
                aligned_tmp = []
                for i, slc in enumerate(data_tmp):
                    M = np.float32([[1, 0, self.psh[i][1]], [0, 1, self.psh[i][0]]])
                    shifted = cv2.warpAffine(slc, M, slc.shape, borderMode=cv2.BORDER_REFLECT)
                    aligned_tmp.append(shifted)
                aligned_tmp = np.asarray(aligned_tmp)
                self.imgs[j] = aligned_tmp


    def repeat(self):
        self.data = self.aligned_p
        self.prepare_alignment(pad=0)
        self.filter_adjust()
    
    def save_result(self):
        tifffile.imwrite("[alinged]ref_tilt_series.tif", self.aligned_p)
        if self.child_:
            self.merged = np.zeros((self.aligned_p.shape[0], self.aligned_p.shape[1], self.aligned_p.shape[2]*(1+len(self.imgs))))
            self.merged[:, :, :self.aligned_p.shape[2]] = self.aligned_p
            for i, im in enumerate(self.imgs):
                tifffile.imwrite("[alinged]child_tilt_series_%02d.tif"%(i+1), im)
                self.merged[:, :, (i+1)*self.aligned_p.shape[1]:(i+2)*self.aligned_p.shape[2]] = im
            tifffile.imwrite("[alinged]merged_all_tilt_series.tif", self.merged)


###########################################################################
# functions
###########################################################################
def radial_indices(shape, radial_range, center=None):
    y, x = np.indices(shape)
    if not center:
        center = np.array([(y.max()-y.min())/2.0, (x.max()-x.min())/2.0])
    
    r = np.hypot(y - center[0], x - center[1])
    ri = np.ones(r.shape)
    
    if len(np.unique(radial_range)) > 1:
        ri[np.where(r <= radial_range[0])] = 0
        ri[np.where(r > radial_range[1])] = 0
        
    else:
        r = np.round(r)
        ri[np.where(r != round(radial_range[0]))] = 0
    
    return ri

def cos_stretch(img, t_angle, orientation="y"):
    img_shape = img.shape
    if orientation=="y":
        new_shape = (img_shape[1], int(img_shape[0]*(1/np.cos(t_angle))))
        filtered = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
        tmp = int((filtered.shape[0] - img_shape[0])/2)
        return filtered[tmp:(tmp+img_shape[0]), :]
    elif orientation=="x":
        new_shape = (int(img_shape[1]*(1/np.cos(t_angle))), img_shape[0])
        filtered = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
        tmp = int((filtered.shape[1] - img_shape[1])/2)
        return filtered[:, tmp:(tmp+img_shape[1])]
    else:
        print("Warning! the orientation argument must be ""x"" or ""y""")
        
def sobel_xy(img, k, datatype):
    return cv2.Sobel(img, ddepth=datatype, dx=1, dy=0, ksize=k) \
        + cv2.Sobel(img, ddepth=datatype, dx=0, dy=1, ksize=k)

def scharr_xy(img, datatype):
    return cv2.Scharr(img, ddepth=datatype, dx=1, dy=0) \
        + cv2.Scharr(img, ddepth=datatype, dx=0, dy=1)

def hanning_2d(shape):
    hw_1 = np.hanning(shape[0]).reshape(-1, 1)
    hw_2 = np.hanning(shape[1]).reshape(1, -1)
    han_win = np.matmul(hw_1, hw_2).reshape(shape)
    return han_win / np.max(han_win)

def template_match(ref, tar, hr, wr, verbose=False):
    shape = ref.shape
    h = int(shape[0]*hr)
    w = int(shape[1]*wr)
    iy = int((shape[0] - h) / 2)
    ix = int((shape[1] - w) / 2)
    
    if verbose:
        print("- template location, (x, width, y, height)")
        print(ix, w, iy, h)
        
    tar_temp = tar[iy:(iy+h), ix:(ix+w)]

    return iy, ix, cv2.matchTemplate(ref, tar_temp, method=cv2.TM_CCOEFF)


def uint8astype(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * (2**8-1)
    return img.astype(np.uint8)

def uint16astype(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * (2**16-1)
    return img.astype(np.uint16)

def uint32astype(img):
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * (2**32-1)
    return img.astype(np.uint32)


def selective_normalize(data, threshold=0, datatype="uint16"):
    
    if threshold:    
        indices_part = np.where(data > np.percentile(data, threshold))
        part = data[indices_part]
        
    else:
        indices_part = np.nonzero(data)
        part = data[indices_part]
    
    if datatype=="uint16":
        part = uint16astype(part)
        normalized = np.zeros(data.shape, dtype=np.uint16)
        normalized[indices_part] = part
    elif datatype=="uint32":
        part = uint32astype(part)
        normalized = np.zeros(data.shape, dtype=np.uint32)
        normalized[indices_part] = part
    elif datatype=="uint8":
        part = uint8astype(part)
        normalized = np.zeros(data.shape, dtype=np.uint8)
        normalized[indices_part] = part
    else:
        print("wrong datatype!")
        return
    
    return normalized

def kill_low_int(data, threshold=50):
       
    indices_part = np.where(data > np.percentile(data, threshold))
    part = data[indices_part].copy()
    part = part - np.min(part)
    part = part / np.max(part)
    
    output = np.zeros(data.shape)
    output[indices_part] = part

    return output