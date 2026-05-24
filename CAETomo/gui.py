import streamlit as st
import os
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile

import sys
sys.path.append('/home/ryuserve/github_repo/CAETomo/test')
from CAETomo.cae import cae
from CAETomo.alignment import tilt_series_alignment
from CAETomo.reconstruction import CSET

class StreamlitConsole:
    def __init__(self, container, step_key):
        self.container = container
        self.step_key = step_key
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        self.container.code(self.buffer, language="text")
        st.session_state.cae_outputs[self.step_key]['log'] = self.buffer

    def flush(self): pass

def run_step(step_key, log_container, fig_container, success_msg, func, *args, **kwargs):
    st.session_state.cae_outputs[step_key] = {'log': '', 'figs': []}
    original_show = plt.show
    plt.close('all') 
    
    def dynamic_show():
        fig = plt.gcf()
        fig_container.pyplot(fig)
        st.session_state.cae_outputs[step_key]['figs'].append(fig)
        plt.close(fig) 
        
    plt.show = dynamic_show
    
    try:
        with contextlib.redirect_stdout(StreamlitConsole(log_container, step_key)):
            res = func(*args, **kwargs)
            if res is not None and isinstance(res, plt.Figure):
                fig_container.pyplot(res)
                st.session_state.cae_outputs[step_key]['figs'].append(res)
                plt.close(res)
                
        fignums = plt.get_fignums()
        for n in fignums:
            fig = plt.figure(n)
            fig_container.pyplot(fig)
            st.session_state.cae_outputs[step_key]['figs'].append(fig)
            plt.close(fig)
            
        st.success(success_msg)
    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
    finally:
        plt.show = original_show

def render_step_results(step_key, log_container, fig_container, skip_figs=False):
    if st.session_state.cae_outputs[step_key]['log']:
        log_container.code(st.session_state.cae_outputs[step_key]['log'], language="text")
    if not skip_figs and st.session_state.cae_outputs[step_key]['figs']:
        with fig_container:
            for fig in st.session_state.cae_outputs[step_key]['figs']:
                st.pyplot(fig)

st.set_page_config(page_title="CAETomo Interface", layout="wide")
st.title("CAETomo GUI")

with st.sidebar:
    st.header("Hardware Configuration")
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    st.write(f"CUDA Devices Found: {device_count}")
    
    device_options = ["cpu"] + [f"cuda:{i}" for i in range(device_count)]
    selected_device = st.selectbox("Select Torch Device", device_options)
    torch_device = torch.device(selected_device)
    if cuda_available and selected_device != "cpu":
        torch.cuda.set_device(torch_device)

if 'cae_model' not in st.session_state: st.session_state.cae_model = None
if 'align_model' not in st.session_state: st.session_state.align_model = None
if 'recon_model' not in st.session_state: st.session_state.recon_model = None

if 'cae_outputs' not in st.session_state:
    st.session_state.cae_outputs = {
        key: {'log': '', 'figs': []} for key in [
            'cae_init', 'cae_bin', 'cae_center', 'cae_input', 'cae_train', 'cae_res',
            'al_init', 'al_prep', 'al_prev', 'al_calc', 'al_apply',
            're_init', 're_nufft', 're_run'
        ]
    }

tab_cae, tab_align, tab_recon = st.tabs([
    "1. 1D-CAE", 
    "2. Tilt Axis Alignment", 
    "3. CS-ET Reconstruction"
])

with tab_cae:
    st.header("Step 1: Load Data")
    file_paths_input = st.text_area("Enter absolute file paths (one path per line):", height=60, help="Paste the full system paths to your .dm3, .dm4, or .tif files here.", key="cae_paths")
    
    c1, c2 = st.columns(2)
    with c1:
        dat_dim = st.number_input("Data Dimension", value=3, min_value=2, max_value=4, help="Dimensions of hyperspectral data (e.g., 3 for EELS, 4 for 4D-STEM).", key="cae_dim")
        dat_unit = st.text_input("Data Unit", value="eV", help="Unit of the spectral axis.", key="cae_unit")
        cr_range_input = st.text_input("Crop Range (start, end, step)", value="1.0, 3.56, 0.01", help="Discard irrelevant parts of the spectrum. Float for DM files, integer for index range.", key="cae_cr")
        dat_scale = st.number_input("Data Scale", value=1.0, help="Specifies the actual step size if integer crop ranges are used.", key="cae_scale")
    with c2:
        rescale = st.checkbox("Rescale Data", value=False, help="If True, each data array will be divided by its maximum value during loading.", key="cae_rescale")
        dm_file = st.checkbox("DM File Format", value=True, help="Check this if you are using DM3/DM4 files requiring hyperspy.", key="cae_dm")
        verbose = st.checkbox("Verbose Output", value=True, help="Print progress and shapes to the console.", key="cae_verb")

    log_c_init = st.empty(); fig_c_init = st.container()
    is_run_init = False
    if st.button("1. Load Data", key="btn_c_init"):
        paths = [p.strip() for p in file_paths_input.split('\n') if p.strip()]
        if not paths: st.error("Provide valid paths.")
        else:
            def init_cae():
                cr = [float(x.strip()) for x in cr_range_input.split(",")] if cr_range_input else None
                st.session_state.cae_model = cae(paths, dat_dim, dat_unit, cr, dat_scale, rescale, dm_file)
            run_step('cae_init', log_c_init, fig_c_init, "Model Initialized.", init_cae)
            is_run_init = True
    render_step_results('cae_init', log_c_init, fig_c_init, skip_figs=is_run_init)

    st.write("---")
    st.header("Step 2: Binning (Optional)")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        bin_y = st.number_input("Bin Y", value=1, min_value=1, help="Binning size in the height direction.", key="cb_y")
        bin_x = st.number_input("Bin X", value=1, min_value=1, help="Binning size in the width direction.", key="cb_x")
    with b_col2:
        str_y = st.number_input("Stride Y", value=1, min_value=1, help="Stride in the height direction.", key="cs_y")
        str_x = st.number_input("Stride X", value=1, min_value=1, help="Stride in the width direction.", key="cs_x")
    with b_col3:
        offset = st.number_input("Offset", value=0, help="Offset for the spectral depth dimension.", key="c_off")
        rescale_0to1_bin = st.checkbox("Rescale 0 to 1 (Binning)", value=True, help="Rescale each binned data pixel from 0 to 1.", key="c_resb")

    log_c_bin = st.empty(); fig_c_bin = st.container()
    is_run_bin = False
    if st.button("2. Execute Binning", key="btn_c_bin"):
        if st.session_state.cae_model: 
            run_step('cae_bin', log_c_bin, fig_c_bin, "Binning complete.", st.session_state.cae_model.binning, bin_y, bin_x, str_y, str_x, offset=offset, rescale_0to1=rescale_0to1_bin)
            is_run_bin = True
        else: st.warning("Initialize Model first.")
    render_step_results('cae_bin', log_c_bin, fig_c_bin, skip_figs=is_run_bin)

    st.write("---")
    st.header("Step 3: Find Center (4D-STEM Only)")
    if dat_dim == 4:
        fc_col1, fc_col2 = st.columns(2)
        with fc_col1:
            cbox_edge = st.number_input("Center Box Edge", value=7, help="The edge length of the center box for finding the center position.", key="c_cbox")
            center_remove = st.number_input("Center Remove Range", value=0, help="If greater than zero, the center box specified by this radius will be removed.", key="c_crem")
        with fc_col2:
            fc_result_visual = st.checkbox("Show Center Visuals", value=True, help="Display the computed centers over the diffraction patterns.", key="c_cvis")
            fc_log_scale = st.checkbox("Log Scale Center Maps", value=True, help="Converts the intensities into log-scale for visualization.", key="c_clog")
            
        log_c_cen = st.empty(); fig_c_cen = st.container()
        is_run_cen = False
        if st.button("3. Find Center", key="btn_c_cen"):
            if st.session_state.cae_model: 
                run_step('cae_center', log_c_cen, fig_c_cen, "Centers located.", st.session_state.cae_model.load_data.find_center, cbox_edge=cbox_edge, center_remove=center_remove, result_visual=fc_result_visual, log_scale=fc_log_scale)
                is_run_cen = True
            else: st.warning("Initialize Model first.")
        render_step_results('cae_center', log_c_cen, fig_c_cen, skip_figs=is_run_cen)
    else:
        st.info("Find Center options are hidden. Change Data Dimension to 4 in Step 1 to access these controls.")

    st.write("---")
    st.header("Step 4: Prepare Input Dataset & Batches")
    mi_col1, mi_col2, mi_col3 = st.columns(3)
    with mi_col1:
        min_val = st.number_input("Minimum Value", value=1e-6, format="%e", help="Lower clipping bound for the data matrix to ensure non-negativity.", key="c_mval")
        w_size = st.number_input("Window Size", value=0, help="Crop distance for box flattening.", key="c_wsize")
        radial_range_input = st.text_input("Radial Range (start, end, step)", value="", help="Provide comma-separated values for radial flattening.", key="c_radr")
    with mi_col2:
        max_normalize = st.checkbox("Max Normalize", value=True, help="Divide the matrix by its maximum values.", key="c_mnorm")
        rescale_0to1_input = st.checkbox("Rescale 0 to 1", value=False, help="Rescale final flattened inputs from 0 to 1.", key="c_resi")
        mi_log_scale = st.checkbox("Log Scale", value=False, help="Apply logarithmic scaling to the input matrix.", key="c_mlog")
    with mi_col3:
        radial_flat = st.checkbox("Radial Flat", value=True, help="Use radial flattening instead of box flattening (for 4D-STEM).", key="c_rflat")
        final_dim = st.number_input("Final Dimension", value=1, min_value=1, max_value=2, help="Determines the shape of the flattened dataset (1 or 2).", key="c_fdim")
        batch_size = st.number_input("Mini-Batch Size", value=128, min_value=1, help="Number of samples per training batch.")

    log_c_in = st.empty(); fig_c_in = st.container()
    is_run_in = False
    if st.button("4. Prepare Input Matrix", key="btn_c_in"):
        if st.session_state.cae_model:
            def prep_cae():
                rr = [int(x.strip()) for x in radial_range_input.split(",")] if radial_range_input else None
                st.session_state.cae_model.make_input(min_val=min_val, max_normalize=max_normalize, rescale_0to1=rescale_0to1_input, log_scale=mi_log_scale, radial_flat=radial_flat, w_size=w_size, radial_range=rr, final_dim=final_dim)
                st.session_state.cae_model.create_mini_batch(batch_size=batch_size)
            run_step('cae_input', log_c_in, fig_c_in, "Dataset flattened and batches prepared.", prep_cae)
            is_run_in = True
        else: st.warning("Initialize Model first.")
    render_step_results('cae_input', log_c_in, fig_c_in, skip_figs=is_run_in)

    st.write("---")
    st.header("Step 5: Construct & Train 1D-CAE")
    c6, c7 = st.columns(2)
    with c6:
        num_comp = st.number_input("Number of Components", value=5, min_value=1, help="Target dimensions / number of components to extract.")
        channels_str = st.text_input("Channels", value="8, 16, 32, 5", help="Comma-separated list of output channels for each 1D-CNN layer.")
        kernels_str = st.text_input("Kernels", value="64, 32, 16, 7", help="Comma-separated list of kernel sizes for each 1D-CNN layer.")
        pooling_str = st.text_input("Poolings", value="2, 2, 2, 2", help="Comma-separated list of average pooling strides.")
    with c7:
        opt = st.selectbox("Optimizer", ["ADAM", "SGD"], help="Optimizer for gradient descent.")
        loss = st.selectbox("Loss Function", ["MSE", "BCE"], help="Loss function (Mean Squared Error or Binary Cross Entropy).")
        lr = st.number_input("Learning Rate", value=0.001, format="%f", help="Learning rate for the optimizer.")
        epochs = st.number_input("Epochs", value=100, min_value=1, help="Total number of training passes over the dataset.")

    log_c_tr = st.empty(); fig_c_tr = st.container()
    is_run_tr = False
    if st.button("5. Train Autoencoder"):
        if st.session_state.cae_model:
            def train_cae():
                ch = [int(x.strip()) for x in channels_str.split(",")]
                kn = [int(x.strip()) for x in kernels_str.split(",")]
                pl = [int(x.strip()) for x in pooling_str.split(",")]
                st.session_state.cae_model.create_autoencoder(num_comp=num_comp, channels=ch, kernels=kn, pooling=pl, cuda_device=torch_device)
                st.session_state.cae_model.training(optimizer=opt, loss_fn=loss, l_rate=lr, n_epoch=epochs)
            run_step('cae_train', log_c_tr, fig_c_tr, "Training complete.", train_cae)
            is_run_tr = True
        else: st.warning("Prepare batches first.")
    render_step_results('cae_train', log_c_tr, fig_c_tr, skip_figs=is_run_tr)

    st.write("---")
    st.header("Step 6: Display & Export Results")
    
    cae_save_col1, cae_save_col2 = st.columns(2)
    with cae_save_col1:
        cae_save_dir = st.text_input("Save Directory", value="./caetomo_output", help="Directory path where the CAE feature maps and components will be saved.", key="cae_dir")
    with cae_save_col2:
        cae_save_prefix = st.text_input("File Prefix", value="cae_out_", help="Prefix added to the exported file names.", key="cae_pref")
        
    log_c_res = st.empty(); fig_c_res = st.container()
    is_run_res = False
    if st.button("6. Display & Export"):
        if st.session_state.cae_model:
            os.makedirs(cae_save_dir, exist_ok=True)
            full_cae_prefix = os.path.join(cae_save_dir, cae_save_prefix)
            run_step('cae_res', log_c_res, fig_c_res, "Exported successfully.", st.session_state.cae_model.show_result, save_result=True, save_prefix=full_cae_prefix)
            is_run_res = True
        else: st.warning("Train model first.")
    render_step_results('cae_res', log_c_res, fig_c_res, skip_figs=is_run_res)


with tab_align:
    st.header("Tilt Series Alignment Pipeline")
    ref_file = st.text_input("Reference Tilt Series Path", help="Path to reference ADF-STEM tilt series for calculating alignment shifts.")
    child_files = st.text_area("Child Tilt Series Paths (optional)", height=60, help="Paths to feature map tilt series to be aligned using the exact same reference shifts.")
    ang_str = st.text_input("Angles Range (start, end, steps)", value="-70, 40, 12", help="Mapping of tilt angles in degrees (start, end, steps).")

    log_al_init = st.empty(); fig_al_init = st.container()
    is_al_init = False
    if st.button("Initialize Alignment Tool"):
        if not ref_file: st.error("Reference path required.")
        else:
            def init_align():
                ang_params = [float(x.strip()) for x in ang_str.split(",")]
                angles_deg = np.linspace(ang_params[0], ang_params[1], int(ang_params[2]))
                angles_rad = angles_deg * np.pi / 180
                children = [p.strip() for p in child_files.split('\n') if p.strip()]
                st.session_state.align_model = tilt_series_alignment(ref_file, angles_rad, img_adr=children if children else None)
            run_step('al_init', log_al_init, fig_al_init, "Alignment tool initialized.", init_align)
            is_al_init = True
    render_step_results('al_init', log_al_init, fig_al_init, skip_figs=is_al_init)

    st.write("---")
    pad_val = st.number_input("Padding Size", value=20, min_value=0, help="Amount of zero-padding to add around the image to prevent edge artifacts during affine shifting.")
    log_al_prep = st.empty(); fig_al_prep = st.container()
    is_al_prep = False
    if st.button("Prepare FFTs & Padding"):
        if st.session_state.align_model:
            run_step('al_prep', log_al_prep, fig_al_prep, "Transforms calculated.", st.session_state.align_model.prepare_alignment, pad=pad_val)
            is_al_prep = True
        else: st.warning("Initialize tool first.")
    render_step_results('al_prep', log_al_prep, fig_al_prep, skip_figs=is_al_prep)

    st.write("---")
    st.subheader("Filter Adjustments & Live Preview")

    # Determine dynamic ranges/values if alignment model is initialized
    if st.session_state.align_model is not None:
        num_img = st.session_state.align_model.num_img
        snum_max = max(1, num_img - 2)
        snum_default = int(num_img / 2)
        height, width = st.session_state.align_model.data_original.shape[1:3]
        max_cutoff = max(width, height)
    else:
        num_img = 10
        snum_max = 8
        snum_default = 5
        height, width = 512, 512
        max_cutoff = 512

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        snum = st.slider("Tilt Index", min_value=1, max_value=snum_max, value=snum_default, help="Index of the tilt image to use for previewing the filter effect.")
        xp = st.slider("X Position", min_value=0, max_value=width-1, value=0, help="Crop window X start coordinate for cross-correlation.")
        yp = st.slider("Y Position", min_value=0, max_value=height-1, value=0, help="Crop window Y start coordinate for cross-correlation.")
        w = st.slider("Width", min_value=1, max_value=width, value=width, help="Crop window width.")
        h = st.slider("Height", min_value=1, max_value=height, value=height, help="Crop window height.")
    with f2:
        lc = st.slider("Low Cut-off", min_value=0, max_value=max_cutoff, value=0, help="Low cut-off frequency for the radial bandpass filter.")
        hc = st.slider("High Cut-off", min_value=0, max_value=max_cutoff, value=0, help="High cut-off frequency for the radial bandpass filter. (0 to bypass)")
        stretch_x = st.checkbox("X Stretch", value=False, help="Apply cosine stretching to compensate for the tilt angle projection (X-axis).")
        stretch_y = st.checkbox("Y Stretch", value=False, help="Apply cosine stretching to compensate for the tilt angle projection (Y-axis).")
    with f3:
        hw = st.checkbox("Hanning Window", value=False, help="Apply a Hanning window to reduce edge ringing in FFT phase correlation.")
        gas_k = st.number_input("Gauss Kernel", value=0, help="Gaussian blur kernel size to smooth high-frequency noise (0 disables).")
        gas_s = st.number_input("Gauss Sigma", value=0.0, help="Gaussian blur sigma parameter.")
        gas_h = st.checkbox("Gauss High Pass", value=False, help="Subtract the blurred image to create a high-pass Gaussian filter.")
    with f4:
        lap_k = st.number_input("Laplacian Kernel", value=0, help="Laplacian edge detection filter kernel size.")
        sob_k = st.number_input("Sobel Kernel", value=0, help="Sobel edge detection filter kernel size.")
        sch_a = st.checkbox("Scharr Filter", value=False, help="Apply Scharr edge detection filter to emphasize structural boundaries.")

    log_al_prev = st.empty()
    fig_al_prev = st.container()
    if st.session_state.align_model and hasattr(st.session_state.align_model, 'fft_stack'):
        with fig_al_prev:
            try:
                fig = st.session_state.align_model.preview_streamlit_filter(
                    int(snum), int(xp), int(yp), int(w), int(h), lc, hc,
                    stretch_x, stretch_y, hw, int(gas_k), gas_s, gas_h, int(lap_k), int(sob_k), sch_a
                )
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Error rendering filter preview: {e}")
    else:
        st.info("Prepare alignment first to see live filter preview.")

    st.write("---")
    
    col_calc, col_apply = st.columns(2)
    
    with col_calc:
        st.subheader("Calculate Shifts")
        log_al_calc = st.empty(); fig_al_calc = st.container()
        is_al_calc = False
        if st.button("Compute Cross-Correlations"):
            if st.session_state.align_model:
                run_step('al_calc', log_al_calc, fig_al_calc, "Shifts computed.", st.session_state.align_model.apply_streamlit_shift, lc, hc, int(yp), int(xp), int(h), int(w), stretch_x, stretch_y, hw, int(gas_k), gas_s, gas_h, int(lap_k), int(sob_k), sch_a)
                is_al_calc = True
            else: st.warning("Prepare alignment first.")
        render_step_results('al_calc', log_al_calc, fig_al_calc, skip_figs=is_al_calc)

    with col_apply:
        st.subheader("Apply Shifts & Interactive Preview")
        log_al_app = st.empty(); fig_al_app = st.container()
        is_al_app = False
        if st.button("Apply Alignment Shifts"):
            if st.session_state.align_model:
                run_step('al_apply', log_al_app, fig_al_app, "Transforms applied.", st.session_state.align_model.apply_alignment)
                is_al_app = True
            else: st.warning("Compute shifts first.")
        render_step_results('al_apply', log_al_app, fig_al_app, skip_figs=is_al_app)
        
        if st.session_state.align_model and hasattr(st.session_state.align_model, 'aligned_p'):
            st.write("---")
            max_slices = st.session_state.align_model.aligned_p.shape[0]
            slice_idx = st.slider("Select Slice (Tilt Index) to View", min_value=1, max_value=max_slices, value=1, help="Scroll through the image stack to check alignment quality.")
            threshold = st.slider("Percentile Threshold", min_value=0, max_value=100, value=10, help="Filters out low intensity noise for clearer viewing.")
                
            fig, ax = plt.subplots(figsize=(5, 5))
            img_slice = st.session_state.align_model.aligned_p[slice_idx - 1].copy()
            
            indices_part = np.where(img_slice > np.percentile(img_slice, threshold))
            part = img_slice[indices_part].copy()
            if len(part) > 0:
                part = part - np.min(part)
                part = part / np.max(part)
            output = np.zeros(img_slice.shape)
            output[indices_part] = part
            
            ax.imshow(output, cmap="inferno")
            ax.axis("off")
            ax.set_title(f"Aligned Slice No.{slice_idx}, Threshold Percentile: {threshold}")
            st.pyplot(fig)
            plt.close(fig)

            st.write("---")
            st.subheader("Export Aligned Tilt Series")
            save_dir = st.text_input("Save Directory", value="./results", help="Directory path where the aligned TIFF files will be saved.", key="al_dir")
            save_prefix = st.text_input("File Prefix", value="aligned_", help="Prefix for the saved file names.", key="al_pref")

            if st.button("Save TIFF Files", key="al_save"):
                try:
                    os.makedirs(save_dir, exist_ok=True)
                    ref_out = os.path.join(save_dir, f"{save_prefix}ref_tilt_series.tif")
                    tifffile.imwrite(ref_out, st.session_state.align_model.aligned_p)
                    
                    if st.session_state.align_model.child_:
                        ap_shape = st.session_state.align_model.aligned_p.shape
                        merged = np.zeros((ap_shape[0], ap_shape[1], ap_shape[2] * (1 + len(st.session_state.align_model.imgs))))
                        merged[:, :, :ap_shape[2]] = st.session_state.align_model.aligned_p
                        
                        for i, im in enumerate(st.session_state.align_model.imgs):
                            child_out = os.path.join(save_dir, f"{save_prefix}child_tilt_series_{i+1:02d}.tif")
                            tifffile.imwrite(child_out, im.astype(np.float32))
                            merged[:, :, (i+1)*ap_shape[2]:(i+2)*ap_shape[2]] = im
                            
                        merged_out = os.path.join(save_dir, f"{save_prefix}merged_all_tilt_series.tif")
                        tifffile.imwrite(merged_out, merged.astype(np.float32))
                    st.success(f"Alignment files successfully written to: {os.path.abspath(save_dir)}")
                except Exception as e:
                    st.error(f"Failed to save TIFFs: {str(e)}")


with tab_recon:
    st.header("Compressed Sensing Electron Tomography (CS-ET)")
    recon_paths = st.text_area("Aligned Tilt Series Paths (absolute)", height=60, help="Paths to the aligned tilt series TIFF files for tomographic reconstruction.")
    recon_angs = st.text_input("Angles Range (start, end, steps)", value="-70, 40, 12", help="Mapping of tilt angles in degrees (start, end, steps).", key="recon_angs")

    log_re_init = st.empty(); fig_re_init = st.container()
    is_re_init = False
    if st.button("Initialize CS-ET Model"):
        r_paths = [p.strip() for p in recon_paths.split('\n') if p.strip()]
        if not r_paths: st.error("Paths required.")
        else:
            def init_recon():
                ap = [float(x.strip()) for x in recon_angs.split(",")]
                ta = np.linspace(ap[0], ap[1], int(ap[2])) * np.pi / 180
                st.session_state.recon_model = CSET(r_paths, ta, cuda_device=torch_device)
            run_step('re_init', log_re_init, fig_re_init, "CS-ET model ready.", init_recon)
            is_re_init = True
    render_step_results('re_init', log_re_init, fig_re_init, skip_figs=is_re_init)

    st.write("---")
    r1, r2 = st.columns(2)
    with r1: pad_ratio = st.number_input("Padding Ratio", value=2.0, min_value=1.0, help="Oversampling ratio for the Non-Uniform FFT (NUFFT) grid. Must be > 1.0.")
    with r2: t_axis = st.selectbox("Tilt Axis Direction", ["horizontal", "vertical"], help="Axis of rotation for the tilt series.")

    log_re_nu = st.empty(); fig_re_nu = st.container()
    is_re_nu = False
    if st.button("Prepare NUFFT Operators"):
        if st.session_state.recon_model:
            run_step('re_nufft', log_re_nu, fig_re_nu, "Non-uniform FFT initialized.", st.session_state.recon_model.NUFFT, pad_ratio=pad_ratio, tilt_axis=t_axis)
            is_re_nu = True
        else: st.warning("Initialize model first.")
    render_step_results('re_nufft', log_re_nu, fig_re_nu, skip_figs=is_re_nu)

    st.write("---")
    st.subheader("Training Parameters")
    r3, r4 = st.columns(2)
    with r3: 
        n_iters = st.number_input("Iterations", value=100, min_value=1, help="Number of SGD iterations for the CS-ET reconstruction.")
        cs_lr = st.number_input("Optimizer LR", value=2e-5, format="%e", help="Learning rate for the reconstruction optimizer.")
    with r4: 
        lam_l1 = st.number_input("Lambda L1", value=5e-4, format="%e", help="Weight for the L1 sparsity penalty (Compressed Sensing).")
        lam_tv = st.number_input("Lambda Total Variation", value=3.0, format="%f", help="Weight for the Total Variation (TV) penalty to denoise the volume.")
        save_recon = st.checkbox("Export Tiffs to Disk", value=True, help="Automatically save the 3D reconstructed slices to disk.")
    
    st.write("---")
    st.subheader("Export Configuration")
    re_save_col1, re_save_col2 = st.columns(2)
    with re_save_col1:
        recon_dir = st.text_input("Save Directory", value="./reconstructed", help="Directory path where the reconstructions will be saved.", key="re_dir")
    with re_save_col2:
        recon_prefix = st.text_input("Output Prefix", value="rec_", help="Prefix added to the exported file names.", key="re_pref")

    log_re_run = st.empty(); fig_re_run = st.container()
    is_re_run = False
    if st.button("Execute Volume Reconstruction"):
        if st.session_state.recon_model:
            os.makedirs(recon_dir, exist_ok=True)
            full_recon_prefix = os.path.join(recon_dir, recon_prefix)
            run_step('re_run', log_re_run, fig_re_run, "Tomographic reconstruction complete.", st.session_state.recon_model.recontruct, n_iter=n_iters, lmbd_l1=lam_l1, lmbd_tv=lam_tv, l_rate=cs_lr, save_result=save_recon, save_adr=full_recon_prefix, verbose=True)
            is_re_run = True
        else: st.warning("Prepare NUFFT first.")
    render_step_results('re_run', log_re_run, fig_re_run, skip_figs=is_re_run)