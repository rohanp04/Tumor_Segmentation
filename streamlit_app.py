# streamlit_app.py

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st
from monai.data import Dataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, EnsureTyped, Orientationd, Spacingd,
    CropForegroundd, Resized, Compose, NormalizeIntensityd
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# -------------------------
# CONFIG
# -------------------------
BASE_DIR = "sample data"   # <-- change this to your dataset path
MODEL_PATH = "tumor_segmentation_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPATIAL_SIZE = (128, 128, 128)
CLASS_NAMES = ["Background", "Necrotic", "Edema", "Enhancing"]

# -------------------------
# Helper: find modalities
# -------------------------
def find_case_files(case_dir: str):
    files = sorted(glob.glob(os.path.join(case_dir, "*.nii*")))
    if len(files) < 5:
        return None
    name_map = {os.path.basename(p).lower(): p for p in files}

    def pick_like(key_options):
        for k in key_options:
            for name, path in name_map.items():
                if k in name:
                    return path
        return None

    flair = pick_like(["flair"])
    t1 = pick_like(["t1.nii", "_t1."])
    t1ce = pick_like(["t1ce", "t1gd"])
    t2 = pick_like(["t2.nii", "_t2."])
    seg = pick_like(["seg", "mask"])

    if None in [flair, t1, t1ce, t2, seg]:
        return None
    return {"image": [flair, t1, t1ce, t2], "label": seg, "name": os.path.basename(case_dir)}

# -------------------------
# Dice calculation
# -------------------------
def dice_coef(pred, gt, cls):
    pred_mask, gt_mask = (pred == cls), (gt == cls)
    inter, union = np.sum(pred_mask & gt_mask), (np.sum(pred_mask) + np.sum(gt_mask))
    return (2 * inter / union) if union > 0 else 1.0

# -------------------------
# Visualization
# -------------------------
def show_slice(img, gt, pred, idx):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img[1, :, :, idx], cmap="gray")
    axs[0].set_title("T1CE")
    axs[1].imshow(gt[:, :, idx], cmap="jet", vmin=0, vmax=3)
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred[:, :, idx], cmap="jet", vmin=0, vmax=3)
    axs[2].set_title("Prediction")
    st.pyplot(fig)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🧠 Brain Tumor Segmentation - MONAI")

# Model description
st.markdown(
    """
    **Model Description:**  
    This is a brain tumor segmentation model designed to analyze MRI scans of patients.  
    The model uses a **Graph Convolutional Network (GCN)** approach combined with a **base 3D U-Net architecture**.  
    It segments key tumor regions, including necrotic tissue, edema, and enhancing tumor areas.  
    The outputs include both visual segmentation maps and Dice coefficient scores for evaluation.
    """
)

st.sidebar.header("⚙️ Inference Options")

# List cases
case_dirs = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
cases = [os.path.basename(c) for c in case_dirs]
selected_case = st.sidebar.selectbox("Choose a case:", cases)

# Run inference button
if st.sidebar.button("Run Inference"):
    case_path = [c for c in case_dirs if os.path.basename(c) == selected_case][0]
    test_case = find_case_files(case_path)

    if test_case is None:
        st.error(
            "❌ Case files not found or incomplete. Make sure your folder contains:\n"
            "- FLAIR\n- T1\n- T1CE\n- T2\n- Segmentation mask"
        )
        st.stop()  # stop execution if data is incomplete

    for k, v in test_case.items():
        if isinstance(v, list):
            test_case[k] = [os.path.normpath(p) for p in v]
        else:
            test_case[k] = os.path.normpath(v)

    # Debug: show files found
    st.write("✅ Files detected for this case:")
    for k, v in test_case.items():
        st.write(f"{k}: {v}")

    # Transforms
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=SPATIAL_SIZE, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"])
    ])

    dataset = Dataset(data=[test_case], transform=transforms)
    sample = dataset[0]
    input_tensor = sample["image"].unsqueeze(0).to(DEVICE)
    ground_truth = sample["label"].cpu().numpy().squeeze()

    # Load model
    model = UNet(
        spatial_dims=3, in_channels=4, out_channels=4,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    st.info(f"✅ Loaded model (Val Dice: {checkpoint['val_dice']:.4f})")

    # Inference
    with torch.no_grad():
        logits = sliding_window_inference(input_tensor, SPATIAL_SIZE, 1, model, overlap=0.25)
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).cpu().numpy().squeeze()

    # Dice Scores
    st.subheader("📊 Dice Scores")
    for i, name in enumerate(CLASS_NAMES):
        st.write(f"**{name}:** {dice_coef(prediction, ground_truth, i):.4f}")

    # Visualization
    st.subheader("🖼️ Visualization")
    mid_slice = prediction.shape[-1] // 2
    show_slice(sample["image"].cpu().numpy(), ground_truth, prediction, mid_slice)
