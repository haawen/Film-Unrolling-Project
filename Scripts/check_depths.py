import nibabel as nib
import glob

files = sorted(glob.glob("/data/user/li_k1/M_thesis/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/imagesTr/*.nii.gz"))
depths = []
for f in files:
    shape = nib.load(f).shape
    depths.append(shape[2] if len(shape) >= 3 else shape[0])
    print(f"{f.split('/')[-1]:>50s}  shape={shape}")
print(f"\nMin depth: {min(depths)}, Max depth: {max(depths)}, All depths: {sorted(set(depths))}")
