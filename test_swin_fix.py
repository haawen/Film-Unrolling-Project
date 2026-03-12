"""
Local smoke test for train_monai.py transforms + model forward pass.
Tests both unet3d (patch 16,192,192) and swinunetr (patch 32,192,192)
with a synthetic volume mimicking the real data shape (20, 3063, 3062).
Uses a SMALL crop of (20, 256, 256) to keep memory manageable.
"""
import sys
import os
import tempfile
import numpy as np
import torch

# Ensure we can import from Scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def make_fake_nifti(tmpdir, n_cases=2, shape=(20, 256, 256)):
    """Create fake NIfTI files mimicking the dataset structure."""
    import nibabel as nib
    img_dir = os.path.join(tmpdir, "imagesTr")
    lbl_dir = os.path.join(tmpdir, "labelsTr")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cases = []
    for i in range(n_cases):
        name = f"test_{i:04d}"
        img_data = np.random.randn(*shape).astype(np.float32)
        lbl_data = np.random.randint(0, 3, size=shape).astype(np.uint8)

        img_path = os.path.join(img_dir, f"{name}_0000.nii.gz")
        lbl_path = os.path.join(lbl_dir, f"{name}.nii.gz")

        nib.save(nib.Nifti1Image(img_data, np.eye(4)), img_path)
        nib.save(nib.Nifti1Image(lbl_data, np.eye(4)), lbl_path)

        cases.append({"image": img_path, "label": lbl_path})
    return cases


def test_transforms(model_name, patch_size, cases):
    """Test that transforms run without error on the fake data."""
    from Scripts.train_monai import train_transforms, val_transforms

    print(f"\n--- Testing {model_name} transforms with patch_size={patch_size} ---")

    t_train = train_transforms(patch_size)
    t_val = val_transforms(patch_size)

    print("  Testing train transform...")
    result = t_train(cases[0])
    if isinstance(result, list):
        for i, r in enumerate(result):
            print(f"    Sample {i}: image={r['image'].shape}, label={r['label'].shape}")
    else:
        print(f"    image={result['image'].shape}, label={result['label'].shape}")

    print("  Testing val transform...")
    result = t_val(cases[1])
    print(f"    image={result['image'].shape}, label={result['label'].shape}")

    print(f"  [OK] {model_name} transforms passed.")
    return True


def test_model_forward(model_name, patch_size):
    """Test that the model can do a forward pass with the given patch size."""
    from Scripts.train_monai import build_model

    print(f"\n--- Testing {model_name} forward pass with patch_size={patch_size} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, patch_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    x = torch.randn(1, 1, *patch_size, device=device)
    with torch.no_grad():
        y = model(x)
    print(f"  Input: {x.shape} -> Output: {y.shape}")
    print(f"  [OK] {model_name} forward pass passed.")
    return True


def main():
    # Use small spatial dims to keep test fast (depth=20 matches real data)
    vol_shape = (20, 256, 256)
    tests = [
        ("unet3d", (16, 192, 192)),
        ("swinunetr", (32, 192, 192)),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Creating fake NIfTI data in {tmpdir} with shape {vol_shape}...")
        cases = make_fake_nifti(tmpdir, n_cases=2, shape=vol_shape)

        all_ok = True
        for model_name, patch_size in tests:
            try:
                test_transforms(model_name, patch_size, cases)
            except Exception as e:
                print(f"  [FAIL] {model_name} transforms: {e}")
                all_ok = False

            try:
                test_model_forward(model_name, patch_size)
            except Exception as e:
                print(f"  [FAIL] {model_name} forward: {e}")
                all_ok = False

    if all_ok:
        print("\n=== ALL TESTS PASSED ===")
    else:
        print("\n=== SOME TESTS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
