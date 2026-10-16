

# ===== appended: RealVideo dataset for SELF-SUPERVISED training on a real noisy video =====
# Returns (frames, frames) 5-frame 15-ch stacks from a folder of PNGs, NO synthetic noise
# added. The blind-spot loss trains on the noisy input directly, so both tensors = our
# real frames (the "clean" copy is only used for meaningless val-PSNR logging).
class RealVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, patch_size=None, stride=64, n_frames=5):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        assert len(self.files) > 0, f"no png frames in {data_path}"
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.bound = len(self.files)
        self.len = self.bound
        self.transform = transforms.Compose([transforms.ToTensor()])
        im0 = np.array(Image.open(self.files[0]).convert("RGB"))
        H, W = im0.shape[:2]
        if self.size is not None:
            self.n_H = int((H - self.size) / self.stride) + 1
            self.n_W = int((W - self.size) / self.stride) + 1
            self.n_patches = self.n_H * self.n_W
            self.len = self.bound * self.n_patches

    def __len__(self):
        return self.len

    def _load(self, i):
        i = min(max(i, 0), self.bound - 1)
        return np.array(Image.open(self.files[i]).convert("RGB"))

    def __getitem__(self, index):
        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches
        x = (self.n_frames - 1) // 2
        seq = [self._load(index + o) for o in range(-x, x + 1)]   # f(idx-2)..f(idx+2)
        Img = np.concatenate(seq, axis=2)                          # H x W x (3*nf)
        if self.size is not None:
            nh = (patch // self.n_W) * self.stride
            nw = (patch % self.n_W) * self.stride
            Img = Img[nh:nh + self.size, nw:nw + self.size, :]
        t = self.transform(np.ascontiguousarray(Img)).type(torch.FloatTensor)  # [3*nf,h,w] in [0,1]
        return t, t.clone()


@register_dataset("RealVideo")
def load_RealVideo(data, batch_size=8, dataset=None, video=None, image_size=None, stride=64,
                   n_frames=5, aug=0, dist="G", mode="S", noise_std=30, min_noise=0,
                   max_noise=100, sample=False, heldout=False):
    train_dataset = RealVideo(data, patch_size=image_size, stride=stride, n_frames=n_frames)
    test_dataset = RealVideo(data, patch_size=None, stride=stride, n_frames=n_frames)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    return train_loader, test_loader
