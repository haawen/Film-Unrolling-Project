"""Denoise our REAL (already-noisy) stabilized video with pretrained FastDVDnet.
Unlike test_fastdvdnet.py this does NOT add synthetic noise — it feeds the real
frames straight in and denoises at a chosen sigma (the conditioning noise level).
Grayscale video is fed as 3-ch (pretrained model is RGB). Sweeps a few sigmas."""
import os, argparse, cv2, numpy as np, torch, torch.nn as nn
from collections import OrderedDict
from models import FastDVDnet
from fastdvdnet import denoise_seq_fastdvdnet

def remove_dataparallel_wrapper(sd):   # inlined to avoid utils.py's heavy imports
    return OrderedDict((k.replace('module.', ''), v) for k, v in sd.items())

def load_model(mf, device):
    m = FastDVDnet(num_input_frames=5)
    sd = torch.load(mf, map_location=device)
    if device.type == 'cuda':
        m = nn.DataParallel(m, device_ids=[0]).cuda(); m.load_state_dict(sd)
    else:
        m.load_state_dict(remove_dataparallel_wrapper(sd))
    m.eval(); return m

def read_video(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 15; fr = []
    while True:
        r, f = cap.read()
        if not r: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr).astype(np.float32)/255.0, fps

def write_video(path, arr, fps):
    T, H, W, _ = arr.shape
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in arr:
        vw.write(cv2.cvtColor((np.clip(f,0,1)*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    vw.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--sigmas', default='15,25,35')
    ap.add_argument('--model', default='./model.pth')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', dev, flush=True)
    model = load_model(a.model, dev)
    vid, fps = read_video(a.input)                 # [T,H,W,3]
    T, H, W, _ = vid.shape
    print(f'loaded {T} frames {H}x{W} fps~{fps:.1f}', flush=True)
    ph, pw = (16-H % 16) % 16, (16-W % 16) % 16
    seq = np.pad(vid, ((0,0),(0,ph),(0,pw),(0,0)), mode='reflect')
    seq = torch.from_numpy(seq.transpose(0,3,1,2)).contiguous().to(dev)   # [T,3,H2,W2]
    os.makedirs(a.outdir, exist_ok=True)
    for s in [float(x) for x in a.sigmas.split(',')]:
        with torch.no_grad():
            nstd = torch.FloatTensor([s/255.]).to(dev)
            den = denoise_seq_fastdvdnet(seq=seq, noise_std=nstd, temp_psz=5, model_temporal=model)
        out = den.cpu().numpy().transpose(0,2,3,1)[:, :H, :W, :]
        write_video(os.path.join(a.outdir, f'fastdvdnet_s{int(s)}.mp4'), out, 15)
        print('wrote sigma', s, flush=True)
    print('done')

if __name__ == '__main__':
    main()
