"""Zero-shot UDVD (pretrained blind_video_net) on our REAL noisy video.
Mirrors notebook_demos/denoising_demo.ipynb: build a 5-frame 15-ch input per
center frame, run the blind-spot model, MMSE post_process with the center frame
at a chosen sigma. No added synthetic noise. Sweeps sigma."""
import os, argparse, numpy as np, cv2, torch
# torch 2.6 defaults weights_only=True which rejects the checkpoint's argparse.Namespace;
# these are trusted official UDVD weights -> force weights_only=False.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})
import utils

def read_video(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 15; fr = []
    while True:
        r, f = cap.read()
        if not r: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr).astype(np.float32)/255., fps

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
    ap.add_argument('--model', default='pretrained/blind_video_net.pt')
    ap.add_argument('--sigmas', default='15,25')
    ap.add_argument('--blind', action='store_true',
                    help='trained blind model: load old=False, use estimated sigma (single output)')
    ap.add_argument('--outname', default='udvd_s')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', dev, flush=True)
    model, _, _ = utils.load_model(a.model, parallel=True, old=(not a.blind))
    model.to(dev).eval()
    vid, fps = read_video(a.input); T, H, W, _ = vid.shape
    print(f'loaded {T} frames {H}x{W}', flush=True)
    P = 32; ph, pw = (P-H % P) % P, (P-W % P) % P
    v = np.pad(vid, ((0,0),(0,ph),(0,pw),(0,0)), mode='reflect')
    frames = torch.from_numpy(v.transpose(0,3,1,2)).contiguous().to(dev)   # [T,3,H2,W2]
    T2 = frames.shape[0]
    def get(t):
        idx = [min(max(t+o,0),T2-1) for o in (-2,-1,0,1,2)]
        return torch.cat([frames[i] for i in idx], 0).unsqueeze(0)         # [1,15,H2,W2]
    os.makedirs(a.outdir, exist_ok=True)
    if a.blind:
        # trained blind model: use the network's estimated per-pixel sigma, single output
        out = np.empty((T, H, W, 3), np.float32)
        with torch.no_grad():
            for t in range(T2):
                inp = get(t)
                o, est = model(inp)
                den, _ = utils.post_process(o, inp[:, 6:9], model="blind-video-net", sigma=est, device=dev)
                out[t] = den[0].cpu().numpy().transpose(1,2,0)[:H, :W]
        write_video(os.path.join(a.outdir, f'{a.outname}.mp4'), out, 15)
        print('wrote blind output', flush=True)
    else:
        for s in [float(x) for x in a.sigmas.split(',')]:
            out = np.empty((T, H, W, 3), np.float32)
            with torch.no_grad():
                for t in range(T2):
                    inp = get(t)
                    o, _ = model(inp)
                    den, _ = utils.post_process(o, inp[:, 6:9], model="blind-video-net", sigma=s/255., device=dev)
                    out[t] = den[0].cpu().numpy().transpose(1,2,0)[:H, :W]
            write_video(os.path.join(a.outdir, f'udvd_zs_s{int(s)}.mp4'), out, 15)
            print('wrote sigma', s, flush=True)
    print('done')

if __name__ == '__main__':
    main()
