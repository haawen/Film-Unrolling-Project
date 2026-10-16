"""Patch UDVD single_train.py for stable training: skip non-finite batches +
gradient clipping (prevents the mid-epoch NaN blow-up seen in the blind run)."""
p = 'single_train.py'
s = open(p).read()
old = ("            model.zero_grad()\n"
       "            loss.backward()\n"
       "            optimizer.step()")
new = ("            if not torch.isfinite(loss):\n"
       "                optimizer.zero_grad(); continue\n"
       "            model.zero_grad()\n"
       "            loss.backward()\n"
       "            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
       "            optimizer.step()")
n = s.count(old)
if "clip_grad_norm_" in s:
    print("already patched")
else:
    s = s.replace(old, new)
    open(p, 'w').write(s)
    print("patched", n, "block(s)")
