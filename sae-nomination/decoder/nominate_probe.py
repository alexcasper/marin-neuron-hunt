import glob, json, os, numpy as np, torch, pyarrow.parquet as pq
from sparsify import Sae
L = int(os.environ["NOM_LAYER"]); LL = f"{L:02d}"
ck = sorted(glob.glob(f"/data/scratch/apx186/marin32b/sae-64k/layer={LL}/sae_L{LL}w64_final_s*"))[0]
sae = Sae.load_from_disk(ck, device="cpu")
Wd = sae.W_dec
n = Wd.shape[0]
top1 = np.empty(n, dtype=np.float32); eff = np.empty(n, dtype=np.float32); l1s = np.empty(n, dtype=np.float32)
with torch.no_grad():
    for i in range(0, n, 4096):
        A = Wd[i:i+4096].detach().float().abs()
        l1 = A.sum(-1)
        P = A / l1[:, None].clamp_min(1e-12)
        l1s[i:i+4096] = l1.numpy()
        top1[i:i+4096] = P.max(-1).values.numpy()
        eff[i:i+4096] = (1.0 / (P*P).sum(-1).clamp_min(1e-12)).numpy()
        del A, P, l1
c = pq.read_table(f"/data/scratch/apx186/marin32b/sae-64k/layer={LL}/census_uniform.parquet")
alive = np.array(c.column("alive").to_numpy(), dtype=bool)
q = lambda a: {p: round(float(np.percentile(a, p)), 5) for p in (50, 90, 99, 100)}
print(f"LAYER {L} alive={int(alive.sum())} ckpt={os.path.basename(ck)}", flush=True)
print(" top1 ALIVE:", q(top1[alive]), flush=True)
print(" top1 DEAD :", q(top1[~alive]), flush=True)
print(" eff  ALIVE:", q(eff[alive]), flush=True)
print(" l1   ALIVE:", q(l1s[alive]), flush=True)
