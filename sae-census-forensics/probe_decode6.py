import glob
import torch
from sparsify import Sae
from train_sae import val_fvu

SCR = "/data/scratch/apx186/marin32b"
CKPT = sorted(glob.glob(f"{SCR}/sae-64k/layer=00/sae_L00w64_final_s*"))[0]
PARQ = f"{SCR}/atlas-resorted/layer00-tokensorted.parquet"
sae = Sae.load_from_disk(CKPT, device="cuda:0"); sae.eval()
v = val_fvu(sae, PARQ, 1920000, 1024, None, "cuda:0", 30000)  # scale_vec=None, exactly as the sweep trained
print(f"val_fvu RAW acts (scale_vec=None, as trained): {v:.4f}")
