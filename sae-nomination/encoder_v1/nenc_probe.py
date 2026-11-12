import glob, os, numpy as np, torch, pyarrow.parquet as pq
from sparsify import Sae
torch.set_num_threads(8)
L=57; LL="57"; SCR="/data/scratch/apx186/marin32b"
ck = sorted(glob.glob(f"{SCR}/sae-64k/layer={LL}/sae_L{LL}w64_final_s*"))[0]
sae = Sae.load_from_disk(ck, device="cpu"); sae.eval()
PARQ=f"{SCR}/atlas-resorted/layer{LL}-tokensorted.parquet"
pf=pq.ParquetFile(PARQ)
t=pf.read_row_group(0,columns=["token_idx","neuron","gate","up"]).to_pydict()
tok=np.asarray(t["token_idx"],dtype=np.int64); neu=np.asarray(t["neuron"],dtype=np.int64)
g=np.asarray(t["gate"],dtype=np.float32); u=np.asarray(t["up"],dtype=np.float32)
act=g/(1.0+np.exp(-g))*u
utok,inv=np.unique(tok,return_inverse=True)
m=np.zeros((len(utok),27648),dtype=np.float32); m[inv,neu]=act
xb=torch.from_numpy(m).to(torch.bfloat16)
codes=sae.encode(xb)
zp=((xb-sae.b_dec.detach())@sae.encoder.weight.detach().T)+sae.encoder.bias.detach()
zpr=torch.relu(zp)
print("pre_acts dtype:", codes.pre_acts.dtype, "zp dtype:", zpr.dtype, flush=True)
print("value diff max:", float((zpr.float()-codes.pre_acts.float()).abs().max()), flush=True)
pos=(zpr>0).sum(1)
print("positives/row: min",int(pos.min()),"p50",int(pos.median()),"frac<32:",float((pos<32).float().mean()), flush=True)
k=codes.top_indices.shape[1]
mine=torch.relu(zp).topk(k,dim=1).indices
ov=torch.tensor([len(set(a.tolist())&set(b.tolist()))/k for a,b in zip(mine,codes.top_indices)])
print("overlap mean:",float(ov.mean()),"rows<0.99:",int((ov<0.99).sum()),"of",len(ov), flush=True)
# boundary ties in encode's own preacts
v,_=codes.pre_acts.float().topk(k+1,dim=1)
ties=(v[:,-1]==v[:,-2]).sum()
print("boundary ties in encode preacts:",int(ties),"of",len(ov), flush=True)
bad=(ov<0.99).nonzero().flatten()[:3]
for r in bad.tolist():
    a=set(mine[r].tolist()); b=set(codes.top_indices[r].tolist())
    print("row",r,"ov",float(ov[r]),"| preact my-kth:",float(torch.sort(zpr[r].float()).values[-k]),"enc-kth:",float(torch.sort(codes.pre_acts[r].float()).values[-k]), flush=True)
