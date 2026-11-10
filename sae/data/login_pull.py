import concurrent.futures as cf
import json, os, subprocess, threading, time
BASE = "https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/resolve/main"
ATLAS = "/data/SPCS-IOT/marin-32b/atlas"
files = json.load(open("/data/scratch/apx186/marin32b/filelist.json"))
todo = [f for f in files
        if not (os.path.exists(os.path.join(ATLAS, f["path"]))
                and os.path.getsize(os.path.join(ATLAS, f["path"])) == f["size"])]
print(f"to fetch: {len(todo)}/{len(files)}", flush=True)
done_ct, lock = [0], threading.Lock()

def fetch(f):
    dst = os.path.join(ATLAS, f["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        sz = os.path.getsize(dst)
        if sz == 0 or sz > f["size"]:
            os.remove(dst)
    for t in range(6):
        if t > 0 and os.path.exists(dst) and os.path.getsize(dst) == 0:
            os.remove(dst)
        r = subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "3",
                            "--retry-delay", "5", "--max-time", "3600",
                            "-o", dst, BASE + "/" + f["path"]], capture_output=True)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
            with lock:
                done_ct[0] += 1
                if done_ct[0] % 25 == 0:
                    print(f"{done_ct[0]}/{len(todo)}", flush=True)
            return None
        time.sleep(5 * (t + 1))
    return f"FAILED {f['path']} rc={r.returncode}"

t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    fails = [r for r in ex.map(fetch, todo) if r]
for m in fails[:40]:
    print(m, flush=True)
print(f"PULL_DONE {len(todo)-len(fails)} ok {len(fails)} failed in {time.time()-t0:.0f}s", flush=True)
