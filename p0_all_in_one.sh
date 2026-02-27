#!/bin/bash
set -e

# ============================================================
#  P0 All-in-One: 环境 → CUDA修复 → P0-a(单卡) → P0-b(8卡并行)
#  用法: bash p0_all_in_one.sh [GPU数量，默认8]
#  不用torchrun，每张卡一个独立进程，数据分片并行
# ============================================================

NUM_GPUS=${1:-8}

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH="$SHARED/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG="$SHARED/data/datasets/coco_val2017/val2017"
COCO_ANN="$SHARED/data/datasets/coco_val2017/annotations/instances_val2017.json"
RESULT_DIR="$SHARED/results"
LOG_DIR="$SHARED/logs"
CODE_DIR="$SHARED/code"
WHEELS="$SHARED/data/wheels"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$CODE_DIR"

echo "================================================================"
echo "  P0 All-in-One: CED公式验证 (${NUM_GPUS}卡并行)"
echo "  CED = JS(P||P^cf) + λ·[H(P^cf)-H(P)]_+"
echo "  通过标准: correct_positive vs hallucination AUC > 0.85"
echo "================================================================"

# ============================================================
# PART 1: 环境
# ============================================================
echo ""
echo "[1/4] 环境搭建..."

if [ ! -d "$SHARED/venv/p0_env" ]; then
    $SHARED/tools/python3.10/bin/python3.10 -m venv $SHARED/venv/p0_env
fi
source $SHARED/venv/p0_env/bin/activate
cd "$WHEELS"

pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    torch torchvision torchaudio 2>/dev/null || true

# huggingface_hub: 装最新版，--no-deps跳过httpx（本地模型不需要）
NEWEST_HF=$(ls -v "$WHEELS"/huggingface_hub*.whl 2>/dev/null | tail -1)
if [ -n "$NEWEST_HF" ]; then
    echo "  安装 huggingface_hub: $(basename $NEWEST_HF) (--no-deps跳过httpx)"
    pip install --force-reinstall --no-index --no-cache-dir --no-deps \
        --no-warn-script-location "$NEWEST_HF" 2>/dev/null || true
fi

# transformers带依赖
TRANS_WHL=$(ls -t "$WHEELS"/transformers*.whl 2>/dev/null | head -1)
if [ -n "$TRANS_WHL" ]; then
    pip install --no-cache-dir --find-links=. --no-warn-script-location \
        --no-deps "$TRANS_WHL" 2>/dev/null || true
fi

# 其他依赖
pip install --no-cache-dir --find-links=. --no-warn-script-location \
    accelerate qwen-vl-utils pillow numpy scipy pandas \
    tqdm scikit-learn pycocotools matplotlib \
    regex tokenizers safetensors filelock packaging pyyaml requests 2>/dev/null || true

# 检查关键import
echo "  检查依赖..."
for mod in regex tokenizers safetensors torch transformers; do
    python -c "import $mod" 2>/dev/null || {
        echo "    ✗ $mod 缺失"; pip install $mod -q 2>/dev/null || true
    }
done

python -c "from transformers import Qwen3VLForConditionalGeneration; print('  ✓ transformers OK')" || {
    echo "  ✗ transformers导入失败"
    pip show transformers huggingface_hub 2>/dev/null | grep -E '^(Name|Version)'
    exit 1
}

# ============================================================
# PART 2: CUDA修复
# ============================================================
echo ""
echo "[2/4] CUDA修复..."

SITE_PACKAGES="$SHARED/venv/p0_env/lib/python3.10/site-packages"
NVIDIA_LIB_PATHS=""
for d in "$SITE_PACKAGES"/nvidia/*/lib "$SITE_PACKAGES/torch/lib"; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="$d:$NVIDIA_LIB_PATHS"
done
[ -n "$NVIDIA_LIB_PATHS" ] && export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

NVJIT="$SITE_PACKAGES/nvidia/nvjitlink/lib/libnvJitLink.so.12"
[ -f "$NVJIT" ] && export LD_PRELOAD="${NVJIT}${LD_PRELOAD:+:$LD_PRELOAD}"

python -c "import torch; assert torch.cuda.is_available(); print(f'  ✓ PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs')" || {
    echo "  ✗ CUDA不可用"; exit 1
}

# ============================================================
# PART 3: 写入Python代码
# ============================================================
echo ""
echo "[3/4] 写入实验代码..."

cat > "$CODE_DIR/p0_all.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
P0 All-in-One
  --mode probe   : P0-a 架构探测（单卡）
  --mode worker   : P0-b 单个GPU worker（处理一个shard）
  --mode analyze  : 合并所有shard结果 + 分析出图
"""
import os, sys, json, random, warnings, argparse, glob
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings("ignore")

MODEL_PATH = os.environ["P0_MODEL"]
COCO_IMG   = os.environ["P0_COCO_IMG"]
COCO_ANN   = os.environ["P0_COCO_ANN"]
RESULT_DIR = os.environ["P0_RESULTS"]

# ═══════════════════════════════════════════════
#  度量
# ═══════════════════════════════════════════════

def _sm(x):
    x = np.asarray(x, dtype=np.float64); x -= x.max()
    e = np.exp(x); return e/e.sum()

def _p(x):
    p = np.clip(np.asarray(x, dtype=np.float64), 1e-12, None); return p/p.sum()

def js(p, q):
    p, q = _p(p), _p(q); m = .5*(p+q)
    return float(.5*entropy(p,m,base=2)+.5*entropy(q,m,base=2))

def kl(p, q):
    return float(entropy(_p(p), _p(q), base=2))

def H(p):
    return float(entropy(_p(p), base=2))

def hid_js(h1, h2, k=4096):
    a, b = h1.cpu().float().numpy(), h2.cpu().float().numpy()
    d = np.union1d(np.argsort(np.abs(a))[-k:], np.argsort(np.abs(b))[-k:])
    return js(_sm(a[d]), _sm(b[d]))

def metrics(ol, cl, lams):
    p, q = _sm(ol), _sm(cl)
    j = js(p,q); k_ = kl(p,q); ho = H(p); hc = H(q); ep = max(0., hc-ho)
    r = {"js":j,"kl":k_,"h_orig":ho,"h_cf":hc,"ep":ep}
    for l in lams:
        r[f"ced_{l:.2f}"] = j+l*ep
        r[f"ced_abs_{l:.2f}"] = j+l*abs(hc-ho)
        r[f"ced_hcf_{l:.2f}"] = j+l*hc
    return r

# ═══════════════════════════════════════════════
#  Hook
# ═══════════════════════════════════════════════

class VTHook:
    def __init__(self): self.cap=None; self._m=None; self._h=None
    def reg(self, model):
        self._h = model.visual.register_forward_hook(self._fn); return self
    def rm(self):
        if self._h: self._h.remove()
    def reset(self): self.cap=None; self._m=None
    def set_rep(self, m): self._m = m
    def _fn(self, mod, inp, out):
        if self._m is None:
            self.cap = out.detach().clone(); return out
        r = self.cap.clone().to(device=out.device, dtype=out.dtype)
        for i,v in self._m.items():
            if 0<=i<r.shape[0]: r[i]=v.to(device=r.device, dtype=r.dtype)
        return r

# ═══════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════

def bbox2idx(bb, iw, ih, gh, gw):
    x,y,w,h = bb
    cs,ce = max(0,int(x/iw*gw)), min(gw,int(np.ceil((x+w)/iw*gw)))
    rs,re = max(0,int(y/ih*gh)), min(gh,int(np.ceil((y+h)/ih*gh)))
    return [r*gw+c for r in range(rs,re) for c in range(cs,ce)]

def ctridx(bbs, iw, ih, gh, gw, n):
    occ = set()
    for b in bbs: occ.update(bbox2idx(b, iw, ih, gh, gw))
    free = sorted(set(range(gh*gw))-occ)
    if len(free)>=n: s=random.randint(0,len(free)-n); return free[s:s+n]
    return free if free else list(range(min(n,gh*gw)))

def mk_inp(proc, img, q, dev):
    from qwen_vl_utils import process_vision_info
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q}]}]
    txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii,vi = process_vision_info(msgs)
    inp = proc(text=[txt], images=ii, videos=vi, padding=True, return_tensors="pt")
    return {k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in inp.items()}

def get_ans(model, proc, img, q, dev):
    inp = mk_inp(proc, img, q, dev)
    with torch.no_grad(): ids = model.generate(**inp, max_new_tokens=20)
    t = proc.decode(ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    if "yes" in t: return 1, t
    if "no" in t: return 0, t
    return -1, t

# ═══════════════════════════════════════════════
#  P0-a: 架构探测
# ═══════════════════════════════════════════════

def run_probe(model, proc, dev):
    print("\n"+"="*60+"\nP0-a: 架构探测\n"+"="*60)
    with open(COCO_ANN) as f: coco=json.load(f)
    info=coco["images"][0]; path=os.path.join(COCO_IMG,info["file_name"])
    img=Image.open(path).convert("RGB"); iw,ih=img.size
    cm={c["id"]:c["name"] for c in coco["categories"]}
    anns=[a for a in coco["annotations"] if a["image_id"]==info["id"]]
    if anns:
        ann=max(anns,key=lambda a:a["bbox"][2]*a["bbox"][3])
        obj=cm[ann["category_id"]]; bb=ann["bbox"]
    else: obj="object"; bb=[iw*.25,ih*.25,iw*.5,ih*.5]
    print(f"图片: {info['file_name']} ({iw}x{ih}), 物体: {obj}")

    inp=mk_inp(proc,img,f"Is there a {obj} in this image? Answer yes or no.",dev)
    gthw=inp["image_grid_thw"]; assert gthw is not None
    gt_,gh,gw=gthw[0].tolist(); print(f"grid: t={gt_},h={gh},w={gw}")

    cap={}
    def _c(m,i,o): cap["s"]=o.shape; cap["t"]=o.detach().clone(); return o
    h=model.visual.register_forward_hook(_c)
    with torch.no_grad(): orig=model(**inp,output_hidden_states=True,return_dict=True)
    h.remove()
    assert "s" in cap, "Hook失败！"
    vs=cap["s"]; act=vs[0] if len(vs)==2 else vs[1]; exp=gt_*gh*gw
    print(f"visual: {list(vs)}, tokens={act}, expected={exp}")
    tok_ok = abs(act-exp)<10

    nl=len(orig.hidden_states); hd=vs[-1]
    print(f"hidden_dim={hd}, layers={nl}")

    # 替换物体区域
    ti=bbox2idx(bb,iw,ih,gh,gw); vf=cap["t"]
    si=sorted(set(range(vf.shape[0]))-set(ti))
    rp=vf[si].mean(0) if si else vf.mean(0)
    mf=vf.clone()
    for i in ti:
        if i<mf.shape[0]: mf[i]=rp
    def _r(m,i,o): return mf.to(device=o.device,dtype=o.dtype)
    h2=model.visual.register_forward_hook(_r)
    with torch.no_grad(): cf=model(**inp,output_hidden_states=True,return_dict=True)
    h2.remove()
    ol_=orig.logits[0,-1].cpu().float().numpy()
    cl_=cf.logits[0,-1].cpu().float().numpy()
    js_obj=js(_sm(ol_),_sm(cl_))

    # 角落控制
    ci=[r*gw+c for r in range(min(2,gh)) for c in range(min(2,gw))]
    cs=sorted(set(range(vf.shape[0]))-set(ci))
    cr=vf[cs].mean(0) if cs else vf.mean(0)
    cf2=vf.clone()
    for i in ci:
        if i<cf2.shape[0]: cf2[i]=cr
    def _r2(m,i,o): return cf2.to(device=o.device,dtype=o.dtype)
    h3=model.visual.register_forward_hook(_r2)
    with torch.no_grad(): co=model(**inp,output_hidden_states=True,return_dict=True)
    h3.remove()
    js_c=js(_sm(ol_),_sm(co.logits[0,-1].cpu().float().numpy()))

    rep_ok=js_obj>1e-6; obj_ok=js_obj>js_c
    print(f"\nJS(物体)={js_obj:.6f}, JS(角落)={js_c:.6f}")
    print(f"  {'✅' if rep_ok else '❌'} 替换影响输出")
    print(f"  {'✅' if obj_ok else '⚠️'} 物体>角落 ({js_obj/max(js_c,1e-12):.1f}x)")

    ok=tok_ok and rep_ok and obj_ok
    sl=[l for l in [16,20,24,28,32] if l<nl]
    nfo={"gh":gh,"gw":gw,"gt":gt_,"hd":hd,"nl":nl,"sl":sl,
         "js_obj":float(js_obj),"js_c":float(js_c),"ok":ok}
    with open(f"{RESULT_DIR}/p0a_info.json","w") as f: json.dump(nfo,f,indent=2)
    print(f"\n{'✅ P0-a PASSED' if ok else '❌ P0-a FAILED'}")
    return ok, nfo

# ═══════════════════════════════════════════════
#  P0-b: 数据加载
# ═══════════════════════════════════════════════

SPATIAL=[
    ("Is the {o} on the left side of the image?",  lambda b,w,h:b[0]+b[2]/2<w/2),
    ("Is the {o} on the right side of the image?", lambda b,w,h:b[0]+b[2]/2>w/2),
    ("Is the {o} in the upper half of the image?", lambda b,w,h:b[1]+b[3]/2<h/2),
    ("Is the {o} in the lower half of the image?", lambda b,w,h:b[1]+b[3]/2>h/2),
]

def load_samples(n=400):
    with open(COCO_ANN) as f: coco=json.load(f)
    cm={c["id"]:c["name"] for c in coco["categories"]}
    im={i["id"]:i for i in coco["images"]}
    ia=defaultdict(list)
    for a in coco["annotations"]: ia[a["image_id"]].append(a)
    ic={iid:set(a["category_id"] for a in aa) for iid,aa in ia.items()}
    ac=set(cm.keys())

    samps=[]; ids=list(ia.keys()); random.shuffle(ids)
    for iid in ids:
        if len(samps)>=n: break
        aa=ia[iid]; info=im[iid]; w,h=info["width"],info["height"]
        p=os.path.join(COCO_IMG,info["file_name"])
        if not os.path.exists(p): continue
        va=[a for a in aa if a["bbox"][2]*a["bbox"][3]/(w*h)>0.02 and not a.get("iscrowd",0)]
        if not va: continue
        tgt=max(va,key=lambda a:a["bbox"][2]*a["bbox"][3])
        cat=cm[tgt["category_id"]]; bb=tgt["bbox"]; abb=[a["bbox"] for a in aa]

        samps.append(dict(task="existence",path=p,w=w,h=h,obj=cat,bb=bb,abb=abb,gt=1,
                          q=f"Is there a {cat} in this image? Answer yes or no."))
        absent=ac-ic.get(iid,set())
        if absent:
            an=cm[random.choice(list(absent))]
            samps.append(dict(task="existence",path=p,w=w,h=h,obj=an,bb=bb,abb=abb,gt=0,
                              q=f"Is there a {an} in this image? Answer yes or no."))
        tmpl,fn=random.choice(SPATIAL); gs=1 if fn(bb,w,h) else 0
        samps.append(dict(task="spatial",path=p,w=w,h=h,obj=cat,bb=bb,abb=abb,gt=gs,
                          q=tmpl.format(o=cat)+" Answer yes or no."))
        same=[a for a in va if a["category_id"]==tgt["category_id"]]; cnt=len(same)
        ask=(cnt+random.choice([1,2]) if random.random()<.5 else max(1,cnt-1)) if random.random()<.5 else cnt
        samps.append(dict(task="counting",path=p,w=w,h=h,obj=cat,bb=bb,abb=abb,
                          gt=1 if ask==cnt else 0,
                          q=f"Are there exactly {ask} {cat}(s) in this image? Answer yes or no."))
    return samps

# ═══════════════════════════════════════════════
#  P0-b: 单样本处理
# ═══════════════════════════════════════════════

def proc_one(s, model, proc, hook, dev, layers, lams):
    try: img=Image.open(s["path"]).convert("RGB")
    except: return None

    pred,ptxt=get_ans(model,proc,img,s["q"],dev)
    if pred==-1: return None
    gt=s["gt"]
    if gt==1 and pred==1: beh="correct_positive"
    elif gt==0 and pred==1: beh="hallucination"
    elif gt==0 and pred==0: beh="correct_negative"
    elif gt==1 and pred==0: beh="miss"
    else: return None

    inp=mk_inp(proc,img,s["q"],dev)
    gthw=inp.get("image_grid_thw")
    if gthw is None: return None
    gh,gw=gthw[0,1].item(),gthw[0,2].item()

    hook.reset()
    with torch.no_grad(): orig=model(**inp,output_hidden_states=True,return_dict=True)
    if hook.cap is None: return None

    ti=bbox2idx(s["bb"],s["w"],s["h"],gh,gw)
    if not ti: return None
    vf=hook.cap
    si=sorted(set(range(vf.shape[0]))-set(ti))
    rp=vf[si].mean(0) if si else vf.mean(0)

    hook.set_rep({i:rp for i in ti})
    with torch.no_grad(): cf=model(**inp,output_hidden_states=True,return_dict=True)
    hook.reset()

    ol=orig.logits[0,-1].cpu().float().numpy()
    cl=cf.logits[0,-1].cpu().float().numpy()
    m=metrics(ol,cl,lams)
    m["cosine"]=float(cosine_dist(
        orig.hidden_states[-1][0,-1].cpu().float().numpy(),
        cf.hidden_states[-1][0,-1].cpu().float().numpy()))
    for lay in layers:
        if lay<len(orig.hidden_states):
            m[f"L{lay}_js"]=hid_js(orig.hidden_states[lay][0,-1],cf.hidden_states[lay][0,-1])

    r={"task":s["task"],"obj":s["obj"],"gt":gt,"pred":pred,"beh":beh,
       "n_tgt":len(ti),"n_tot":gh*gw,**m}

    if beh=="correct_positive":
        ci=ctridx(s["abb"],s["w"],s["h"],gh,gw,max(4,len(ti)))
        csurr=sorted(set(range(vf.shape[0]))-set(ci))
        cr=vf[csurr].mean(0) if csurr else vf.mean(0)
        hook.reset()
        with torch.no_grad(): _=model(**inp,output_hidden_states=False,return_dict=True)
        hook.set_rep({i:cr for i in ci})
        with torch.no_grad(): co=model(**inp,output_hidden_states=False,return_dict=True)
        hook.reset()
        cm=metrics(ol,co.logits[0,-1].cpu().float().numpy(),lams)
        r["ctrl_js"]=cm["js"]
        for l in lams: r[f"ctrl_ced_{l:.2f}"]=cm[f"ced_{l:.2f}"]

    return r

# ═══════════════════════════════════════════════
#  P0-b worker: 处理一个shard
# ═══════════════════════════════════════════════

def run_worker(args):
    sid, ns = args.shard_id, args.num_shards
    lams = args.lambda_e
    print(f"[Worker {sid}/{ns}] 启动，GPU: {os.environ.get('CUDA_VISIBLE_DEVICES','?')}")

    dev = torch.device("cuda:0")  # 每个worker只看到自己的1张卡

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    hook = VTHook().reg(model)

    with open(f"{RESULT_DIR}/p0a_info.json") as f: info=json.load(f)
    layers = info.get("sl", [16,20,24,28,32])

    # 加载全部样本，取自己的shard
    all_samps = load_samples(n=args.num_samples)
    my_samps = [s for i,s in enumerate(all_samps) if i%ns==sid]
    print(f"[Worker {sid}] 总样本{len(all_samps)}, 本shard: {len(my_samps)}")

    results = []
    bc = defaultdict(int)
    for s in tqdm(my_samps, desc=f"W{sid}", position=sid):
        r = proc_one(s, model, proc, hook, dev, layers, lams)
        if r:
            results.append(r); bc[r["beh"]]+=1

    hook.rm()
    df = pd.DataFrame(results)
    out = f"{RESULT_DIR}/p0b_shard_{sid}.csv"
    df.to_csv(out, index=False)
    print(f"[Worker {sid}] 完成: {len(df)}条, 行为={dict(bc)}, 保存到 {out}")

# ═══════════════════════════════════════════════
#  P0-b analyze: 合并 + 分析
# ═══════════════════════════════════════════════

def safe_auc(lab, sc):
    if len(set(lab))<2 or len(lab)<10: return float("nan")
    return roc_auc_score(lab, sc)

def run_analyze(args):
    lams = args.lambda_e

    # 合并所有shard
    files = sorted(glob.glob(f"{RESULT_DIR}/p0b_shard_*.csv"))
    if not files:
        print("没有找到shard结果文件！"); sys.exit(1)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(f"{RESULT_DIR}/p0b_results.csv", index=False)
    print(f"\n合并 {len(files)} 个shard，共 {len(df)} 条结果")

    print("\n"+"="*60+"\n分析结果\n"+"="*60)

    print("\n--- 行为分布 ---")
    for b in ["correct_positive","hallucination","correct_negative","miss"]:
        s=df[df["beh"]==b]
        if len(s)>0: print(f"  {b:20s}: n={len(s):4d}, JS={s['js'].mean():.4f}±{s['js'].std():.4f}")

    cp=df[df["beh"]=="correct_positive"]; hal=df[df["beh"]=="hallucination"]

    if len(cp)<5 or len(hal)<5:
        print(f"\n⚠️  样本不足 (cp={len(cp)}, hal={len(hal)})")
        if df["gt"].nunique()>1:
            print("\n--- 退化: gt=1 vs gt=0 ---")
            for m in ["js"]+[f"ced_{l:.2f}" for l in lams]:
                if m in df: print(f"  {m:20s}: AUC={safe_auc(df['gt'],df[m]):.4f}")
        return

    sub=pd.concat([cp.assign(label=1),hal.assign(label=0)])
    best_a, best_n = 0, ""

    print("\n--- 核心AUC: correct_positive vs hallucination ---")
    a=safe_auc(sub["label"],sub["js"]); print(f"  A. 裸JS:               AUC={a:.4f}")
    if a>best_a: best_a,best_n=a,"js"

    for l in lams:
        c=f"ced_{l:.2f}"
        if c in sub:
            a=safe_auc(sub["label"],sub[c]); tag=" ★" if a>best_a else ""
            print(f"  B. CED(λ={l:.2f}):       AUC={a:.4f}{tag}")
            if a>best_a: best_a,best_n=a,c

    for c,nm in [("ced_hcf_0.10","C. JS+λH(P^cf)"),("ced_abs_0.10","D. JS+λ|ΔH|")]:
        if c in sub:
            a=safe_auc(sub["label"],sub[c]); print(f"  {nm:23s}: AUC={a:.4f}")
            if a>best_a: best_a,best_n=a,c

    a=safe_auc(sub["label"],sub["kl"]); print(f"  E. KL(P||P^cf):        AUC={a:.4f}")
    if a>best_a: best_a,best_n=a,"kl"

    if "cosine" in sub:
        a=safe_auc(sub["label"],sub["cosine"]); print(f"  F. Cosine:             AUC={a:.4f}")
        if a>best_a: best_a,best_n=a,"cosine"

    print(f"\n  → 最优: {best_n}, AUC={best_a:.4f}")

    # 层
    with open(f"{RESULT_DIR}/p0a_info.json") as f: info=json.load(f)
    layers=info.get("sl",[16,20,24,28,32])
    print("\n--- 各层 ---")
    for lay in layers:
        c=f"L{lay}_js"
        if c in sub and sub[c].notna().sum()>10:
            print(f"  Layer {lay}: AUC={safe_auc(sub['label'],sub[c]):.4f}")

    # 跨任务
    print("\n--- 跨任务 ---")
    for task in ["existence","spatial","counting"]:
        t=df[df["task"]==task]; c1=t[t["beh"]=="correct_positive"]; c2=t[t["beh"]=="hallucination"]
        if len(c1)>=3 and len(c2)>=3:
            s2=pd.concat([c1.assign(label=1),c2.assign(label=0)])
            print(f"  {task:12s}: AUC={safe_auc(s2['label'],s2['js']):.4f} (cp={len(c1)},hal={len(c2)})")
        else: print(f"  {task:12s}: 样本不足 (cp={len(c1)},hal={len(c2)})")

    # 控制组
    print("\n--- 控制组 ---")
    if "ctrl_js" in cp.columns:
        cr=cp[cp["ctrl_js"].notna()]
        if len(cr)>0:
            print(f"  物体区域JS:  {cr['js'].mean():.4f}±{cr['js'].std():.4f}")
            print(f"  无物体区域JS: {cr['ctrl_js'].mean():.4f}±{cr['ctrl_js'].std():.4f}")
            print(f"  比值: {cr['js'].mean()/max(cr['ctrl_js'].mean(),1e-8):.1f}x")

    # 熵
    print("\n--- 熵 ---")
    for b in ["correct_positive","hallucination","correct_negative"]:
        s=df[df["beh"]==b]
        if len(s)>0: print(f"  {b:20s}: H(P)={s['h_orig'].mean():.3f}, H(P^cf)={s['h_cf'].mean():.3f}, Δ+={s['ep'].mean():.4f}")

    # ── 可视化 ──
    colors={"correct_positive":"green","hallucination":"red","correct_negative":"blue","miss":"orange"}
    fig,axes=plt.subplots(2,3,figsize=(20,12))

    ax=axes[0,0]
    for b,c in colors.items():
        s=df[df["beh"]==b]
        if len(s)>2: ax.hist(s["js"],bins=25,alpha=.4,color=c,label=f"{b}({len(s)})",density=True)
    ax.set_xlabel("JS"); ax.set_title("JS by Behavior"); ax.legend(fontsize=7)

    ax=axes[0,1]
    if sub["label"].nunique()>1:
        ns_,as_=[],[]
        ns_.append("JS"); as_.append(safe_auc(sub["label"],sub["js"]))
        for l in lams:
            c=f"ced_{l:.2f}"
            if c in sub: ns_.append(f"CED λ={l}"); as_.append(safe_auc(sub["label"],sub[c]))
        ns_.append("KL"); as_.append(safe_auc(sub["label"],sub["kl"]))
        if "cosine" in sub: ns_.append("Cos"); as_.append(safe_auc(sub["label"],sub["cosine"]))
        ax.barh(ns_,as_,color=["g" if a>.85 else "orange" if a>.75 else "r" for a in as_])
        ax.axvline(.85,color="r",ls="--",alpha=.5); ax.set_xlim(.5,1); ax.set_title("Formula Ablation")

    ax=axes[0,2]
    if sub["label"].nunique()>1:
        ln,la=["logits"],[safe_auc(sub["label"],sub["js"])]
        for l in layers:
            c=f"L{l}_js"
            if c in sub and sub[c].notna().sum()>10: ln.append(f"L{l}"); la.append(safe_auc(sub["label"],sub[c]))
        ax.bar(ln,la,color=["g" if a>.85 else "orange" if a>.75 else "r" for a in la])
        ax.axhline(.85,color="r",ls="--",alpha=.5); ax.set_ylim(.5,1); ax.set_title("Layer AUC")

    ax=axes[1,0]
    if sub["label"].nunique()>1:
        fpr,tpr,_=roc_curve(sub["label"],sub["js"])
        ax.plot(fpr,tpr,lw=2,label=f"JS AUC={safe_auc(sub['label'],sub['js']):.3f}")
        ax.plot([0,1],[0,1],"k--",alpha=.3); ax.set_title("ROC"); ax.legend()

    ax=axes[1,1]
    if "ctrl_js" in cp.columns and cp["ctrl_js"].notna().sum()>5:
        ax.boxplot([cp["js"].dropna(),cp["ctrl_js"].dropna()],labels=["Object","Control"])
        ax.set_ylabel("JS"); ax.set_title("Object vs Control")

    ax=axes[1,2]
    ta={}
    for task in ["existence","spatial","counting"]:
        t=df[df["task"]==task]; c1=t[t["beh"]=="correct_positive"]; c2=t[t["beh"]=="hallucination"]
        if len(c1)>=3 and len(c2)>=3:
            s2=pd.concat([c1.assign(label=1),c2.assign(label=0)])
            ta[task]=safe_auc(s2["label"],s2["js"])
    if ta:
        ax.bar(ta.keys(),ta.values(),color=["g" if v>.85 else "orange" for v in ta.values()])
        ax.axhline(.85,color="r",ls="--",alpha=.5); ax.set_ylim(.5,1); ax.set_title("Cross-task")

    plt.tight_layout(); plt.savefig(f"{RESULT_DIR}/p0b_analysis.png",dpi=150); plt.close()
    print(f"\n图表: {RESULT_DIR}/p0b_analysis.png")

    print("\n"+"="*60)
    if best_a>.85: print(f"✅ P0-b PASSED: AUC={best_a:.4f} > 0.85 ({best_n})\n   → Phase 1!")
    elif best_a>.75: print(f"⚠️  P0-b MARGINAL: AUC={best_a:.4f}")
    else: print(f"❌ P0-b FAILED: AUC={best_a:.4f}")
    print("="*60)

# ═══════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--mode",choices=["probe","worker","analyze"],required=True)
    pa.add_argument("--shard_id",type=int,default=0)
    pa.add_argument("--num_shards",type=int,default=8)
    pa.add_argument("--num_samples",type=int,default=400)
    pa.add_argument("--lambda_e",type=float,nargs="+",default=[0.0,0.05,0.1,0.2,0.5])
    pa.add_argument("--seed",type=int,default=42)
    args=pa.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.mode=="probe":
        dev=torch.device("cuda:0")
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print(f"加载模型: {MODEL_PATH}")
        model=Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,torch_dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
        proc=AutoProcessor.from_pretrained(MODEL_PATH)
        ok,_=run_probe(model,proc,dev)
        if not ok: sys.exit(1)

    elif args.mode=="worker":
        run_worker(args)

    elif args.mode=="analyze":
        run_analyze(args)
PYTHON_EOF

echo "  ✓ 代码已写入 $CODE_DIR/p0_all.py"

# ============================================================
# PART 4: 运行
# ============================================================

export P0_MODEL="$MODEL_PATH"
export P0_COCO_IMG="$COCO_IMG"
export P0_COCO_ANN="$COCO_ANN"
export P0_RESULTS="$RESULT_DIR"

cd "$CODE_DIR"

# ── P0-a: 单卡探测 ──
echo ""
echo "[4a/4] P0-a 架构探测（单卡）..."
CUDA_VISIBLE_DEVICES=0 python p0_all.py --mode probe --seed 42 \
    2>&1 | tee "$LOG_DIR/p0a.log"

if ! grep -q "P0-a PASSED" "$LOG_DIR/p0a.log"; then
    echo "P0-a未通过！"; exit 1
fi

# ── P0-b: 8卡并行 ──
echo ""
echo "[4b/4] P0-b CED验证（${NUM_GPUS}卡并行）..."

# 清理旧shard文件
rm -f "$RESULT_DIR"/p0b_shard_*.csv

PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python p0_all.py \
        --mode worker \
        --shard_id $i \
        --num_shards $NUM_GPUS \
        --num_samples 400 \
        --seed 42 \
        > "$LOG_DIR/p0b_worker_${i}.log" 2>&1 &
    PIDS+=($!)
    echo "  启动 Worker $i (PID ${PIDS[-1]}, GPU $i)"
done

# 等待所有worker完成
echo "  等待 ${NUM_GPUS} 个worker..."
FAIL=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]} || {
        echo "  ✗ Worker $i 失败！查看: $LOG_DIR/p0b_worker_${i}.log"
        FAIL=1
    }
done

if [ $FAIL -ne 0 ]; then
    echo "部分worker失败，检查日志:"; ls -la "$LOG_DIR"/p0b_worker_*.log; exit 1
fi

echo "  所有worker完成！"

# ── 合并 + 分析 ──
echo ""
echo "[4c/4] 合并结果 + 分析..."
python p0_all.py --mode analyze --seed 42 \
    2>&1 | tee "$LOG_DIR/p0b_analyze.log"

echo ""
echo "================================================================"
echo "  ✓ P0实验完成！"
echo "  结果: $RESULT_DIR/p0b_results.csv"
echo "  图表: $RESULT_DIR/p0b_analysis.png"
echo "  日志: $LOG_DIR/"
echo "================================================================"
