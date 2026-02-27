#!/bin/bash
set -e

# ============================================================
#  P0 All-in-One: 环境搭建 → CUDA修复 → P0-a架构探测 → P0-b CED验证
#  用法: bash p0_all_in_one.sh
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH="$SHARED/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG="$SHARED/data/datasets/coco_val2017/val2017"
COCO_ANN="$SHARED/data/datasets/coco_val2017/annotations/instances_val2017.json"
RESULT_DIR="$SHARED/results"
LOG_DIR="$SHARED/logs"
CODE_DIR="$SHARED/code"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$CODE_DIR"

echo "================================================================"
echo "  P0 All-in-One: CED公式验证"
echo "  CED = JS(P||P^cf) + λ·[H(P^cf)-H(P)]_+"
echo "  通过标准: correct_positive vs hallucination AUC > 0.85"
echo "================================================================"

# ============================================================
# PART 1: 环境搭建
# ============================================================
echo ""
echo "[1/4] 环境搭建..."

if [ ! -d "$SHARED/venv/p0_env" ]; then
    echo "  创建虚拟环境..."
    $SHARED/tools/python3.10/bin/python3.10 -m venv $SHARED/venv/p0_env
fi
source $SHARED/venv/p0_env/bin/activate

cd $SHARED/data/wheels
echo "  安装依赖..."
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    torch torchvision torchaudio 2>/dev/null || true
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    accelerate huggingface_hub qwen-vl-utils pillow numpy scipy pandas \
    tqdm scikit-learn datasets pycocotools gdown matplotlib 2>/dev/null || true
for whl in transformers*.whl; do
    [ -f "$whl" ] && pip install --no-index --no-cache-dir --find-links=. \
        --no-warn-script-location --no-deps "$whl" 2>/dev/null && break
done

# ============================================================
# PART 2: CUDA nvJitLink 修复
# ============================================================
echo ""
echo "[2/4] 修复CUDA库路径..."

SITE_PACKAGES="$SHARED/venv/p0_env/lib/python3.10/site-packages"
NVIDIA_LIB_PATHS=""
if [ -d "$SITE_PACKAGES/nvidia" ]; then
    for d in "$SITE_PACKAGES"/nvidia/*/lib; do
        [ -d "$d" ] && NVIDIA_LIB_PATHS="$d:$NVIDIA_LIB_PATHS"
    done
fi
[ -d "$SITE_PACKAGES/torch/lib" ] && NVIDIA_LIB_PATHS="$SITE_PACKAGES/torch/lib:$NVIDIA_LIB_PATHS"

if [ -n "$NVIDIA_LIB_PATHS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "  ✓ torch nvidia库已加入LD_LIBRARY_PATH"
fi

NVJIT_LIB="$SITE_PACKAGES/nvidia/nvjitlink/lib/libnvJitLink.so.12"
[ -f "$NVJIT_LIB" ] && export LD_PRELOAD="${NVJIT_LIB}${LD_PRELOAD:+:$LD_PRELOAD}" && \
    echo "  ✓ LD_PRELOAD兜底已设置"

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA不可用'
x = torch.randn(4,4,device='cuda')
print(f'  ✓ PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
" || {
    echo "  ✗ PyTorch CUDA测试失败"
    echo "    1) 检查驱动: nvidia-smi | grep 'CUDA Version'"
    echo "    2) 如果 < 12.4，需换torch: pip download torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121"
    echo "    3) 或安装匹配nvjitlink: pip install nvidia-nvjitlink-cu12==12.4.127"
    exit 1
}

# ============================================================
# PART 3 & 4: 写入Python代码并运行
# ============================================================
echo ""
echo "[3/4] 写入实验代码..."

cat > "$CODE_DIR/p0_all.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
P0 All-in-One: 架构探测(P0-a) + CED信号验证(P0-b)
"""
import os, sys, json, random, warnings, argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import roc_auc_score, roc_curve
warnings.filterwarnings("ignore")

# ── paths (from env) ──
MODEL_PATH  = os.environ["P0_MODEL"]
COCO_IMG    = os.environ["P0_COCO_IMG"]
COCO_ANN    = os.environ["P0_COCO_ANN"]
RESULT_DIR  = os.environ["P0_RESULTS"]

# ════════════════════════════════════════════════════════════
#  度量函数
# ════════════════════════════════════════════════════════════

def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _prob(x):
    p = np.clip(np.asarray(x, dtype=np.float64), 1e-12, None)
    return p / p.sum()

def js_div(p, q):
    p, q = _prob(p), _prob(q)
    m = 0.5*(p+q)
    return float(0.5*entropy(p,m,base=2) + 0.5*entropy(q,m,base=2))

def kl_div(p, q):
    return float(entropy(_prob(p), _prob(q), base=2))

def h_bits(p):
    return float(entropy(_prob(p), base=2))

def hidden_js(h1, h2, top_k=4096):
    a, b = h1.cpu().float().numpy(), h2.cpu().float().numpy()
    dims = np.union1d(np.argsort(np.abs(a))[-top_k:], np.argsort(np.abs(b))[-top_k:])
    return js_div(_softmax(a[dims]), _softmax(b[dims]))

def all_metrics(orig_logits, cf_logits, lam_vals):
    p, q = _softmax(orig_logits), _softmax(cf_logits)
    js = js_div(p, q)
    kl = kl_div(p, q)
    ho, hc = h_bits(p), h_bits(q)
    ep = max(0.0, hc - ho)
    r = {"js": js, "kl": kl, "h_orig": ho, "h_cf": hc, "entropy_penalty": ep}
    for l in lam_vals:
        r[f"ced_{l:.2f}"] = js + l*ep
        r[f"ced_abs_{l:.2f}"] = js + l*abs(hc-ho)
        r[f"ced_hcf_{l:.2f}"] = js + l*hc
    return r

# ════════════════════════════════════════════════════════════
#  Hook
# ════════════════════════════════════════════════════════════

class VTHook:
    def __init__(self):
        self.cap = None
        self._mods = None
        self._h = None
    def register(self, model):
        self._h = model.visual.register_forward_hook(self._fn); return self
    def remove(self):
        if self._h: self._h.remove()
    def reset(self):
        self.cap = None; self._mods = None
    def set_replace(self, m):
        self._mods = m
    def _fn(self, mod, inp, out):
        if self._mods is None:
            self.cap = out.detach().clone(); return out
        r = self.cap.clone().to(device=out.device, dtype=out.dtype)
        for i, v in self._mods.items():
            if 0 <= i < r.shape[0]:
                r[i] = v.to(device=r.device, dtype=r.dtype)
        return r

# ════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════

def bbox_to_idx(bbox, iw, ih, gh, gw):
    x,y,w,h = bbox
    cs, ce = max(0,int(x/iw*gw)), min(gw, int(np.ceil((x+w)/iw*gw)))
    rs, re = max(0,int(y/ih*gh)), min(gh, int(np.ceil((y+h)/ih*gh)))
    return [r*gw+c for r in range(rs,re) for c in range(cs,ce)]

def ctrl_idx(all_bboxes, iw, ih, gh, gw, n):
    occ = set()
    for b in all_bboxes:
        occ.update(bbox_to_idx(b, iw, ih, gh, gw))
    free = sorted(set(range(gh*gw)) - occ)
    if len(free) >= n:
        s = random.randint(0, len(free)-n)
        return free[s:s+n]
    return free if free else list(range(min(n, gh*gw)))

def make_inputs(processor, image, question, device):
    from qwen_vl_utils import process_vision_info
    msgs = [{"role":"user","content":[
        {"type":"image","image":image},
        {"type":"text","text":question}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inp = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt")
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in inp.items()}

def get_answer(model, processor, image, question, device):
    inp = make_inputs(processor, image, question, device)
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=20)
    txt = processor.decode(ids[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    if "yes" in txt: return 1, txt
    if "no"  in txt: return 0, txt
    return -1, txt

# ════════════════════════════════════════════════════════════
#  P0-a: 架构探测
# ════════════════════════════════════════════════════════════

def run_p0a(model, processor, device):
    print("\n" + "="*60)
    print("P0-a: 架构探测")
    print("="*60)

    with open(COCO_ANN) as f: coco = json.load(f)
    img_info = coco["images"][0]
    img_path = os.path.join(COCO_IMG, img_info["file_name"])
    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size
    cat_map = {c["id"]:c["name"] for c in coco["categories"]}
    anns = [a for a in coco["annotations"] if a["image_id"]==img_info["id"]]
    if anns:
        ann = max(anns, key=lambda a: a["bbox"][2]*a["bbox"][3])
        obj = cat_map[ann["category_id"]]; bbox = ann["bbox"]
    else:
        obj = "object"; bbox = [iw*.25, ih*.25, iw*.5, ih*.5]
    print(f"图片: {img_info['file_name']} ({iw}x{ih}), 物体: {obj}")

    question = f"Is there a {obj} in this image? Answer yes or no."
    inp = make_inputs(processor, img, question, device)

    grid_thw = inp.get("image_grid_thw")
    assert grid_thw is not None, "image_grid_thw不存在！"
    gt, gh, gw = grid_thw[0].tolist()
    print(f"grid_thw: t={gt}, h={gh}, w={gw}, total={gt*gh*gw}")

    # ── hook capture ──
    cap = {}
    def _cap(m, i, o):
        cap["shape"] = o.shape; cap["tensor"] = o.detach().clone(); return o
    h = model.visual.register_forward_hook(_cap)
    with torch.no_grad():
        orig = model(**inp, output_hidden_states=True, return_dict=True)
    h.remove()

    assert "shape" in cap, "Hook未捕获到输出！"
    vs = cap["shape"]
    actual = vs[0] if len(vs)==2 else vs[1]
    expected = gt*gh*gw
    print(f"visual输出: {list(vs)}, actual_tokens={actual}, expected={expected}")
    tok_match = abs(actual - expected) < 10
    print(f"  {'✅' if tok_match else '❌'} token数量{'匹配' if tok_match else '不匹配'}")

    hdim = vs[-1]
    n_layers = len(orig.hidden_states)
    print(f"hidden_dim={hdim}, n_layers={n_layers}")

    # ── 替换物体区域 ──
    tidx = bbox_to_idx(bbox, iw, ih, gh, gw)
    vf = cap["tensor"]
    surr = sorted(set(range(vf.shape[0])) - set(tidx))
    repl = vf[surr].mean(0) if surr else vf.mean(0)

    mod_f = vf.clone()
    for i in tidx:
        if i < mod_f.shape[0]: mod_f[i] = repl

    def _rep(m, i, o):
        return mod_f.to(device=o.device, dtype=o.dtype)
    h2 = model.visual.register_forward_hook(_rep)
    with torch.no_grad():
        cf = model(**inp, output_hidden_states=True, return_dict=True)
    h2.remove()

    ol = orig.logits[0,-1].cpu().float().numpy()
    cl = cf.logits[0,-1].cpu().float().numpy()
    js_obj = js_div(_softmax(ol), _softmax(cl))

    # ── 替换角落（控制） ──
    cidx = [r*gw+c for r in range(min(2,gh)) for c in range(min(2,gw))]
    csurr = sorted(set(range(vf.shape[0])) - set(cidx))
    crepl = vf[csurr].mean(0) if csurr else vf.mean(0)
    cf2 = vf.clone()
    for i in cidx:
        if i < cf2.shape[0]: cf2[i] = crepl

    def _rep2(m, i, o):
        return cf2.to(device=o.device, dtype=o.dtype)
    h3 = model.visual.register_forward_hook(_rep2)
    with torch.no_grad():
        co = model(**inp, output_hidden_states=True, return_dict=True)
    h3.remove()

    col = co.logits[0,-1].cpu().float().numpy()
    js_ctrl = js_div(_softmax(ol), _softmax(col))

    replace_works = js_obj > 1e-6
    obj_stronger  = js_obj > js_ctrl

    print(f"\nJS(物体区域)={js_obj:.6f}, JS(角落)={js_ctrl:.6f}")
    print(f"  {'✅' if replace_works else '❌'} 替换影响输出")
    print(f"  {'✅' if obj_stronger else '⚠️'} 物体>角落 ({js_obj/max(js_ctrl,1e-12):.1f}x)")

    all_pass = tok_match and replace_works and obj_stronger
    safe_layers = [l for l in [16,20,24,28,32] if l < n_layers]

    info = {"gh":gh,"gw":gw,"gt":gt,"hdim":hdim,"n_layers":n_layers,
            "js_obj":float(js_obj),"js_ctrl":float(js_ctrl),
            "safe_layers":safe_layers,"passed":all_pass}
    with open(f"{RESULT_DIR}/p0a_info.json","w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'✅ P0-a PASSED' if all_pass else '❌ P0-a FAILED'}")
    return all_pass, info

# ════════════════════════════════════════════════════════════
#  P0-b: CED信号验证
# ════════════════════════════════════════════════════════════

SPATIAL = [
    ("Is the {o} on the left side of the image?",  lambda b,w,h: b[0]+b[2]/2<w/2),
    ("Is the {o} on the right side of the image?",  lambda b,w,h: b[0]+b[2]/2>w/2),
    ("Is the {o} in the upper half of the image?",  lambda b,w,h: b[1]+b[3]/2<h/2),
    ("Is the {o} in the lower half of the image?",  lambda b,w,h: b[1]+b[3]/2>h/2),
]

def load_samples(n=400, min_ratio=0.02):
    with open(COCO_ANN) as f: coco = json.load(f)
    cat_map = {c["id"]:c["name"] for c in coco["categories"]}
    img_map = {i["id"]:i for i in coco["images"]}
    img_anns = defaultdict(list)
    for a in coco["annotations"]: img_anns[a["image_id"]].append(a)
    img_cats = {iid: set(a["category_id"] for a in aa) for iid,aa in img_anns.items()}
    all_cids = set(cat_map.keys())

    samples = []; ids = list(img_anns.keys()); random.shuffle(ids)
    for iid in ids:
        if len(samples) >= n: break
        aa = img_anns[iid]; info = img_map[iid]
        w, h = info["width"], info["height"]
        p = os.path.join(COCO_IMG, info["file_name"])
        if not os.path.exists(p): continue
        valid = [a for a in aa if a["bbox"][2]*a["bbox"][3]/(w*h)>min_ratio and not a.get("iscrowd",0)]
        if not valid: continue
        tgt = max(valid, key=lambda a: a["bbox"][2]*a["bbox"][3])
        cat = cat_map[tgt["category_id"]]; bbox = tgt["bbox"]
        all_bb = [a["bbox"] for a in aa]

        # 存在性正例
        samples.append(dict(task="existence", path=p, w=w, h=h, obj=cat, bbox=bbox, all_bb=all_bb, gt=1,
                            q=f"Is there a {cat} in this image? Answer yes or no."))
        # 存在性反例
        absent = all_cids - img_cats.get(iid, set())
        if absent:
            an = cat_map[random.choice(list(absent))]
            samples.append(dict(task="existence", path=p, w=w, h=h, obj=an, bbox=bbox, all_bb=all_bb, gt=0,
                                q=f"Is there a {an} in this image? Answer yes or no."))
        # 空间
        tmpl, fn = random.choice(SPATIAL)
        gs = 1 if fn(bbox,w,h) else 0
        samples.append(dict(task="spatial", path=p, w=w, h=h, obj=cat, bbox=bbox, all_bb=all_bb, gt=gs,
                            q=tmpl.format(o=cat)+" Answer yes or no."))
        # 计数
        same = [a for a in valid if a["category_id"]==tgt["category_id"]]
        cnt = len(same)
        ask = (cnt + random.choice([1,2]) if random.random()<0.5 else max(1,cnt-1)) if random.random()<0.5 else cnt
        samples.append(dict(task="counting", path=p, w=w, h=h, obj=cat, bbox=bbox, all_bb=all_bb,
                            gt=1 if ask==cnt else 0,
                            q=f"Are there exactly {ask} {cat}(s) in this image? Answer yes or no."))

    tc = defaultdict(int)
    for s in samples: tc[s["task"]] += 1
    print(f"  样本: {len(samples)} ({dict(tc)})")
    return samples

def process_one(s, model, processor, hook, dev, layers, lams):
    try:
        img = Image.open(s["path"]).convert("RGB")
    except: return None

    pred, ptxt = get_answer(model, processor, img, s["q"], dev)
    if pred == -1: return None

    gt = s["gt"]
    if   gt==1 and pred==1: beh = "correct_positive"
    elif gt==0 and pred==1: beh = "hallucination"
    elif gt==0 and pred==0: beh = "correct_negative"
    elif gt==1 and pred==0: beh = "miss"
    else: return None

    inp = make_inputs(processor, img, s["q"], dev)
    gthw = inp.get("image_grid_thw")
    if gthw is None: return None
    gh, gw = gthw[0,1].item(), gthw[0,2].item()

    # original forward
    hook.reset()
    with torch.no_grad():
        orig = model(**inp, output_hidden_states=True, return_dict=True)
    if hook.cap is None: return None

    # 物体区域替换
    tidx = bbox_to_idx(s["bbox"], s["w"], s["h"], gh, gw)
    if not tidx: return None
    vf = hook.cap
    surr = sorted(set(range(vf.shape[0]))-set(tidx))
    repl = vf[surr].mean(0) if surr else vf.mean(0)

    hook.set_replace({i:repl for i in tidx})
    with torch.no_grad():
        cf = model(**inp, output_hidden_states=True, return_dict=True)
    hook.reset()

    ol = orig.logits[0,-1].cpu().float().numpy()
    cl = cf.logits[0,-1].cpu().float().numpy()
    m = all_metrics(ol, cl, lams)

    # cosine
    m["cosine"] = float(cosine_dist(
        orig.hidden_states[-1][0,-1].cpu().float().numpy(),
        cf.hidden_states[-1][0,-1].cpu().float().numpy()))

    # 中间层
    for lay in layers:
        if lay < len(orig.hidden_states):
            m[f"L{lay}_js"] = hidden_js(orig.hidden_states[lay][0,-1], cf.hidden_states[lay][0,-1])

    r = {"task":s["task"],"obj":s["obj"],"gt":gt,"pred":pred,"beh":beh,
         "n_tgt":len(tidx),"n_tot":gh*gw, **m}

    # 控制组（仅correct_positive）
    if beh == "correct_positive":
        ci = ctrl_idx(s["all_bb"], s["w"], s["h"], gh, gw, max(4,len(tidx)))
        csurr = sorted(set(range(vf.shape[0]))-set(ci))
        cr = vf[csurr].mean(0) if csurr else vf.mean(0)
        hook.reset()
        with torch.no_grad(): _ = model(**inp, output_hidden_states=False, return_dict=True)
        hook.set_replace({i:cr for i in ci})
        with torch.no_grad(): co = model(**inp, output_hidden_states=False, return_dict=True)
        hook.reset()
        cm = all_metrics(ol, co.logits[0,-1].cpu().float().numpy(), lams)
        r["ctrl_js"] = cm["js"]
        for l in lams: r[f"ctrl_ced_{l:.2f}"] = cm[f"ced_{l:.2f}"]

    return r

def safe_auc(lab, sc):
    if len(set(lab))<2 or len(lab)<10: return float("nan")
    return roc_auc_score(lab, sc)

def run_p0b(model, processor, device, info, args):
    print("\n" + "="*60)
    print("P0-b: CED信号验证")
    print("="*60)

    hook = VTHook().register(model)
    layers = info.get("safe_layers", [16,20,24,28,32])
    lams = args.lambda_e

    samples = load_samples(n=args.num_samples)
    results = []; bc = defaultdict(int)

    for s in tqdm(samples, desc="P0-b"):
        r = process_one(s, model, processor, hook, device, layers, lams)
        if r:
            results.append(r); bc[r["beh"]] += 1
        if len(results)%50==0 and len(results)>0:
            df_t = pd.DataFrame(results)
            cp_t = df_t[df_t["beh"]=="correct_positive"]
            hl_t = df_t[df_t["beh"]=="hallucination"]
            if len(cp_t)>5 and len(hl_t)>5:
                sub = pd.concat([cp_t,hl_t])
                a = safe_auc((sub["beh"]=="correct_positive").astype(int), sub["js"])
                tqdm.write(f"  n={len(results)}, {dict(bc)}, AUC(js)={a:.3f}")

    hook.remove()
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULT_DIR}/p0b_results.csv", index=False)
    print(f"\n保存 {len(df)} 条结果到 {RESULT_DIR}/p0b_results.csv")

    # ── 分析 ──
    print("\n" + "="*60)
    print("分析结果")
    print("="*60)

    print("\n--- 行为分布 ---")
    for b in ["correct_positive","hallucination","correct_negative","miss"]:
        s = df[df["beh"]==b]
        if len(s)>0:
            print(f"  {b:20s}: n={len(s):4d}, JS={s['js'].mean():.4f}±{s['js'].std():.4f}")

    cp = df[df["beh"]=="correct_positive"]
    hal = df[df["beh"]=="hallucination"]

    if len(cp)<5 or len(hal)<5:
        print(f"\n⚠️  样本不足 (cp={len(cp)}, hal={len(hal)})")
        print("  模型在该数据上不怎么犯错，建议增加样本量")
        if df["gt"].nunique()>1:
            print("\n--- 退化方案: gt=1 vs gt=0 ---")
            for m in ["js"]+[f"ced_{l:.2f}" for l in lams]:
                if m in df: print(f"  {m:20s}: AUC={safe_auc(df['gt'],df[m]):.4f}")
        return

    sub = pd.concat([cp.assign(label=1), hal.assign(label=0)])

    print("\n--- 核心AUC: correct_positive vs hallucination ---")
    best_auc, best_name = 0, ""

    # A: 裸JS
    a = safe_auc(sub["label"], sub["js"])
    print(f"  A. 裸JS:               AUC={a:.4f}")
    if a > best_auc: best_auc, best_name = a, "js"

    # B: CED各λ
    for l in lams:
        c = f"ced_{l:.2f}"
        if c in sub:
            a = safe_auc(sub["label"], sub[c])
            tag = " ★" if a > best_auc else ""
            print(f"  B. CED(λ={l:.2f}):       AUC={a:.4f}{tag}")
            if a > best_auc: best_auc, best_name = a, c

    # C: JS+λH(P^cf)
    c = "ced_hcf_0.10"
    if c in sub:
        a = safe_auc(sub["label"], sub[c])
        print(f"  C. JS+λH(P^cf):        AUC={a:.4f}")
        if a > best_auc: best_auc, best_name = a, c

    # D: JS+λ|ΔH|
    c = "ced_abs_0.10"
    if c in sub:
        a = safe_auc(sub["label"], sub[c])
        print(f"  D. JS+λ|ΔH|:           AUC={a:.4f}")
        if a > best_auc: best_auc, best_name = a, c

    # E: KL
    a = safe_auc(sub["label"], sub["kl"])
    print(f"  E. KL(P||P^cf):        AUC={a:.4f}")
    if a > best_auc: best_auc, best_name = a, "kl"

    # F: cosine
    if "cosine" in sub:
        a = safe_auc(sub["label"], sub["cosine"])
        print(f"  F. Cosine dist:        AUC={a:.4f}")
        if a > best_auc: best_auc, best_name = a, "cosine"

    print(f"\n  → 最优: {best_name}, AUC={best_auc:.4f}")

    # 中间层
    print("\n--- 各层AUC ---")
    for lay in layers:
        c = f"L{lay}_js"
        if c in sub and sub[c].notna().sum()>10:
            print(f"  Layer {lay}: AUC={safe_auc(sub['label'], sub[c]):.4f}")

    # 跨任务
    print("\n--- 跨任务 ---")
    for task in ["existence","spatial","counting"]:
        t = df[df["task"]==task]
        c1, c2 = t[t["beh"]=="correct_positive"], t[t["beh"]=="hallucination"]
        if len(c1)>=3 and len(c2)>=3:
            s2 = pd.concat([c1.assign(label=1),c2.assign(label=0)])
            print(f"  {task:12s}: AUC={safe_auc(s2['label'],s2['js']):.4f} (cp={len(c1)},hal={len(c2)})")
        else:
            print(f"  {task:12s}: 样本不足 (cp={len(c1)},hal={len(c2)})")

    # 控制组
    print("\n--- 控制组 ---")
    cr = cp[cp.get("ctrl_js",pd.Series(dtype=float)).notna()] if "ctrl_js" in cp else pd.DataFrame()
    if len(cr)>0:
        print(f"  物体区域JS:  {cr['js'].mean():.4f}±{cr['js'].std():.4f}")
        print(f"  无物体区域JS: {cr['ctrl_js'].mean():.4f}±{cr['ctrl_js'].std():.4f}")
        print(f"  比值: {cr['js'].mean()/max(cr['ctrl_js'].mean(),1e-8):.1f}x")

    # 熵
    print("\n--- 熵分析 ---")
    for b in ["correct_positive","hallucination","correct_negative"]:
        s = df[df["beh"]==b]
        if len(s)>0:
            print(f"  {b:20s}: H(P)={s['h_orig'].mean():.3f}, H(P^cf)={s['h_cf'].mean():.3f}, Δ+={s['entropy_penalty'].mean():.4f}")

    # ── 可视化 ──
    fig, axes = plt.subplots(2, 3, figsize=(20,12))
    colors = {"correct_positive":"green","hallucination":"red","correct_negative":"blue","miss":"orange"}

    ax = axes[0,0]
    for b,c in colors.items():
        s = df[df["beh"]==b]
        if len(s)>2: ax.hist(s["js"],bins=25,alpha=.4,color=c,label=f"{b}({len(s)})",density=True)
    ax.set_xlabel("JS"); ax.set_title("JS by Behavior"); ax.legend(fontsize=7)

    ax = axes[0,1]
    if sub["label"].nunique()>1:
        names, aucs = [], []
        names.append("JS"); aucs.append(safe_auc(sub["label"],sub["js"]))
        for l in lams:
            c = f"ced_{l:.2f}"
            if c in sub: names.append(f"CED λ={l}"); aucs.append(safe_auc(sub["label"],sub[c]))
        names.append("KL"); aucs.append(safe_auc(sub["label"],sub["kl"]))
        if "cosine" in sub: names.append("Cos"); aucs.append(safe_auc(sub["label"],sub["cosine"]))
        ax.barh(names, aucs, color=["g" if a>.85 else "orange" if a>.75 else "r" for a in aucs])
        ax.axvline(.85,color="r",ls="--",alpha=.5); ax.set_xlim(.5,1); ax.set_title("Formula Ablation")

    ax = axes[0,2]
    if sub["label"].nunique()>1:
        ln, la = ["logits"], [safe_auc(sub["label"],sub["js"])]
        for l in layers:
            c = f"L{l}_js"
            if c in sub and sub[c].notna().sum()>10:
                ln.append(f"L{l}"); la.append(safe_auc(sub["label"],sub[c]))
        ax.bar(ln,la,color=["g" if a>.85 else "orange" if a>.75 else "r" for a in la])
        ax.axhline(.85,color="r",ls="--",alpha=.5); ax.set_ylim(.5,1); ax.set_title("Layer AUC")

    ax = axes[1,0]
    if sub["label"].nunique()>1:
        fpr,tpr,_ = roc_curve(sub["label"],sub["js"])
        ax.plot(fpr,tpr,lw=2,label=f"JS AUC={safe_auc(sub['label'],sub['js']):.3f}")
        ax.plot([0,1],[0,1],"k--",alpha=.3); ax.set_title("ROC"); ax.legend()

    ax = axes[1,1]
    if "ctrl_js" in cp and cp["ctrl_js"].notna().sum()>5:
        ax.boxplot([cp["js"].dropna(),cp["ctrl_js"].dropna()],labels=["Object","Control"])
        ax.set_ylabel("JS"); ax.set_title("Object vs Control")

    ax = axes[1,2]
    ta = {}
    for task in ["existence","spatial","counting"]:
        t = df[df["task"]==task]
        c1,c2 = t[t["beh"]=="correct_positive"],t[t["beh"]=="hallucination"]
        if len(c1)>=3 and len(c2)>=3:
            s2 = pd.concat([c1.assign(label=1),c2.assign(label=0)])
            ta[task] = safe_auc(s2["label"],s2["js"])
    if ta:
        ax.bar(ta.keys(),ta.values(),color=["g" if v>.85 else "orange" for v in ta.values()])
        ax.axhline(.85,color="r",ls="--",alpha=.5); ax.set_ylim(.5,1); ax.set_title("Cross-task")

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/p0b_analysis.png", dpi=150); plt.close()
    print(f"\n图表: {RESULT_DIR}/p0b_analysis.png")

    # ── Pass/Fail ──
    print("\n" + "="*60)
    if best_auc > 0.85:
        print(f"✅ P0-b PASSED: AUC={best_auc:.4f} > 0.85 ({best_name})")
        print("   → 进入 Phase 1 GRPO!")
    elif best_auc > 0.75:
        print(f"⚠️  P0-b MARGINAL: AUC={best_auc:.4f} (0.75~0.85)")
        print("   建议调整替换策略或增加样本")
    else:
        print(f"❌ P0-b FAILED: AUC={best_auc:.4f}")
        print("   建议: 切换VCD噪声/MaskCD/PROJECTAWAY")
    print("="*60)

# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--lambda_e", type=float, nargs="+", default=[0.0,0.05,0.1,0.2,0.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_p0a", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda:0")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"\n加载模型: {MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # P0-a
    if not args.skip_p0a:
        passed, info = run_p0a(model, processor, device)
        if not passed:
            print("\nP0-a未通过，退出。")
            sys.exit(1)
    else:
        with open(f"{RESULT_DIR}/p0a_info.json") as f:
            info = json.load(f)
        print("跳过P0-a，使用已有探测结果")

    # P0-b
    run_p0b(model, processor, device, info, args)

if __name__ == "__main__":
    main()
PYTHON_EOF

echo "  ✓ 代码已写入 $CODE_DIR/p0_all.py"

# ============================================================
# PART 4: 运行
# ============================================================
echo ""
echo "[4/4] 运行P0实验..."

export P0_MODEL="$MODEL_PATH"
export P0_COCO_IMG="$COCO_IMG"
export P0_COCO_ANN="$COCO_ANN"
export P0_RESULTS="$RESULT_DIR"

cd "$CODE_DIR"
CUDA_VISIBLE_DEVICES=0 python p0_all.py \
    --num_samples 400 \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/p0_all.log"

echo ""
echo "================================================================"
echo "  完成！"
echo "  结果: $RESULT_DIR/p0b_results.csv"
echo "  图表: $RESULT_DIR/p0b_analysis.png"
echo "  日志: $LOG_DIR/p0_all.log"
echo "================================================================"
