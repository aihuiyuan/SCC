import argparse, random, re, pandas as pd
from pathlib import Path

DEF_AUG_PER_SAMPLE = 1       
DEF_METHODS = ["synonym","insert","delete"]  
INS_WORDS = ["真的","其实","有点","挺","非常","比较","可能","稍微","特别","相当","还是"]

SYN_MAP = {
    "好": ["很好","不错","给力","棒","优秀","出色","赞"],
    "不错": ["挺好","很好","不错","还可以","可以"],
    "喜欢": ["喜爱","中意","很爱","挺喜欢","偏爱"],
    "满意": ["称心","合意","挺满意","还满意"],
    "推荐": ["安利","值得买","建议购买","值得推荐"],
    "差": ["糟","不好","很差","差劲","不行"],
    "垃圾": ["稀烂","很烂","渣","不堪","一塌糊涂"],
    "失望": ["不满意","很失望","沮丧","糟心","心累"],
    "一般": ["还行","可以","中等","凑合","中规中矩","一般般","尚可"],
}
EMO_PAT = re.compile(r"EMO_[^\s]+")  
NUM_PAT = re.compile(r"#NUM#")

def safe_split(text):

    parts, i = [], 0
    while i < len(text):
        m = EMO_PAT.search(text, i) or NUM_PAT.search(text, i)
        if m and m.start() == i:
            parts.append(m.group())
            i = m.end(); continue
        ch = text[i]
        parts.append(ch)
        i += 1
    return parts

def synonym_replace(text, p=0.15):
    toks = safe_split(text)
    for i, tk in enumerate(toks):
        if tk in (" ",) or tk.startswith("EMO_") or tk == "#NUM#": continue
        if random.random() < p:
            cands = []
        
            if tk in SYN_MAP: cands += SYN_MAP[tk]
            if i+1 < len(toks):
                bi = tk + toks[i+1]
                if bi in SYN_MAP: cands += SYN_MAP[bi]
            if cands:
                toks[i] = random.choice(cands)
    out = "".join(toks)
    return out if out != text else text

def random_insert(text, k=1):
    toks = safe_split(text)
    for _ in range(k):
        pos = random.randrange(0, len(toks)+1)
        ins = random.choice(INS_WORDS)
        toks.insert(pos, ins)
    return "".join(toks)

def random_delete(text, p=0.1):
    toks = safe_split(text)
    kept=[]
    for tk in toks:
        if tk.startswith("EMO_") or tk == "#NUM#":
            kept.append(tk); continue
        if random.random() < p: 
            continue
        kept.append(tk)
    out = "".join(kept)

    return out if len(out.strip())>=2 else text

def back_translate(text):
    
    try:
        from googletrans import Translator
        tr = Translator()
        en = tr.translate(text, src="zh-cn", dest="en").text
        zh = tr.translate(en, src="en", dest="zh-cn").text
        return zh
    except Exception:
        return text  

METHOD_FUN = {
    "synonym": lambda s: synonym_replace(s, p=0.15),
    "insert":  lambda s: random_insert(s, k=1),
    "delete":  lambda s: random_delete(s, p=0.12),
    "backtranslate": back_translate,
}

def main(args):
    src = Path(args.in_csv)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)  
    texts = df["text"].astype(str).tolist()

    methods = args.methods.split(",")
    for m in methods:
        if m not in METHOD_FUN:
            raise ValueError(f"未知方法: {m}")

    aug_rows = []
    random.seed(args.seed)
    for t in texts:
        for _ in range(args.aug_per_sample):
            m = random.choice(methods)
            t2 = METHOD_FUN[m](t)
            if t2 and t2 != t:
                aug_rows.append({"text": t2, "aug_type": m})
    aug_df = pd.DataFrame(aug_rows).drop_duplicates("text")

    aug_path  = out_dir / "aug_clean.csv"
    comb_path = out_dir / "clean_with_aug.csv"
    aug_df.to_csv(aug_path, index=False, encoding="utf-8")
    comb = pd.concat([df[["text"]], aug_df[["text"]]], axis=0).drop_duplicates("text").reset_index(drop=True)
    comb.to_csv(comb_path, index=False, encoding="utf-8")

    print({"src": len(df), "aug": len(aug_df), "combined": len(comb)})
    print(f"saved: {aug_path}\nmerged: {comb_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  default="/home/huiyuan/workspace/sentiment/data/clean_step1.csv")
    ap.add_argument("--out_dir", default="/home/huiyuan/workspace/sentiment/data")
    ap.add_argument("--methods", default="synonym,insert,delete")   
    ap.add_argument("--aug_per_sample", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)