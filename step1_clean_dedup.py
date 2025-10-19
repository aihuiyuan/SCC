import re, json, unicodedata, argparse, pandas as pd, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--in_xlsx", default="/home/huiyuan/workspace/sentiment/data/unlabeled.xlsx")
ap.add_argument("--sheet", default=0, help="Excel工作表名或索引")
ap.add_argument("--col",   default=0, type=int, help="文本所在列的索引(从0开始)")
ap.add_argument("--out_dir", default="/home/huiyuan/workspace/sentiment/data")
args = ap.parse_args()
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

def to_halfwidth(s:str)->str:

    return ''.join(unicodedata.normalize('NFKC', c) for c in s)


re_url   = re.compile(r"(https?://\S+|www\.\S+|t\.cn/\S+)", re.I)
re_at    = re.compile(r"@\w+")
re_topic = re.compile(r"#\S+#")
re_mail  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
re_phone = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
re_space = re.compile(r"\s+")
re_num   = re.compile(r"\d+")

re_emo   = re.compile(r"\[([^\[\]\s]{1,16})\]")

def compress_punct(s:str)->str:
    
    s = re.sub(r"[!！]{2,}", "！", s)
    s = re.sub(r"[?？]{2,}", "？", s)
    s = re.sub(r"[.。]{3,}", "。", s)
    s = re.sub(r"(.)\1{3,}", r"\1\1\1", s)
    return s

def keep_charset(s:str)->str:
    
    tokens = []
    def _protect(m):
        tokens.append(m.group(0))
        return f"[[EMO_PROT_{len(tokens)-1}]]"
    s = re.sub(r"EMO_[^\s]+", _protect, s)

    s = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9，。！？；：、“”‘’—\-_\s\[\]]", " ", s)
    
    for i,t in enumerate(tokens):
        s = s.replace(f"[[EMO_PROT_{i}]]", t)
    return s

def normalize(text:str)->str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    t = text.strip()
    t = to_halfwidth(t)
    t = re_url.sub(" ", t)
    t = re_at.sub(" ", t)
    t = re_topic.sub(" ", t)
    t = re_mail.sub(" ", t)
    t = re_phone.sub(" ", t)
    
    t = re_emo.sub(lambda m: f" EMO_{m.group(1)} ", t)
    t = compress_punct(t)
    t = re_space.sub(" ", t).strip()
    t = keep_charset(t)

    t = re_num.sub("#NUM#", t)
    return t

def make_key(norm:str)->str:
    
    k = norm.lower()
    k = re.sub(r"[，。！？；：“”‘’—\-\s_]", "", k)
    return k


df = pd.read_excel(args.in_xlsx, sheet_name=args.sheet, header=None, usecols=[args.col])
df = df.rename(columns={df.columns[0]: "text"})
total = len(df)


df["text"] = df["text"].astype(str).str.strip()
df["norm_text"] = df["text"].apply(normalize)


def is_bad(row)->bool:
    t = row["text"].strip()
    n = row["norm_text"]
    
    zh_len = len(re.findall(r"[\u4e00-\u9fff]", n))
    if zh_len < 3 and len(n) < 6:
        return True
    
    tmp = re.sub(r"EMO_[^\s]+|#NUM#", "", n)
    tmp = re.sub(r"[，。！？；：“”‘’—\-\s_]", "", tmp)
    if len(tmp) == 0:
        return True
    
    if re.search(r"(.)\1{10,}", n):
        return True
    return False

bad_mask = df.apply(is_bad, axis=1)
bad_cnt = int(bad_mask.sum())
df = df[~bad_mask].copy()


dup_strict_cnt = int(df.duplicated(subset=["norm_text"]).sum())
df = df.drop_duplicates(subset=["norm_text"]).copy()


df["key_text"] = df["norm_text"].apply(make_key)
dup_near_cnt = int(df.duplicated(subset=["key_text"]).sum())
df = df.drop_duplicates(subset=["key_text"]).copy()


keep_cols = ["text", "norm_text"]
clean_path = out_dir / "clean_step1.csv"
df[keep_cols].to_csv(clean_path, index=False, encoding="utf-8")

stats = {
    "total": int(total),
    "removed_bad": bad_cnt,
    "removed_strict_dup": dup_strict_cnt,
    "removed_near_dup": dup_near_cnt,
    "kept": int(len(df)),
}
with open(out_dir / "clean_step1_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(json.dumps(stats, ensure_ascii=False))
print(f"saved: {clean_path}")
