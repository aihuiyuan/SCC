
import re, argparse, pandas as pd
from sklearn.model_selection import train_test_split

POS_EMO = {"微笑","笑脸","哈哈","笑哭","开心","喜欢","爱你","赞","强","鼓掌","玫瑰","比心","好耶","牛","OK"}
NEG_EMO = {"生气","愤怒","怒","哭","流泪","悲伤","鄙视","吐","晕","抓狂","心碎","烦","无语","尴尬","难过","衰"}
NEU_EMO = {"思考","冷汗","困惑","疑问","捂脸","叹气","汗","斜眼笑","doge","摊手","无奈","囧"}

POS_W = ["很好","满意","喜欢","推荐","值得","不错","给力","优秀","香","真香","太棒"]
NEG_W = ["差","失望","垃圾","不好","糟糕","退货","坑人","翻车","破","渣","恶心","后悔"]
NEU_W = ["一般","还行","可以","中等","中规中矩","马马虎虎","凑合","一般般","尚可"]

emo_pat = re.compile(r"EMO_([^\s]+)")
sq_pat  = re.compile(r"\[([^\[\]\s]{1,16})\]") 

def label_text(text, norm):
    s = "" if pd.isna(text) else str(text)
    n = "" if pd.isna(norm) else str(norm)
    emos = set(emo_pat.findall(n)) or set(sq_pat.findall(s))

    pos = sum(1 for e in emos if e in POS_EMO)
    neg = sum(1 for e in emos if e in NEG_EMO)
    neu = sum(1 for e in emos if e in NEU_EMO)

    pos += sum(s.count(w) for w in POS_W)
    neg += sum(s.count(w) for w in NEG_W)
    neu += sum(s.count(w) for w in NEU_W)

    scores = [neg, neu, pos]  
    mx = max(scores)
    if mx == 0:               
        return 1, 0
    order = sorted(scores, reverse=True)
    conf = order[0] - (order[1] if len(order) > 1 else 0)
    return scores.index(mx), conf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--drop_lowconf", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if "text" not in df.columns:
        raise ValueError("输入CSV需要包含列: text（可选: norm_text）")

    labels, confs = [], []
    for _, r in df.iterrows():
        lbl, cf = label_text(r.get("text"), r.get("norm_text"))
        labels.append(lbl); confs.append(cf)
    df["label"] = labels; df["conf"] = confs
    if args.drop_lowconf:
        df = df[df["conf"] >= 1]

    data = df[["text","label"]].dropna()
    tr, va = train_test_split(
        data, test_size=args.valid_ratio, stratify=data["label"], random_state=args.seed
    )
    tr.to_csv(f"{args.out_dir}/train.csv", index=False)
    va.to_csv(f"{args.out_dir}/valid.csv", index=False)

    print("train size:", len(tr), "valid size:", len(va))
    print("train dist:", tr["label"].value_counts().to_dict())
    print("valid dist:", va["label"].value_counts().to_dict())

if __name__ == "__main__":
    main()
