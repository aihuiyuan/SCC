# train_bert_frozen.py
import os, json, argparse, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding,
                          EarlyStoppingCallback, set_seed)

# ===== 数据与指标 =====
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tok    = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        enc["labels"] = self.labels[i]
        return enc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"acc": acc, "macro_f1": macro_f1}

# ===== 冻结工具 =====
def freeze_encoder(model):
    base = getattr(model, "bert", None) or getattr(model, "roberta", None) or model.base_model
    for p in base.embeddings.parameters():
        p.requires_grad = False
    for layer in base.encoder.layer:
        for p in layer.parameters():
            p.requires_grad = False
    # 只保留分类头可训练（若存在）
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def count_trainable(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all   = sum(p.numel() for p in model.parameters())
    print(f"[INFO] trainable={n_train} / total={n_all}")
    return n_train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--freeze_mode", choices=["all","encoder"], default="all",
                    help="all=全冻结仅评估；encoder=冻结骨干，仅训分类头")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=3, id2label={0:"NEG",1:"NEU",2:"POS"},
        label2id={"NEG":0,"NEU":1,"POS":2}
    )

    # 冻结
    if args.freeze_mode == "all":
        freeze_all(model)
    else:
        freeze_encoder(model)
    n_train = count_trainable(model)

    # 数据
    df_tr = pd.read_csv(args.train)
    df_va = pd.read_csv(args.valid)
    train_ds = TextDataset(df_tr, tok, args.max_len)
    valid_ds = TextDataset(df_va, tok, args.max_len)
    collator = DataCollatorWithPadding(tok)

    # 训练参数
    targs = TrainingArguments(
        output_dir=args.out_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.out_dir, "logs"),
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # 训练或仅评估
    if args.freeze_mode == "all":
        print("[INFO] freeze_mode=all → 跳过训练，仅评估初始性能")
    else:
        if n_train == 0:
            raise RuntimeError("freeze_mode=encoder 但没有可训练参数")
        trainer.train()

    metrics = trainer.evaluate()
    with open(os.path.join(args.out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存当前模型（便于复现）
    best_dir = os.path.join(args.out_dir, "best")
    model.save_pretrained(best_dir, safe_serialization=True)
    tok.save_pretrained(best_dir)
    print(f"[INFO] Saved to {best_dir}")

if __name__ == "__main__":
    main()
