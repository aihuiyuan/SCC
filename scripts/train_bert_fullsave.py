import os, json, argparse, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback, set_seed
)

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len)
        enc["labels"] = self.labels[i]
        return enc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "acc": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=3,
        id2label={0:"NEG",1:"NEU",2:"POS"},
        label2id={"NEG":0,"NEU":1,"POS":2}
    )

    df_tr = pd.read_csv(args.train)
    df_va = pd.read_csv(args.valid)

    train_ds = TextDataset(df_tr, tok, args.max_len)
    valid_ds = TextDataset(df_va, tok, args.max_len)
    collator = DataCollatorWithPadding(tok)

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
        save_total_limit=1,                 # 只保留一个最优 checkpoint
        load_best_model_at_end=True,        # 训练完自动还原到最优
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

    trainer.train()
    metrics = trainer.evaluate()
    with open(os.path.join(args.out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # —— 保存“最优模型 + 分词器” —— #
    best_dir = os.path.join(args.out_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)          # 保存权重、配置、tokenizer 关联文件
    tok.save_pretrained(best_dir)
    # 也把标签映射单独存一下，方便部署
    with open(os.path.join(best_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": model.config.id2label, "label2id": model.config.label2id},
                  f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
