## 文件说明
- `data/unlabeled`：初始数据
- `data/clean_step1`：清洗后的数据
- `data/clean_with_aug`：在 `clean_step1` 基础上做数据增强的版本
- `data/aug_noemoji`：在`dclean_with_aug`基础上去除表情的最终版本
- `scripts/step1_clean_dedup`：清洗脚本
- `scripts/step2_augment`：增强脚本
- `scripts/step3_label_and_split`：打标签和切分脚本
- `scripts/train_bert`：解冻所有层训练脚本
- `scripts/train_bert_frozen`：冻住所有层训练脚本
- `SCC实验报告.pdf`：完整的实验报告文档

## 使用说明
在wsl上运用 `scripts` 文件夹中的脚本即可复现实验结果
