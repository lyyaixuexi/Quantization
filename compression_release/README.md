# Step of SBS

1. Search configuration.

```bash
cd single_path_compress_binary_gate
sh train.sh
```

Remember to change the path of dataset and pretrained model.

2. Finetuning with the searched configuration

```bash
cd single_path_compress_binary_gate
sh finetune.sh
```