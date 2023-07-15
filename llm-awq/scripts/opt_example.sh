MODEL=a45aa65bbeb77c1558bc99bedc6779195462dab0

# run AWQ search (optional; we provided the pre-computed results)
# python -m awq.entry --model_path /mnt/cephfs/home/lyy/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/$MODEL \
#     --w_bit 4 --q_group_size 128 \
#     --run_awq --dump_awq awq_cache/$MODEL-w4-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path /mnt/cephfs/home/lyy/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0 \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq /mnt/cephfs/home/lyy/Quantization/llm-awq/awq_cache/opt-6.7b-w4-g128.pt \
    --q_backend fake

# generate real quantized weights (w4)
CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path /mnt/cephfs/home/lyy/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0 \
    --w_bit 4 --q_group_size 128 \
    --load_awq /mnt/cephfs/home/lyy/Quantization/llm-awq/awq_cache/opt-6.7b-w4-g128.pt \
    --q_backend real --dump_quant /mnt/cephfs/home/lyy/Quantization/llm-awq/quant_cache/opt-6.7b-w4-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
CUDA_VISIBLE_DEVICES=5 python -m awq.entry --model_path /mnt/cephfs/home/lyy/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0 \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant /mnt/cephfs/home/lyy/Quantization/llm-awq/quant_cache/opt-6.7b-w4-g128-awq.pt

CUDA_VISIBLE_DEVICES=0 python -m awq.entry --model_path /mnt/cephfs/home/lyy/.cache/huggingface/hub/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0 \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant /mnt/cephfs/home/lyy/Quantization/llm-awq/model/OPT/opt-6.7b.pt