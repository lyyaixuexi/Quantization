from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import torch
import argparse
import os
import json
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import pseudo_quantize_model_weight, real_quantize_model_weight
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
import time
from awq.datautils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='path of the hf model')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
# model config
parser.add_argument('--parallel', action='store_true',
                    help="enable model parallelism")
parser.add_argument('--auto_parallel', action='store_true',
                    help="automatically set parallel and batch_size")
# quantization config
parser.add_argument('--w_bit', type=int, default=None)
parser.add_argument('--q_group_size', type=int, default=-1)
parser.add_argument('--no_zero_point', action='store_true',
                    help="disable zero_point")
parser.add_argument('--q_backend', type=str,
                    default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument('--dump_quant', type=str, default=None,
                    help='save quantized model')
parser.add_argument('--load_quant', type=str, default=None,
                    help='load quantized model')
# apply/save/load awq
parser.add_argument('--run_awq', action='store_true',
                    help="perform awq search process")
parser.add_argument('--dump_awq', type=str, default=None,
                    help="save the awq search results")
parser.add_argument('--load_awq', type=str, default=None,
                    help="load the awq search results")
parser.add_argument('--benchmark', type=int, default=0,
                    help='Number of tokens to use for benchmarking.')
parser.add_argument(
    '--dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
    help='Where to extract calibration data from.'
)
parser.add_argument(
    '--nsamples', type=int, default=128,
    help='Number of calibration data samples.'
)
parser.add_argument(
    '--seed',
    type=int, default=0, help='Seed for sampling the calibration data.'
)
parser.add_argument(
    '--model', type=str,
    help='OPT model to load; pass `facebook/opt-X`.'
)
parser.add_argument(
    '--check', action='store_true',
    help='Whether to compute perplexity during benchmarking for verification.'
)

args = parser.parse_args()

DEV = torch.device('cuda:0')

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization

}
print("Quantization config:", q_config)

# build model and tokenizer

def build_model_and_enc(model_path):
    if not os.path.exists(model_path):  # look into ssd
        raise FileNotFoundError(f"{model_path} not found!")
    print(f"* Building model {model_path}")

    # all hf model
    config = AutoConfig.from_pretrained(model_path)
    enc = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if args.load_quant:  # directly load quantized weights
        # no need to really load the fp16 weights... just to get the model structure
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
                                                         torch_dtype=torch.float16)
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True)
        model = load_checkpoint_and_dispatch(
            model, args.load_quant, device_map="balanced",
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer", "LlamaDecoderLayer"]
        )
    else:  # fp16 to quantized
        kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}

        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, **kwargs)

        if args.run_awq:
            awq_results = run_awq(
                model, enc,
                w_bit=args.w_bit, q_config=q_config,
                n_samples=128, seqlen=512,
            )
            if args.dump_awq:
                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(model, awq_results)

        # weight quantization
        if args.w_bit is not None:
            if args.q_backend == "fake":
                assert args.dump_quant is None, \
                    "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
            elif args.q_backend == "real":  # real quantization
                real_quantize_model_weight(
                    model, w_bit=args.w_bit, q_config=q_config
                )
                if args.dump_quant:
                    print(
                        f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

    return model, enc

def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model
def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()

    model = get_opt(args.model)

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    # a hack here to auto set model group
    model, enc = build_model_and_enc(args.model_path)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    # lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
    #
    # if args.tasks is not None:
    #     task_names = args.tasks.split(",")
    #
    #     results = evaluator.simple_evaluate(
    #         model=lm_eval_model,
    #         tasks=task_names,
    #         batch_size=args.batch_size,
    #         no_cache=True,
    #         num_fewshot=args.num_fewshot,
    #     )
    #
    #     print(evaluator.make_table(results))
    #
    #     if args.output_path is not None:
    #         os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    #         # otherwise cannot save
    #         results["config"]["model"] = args.model_path
    #         with open(args.output_path, "w") as f:
    #             json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
