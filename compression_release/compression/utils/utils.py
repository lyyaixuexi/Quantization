from __future__ import print_function

import datetime
import errno
import os
import time
import math
from collections import defaultdict, deque
from torch.autograd.function import Function

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, logger, delimiter="\t"):
        self.logger = logger
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("{} Total time: {}".format(header, total_time_str))


def compute_norm(filter_weight, group_size):
    if filter_weight.ndim == 4:
        n, c, h, w = filter_weight.shape
        filter_weight = filter_weight.reshape(n // group_size, group_size * c * h * w)
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        # l2-norm
        # normalized_filter_weight_norm = (normalized_filter_weight * normalized_filter_weight).sum(1) / (self.group_size * c * h * w)
        # l1-norm
        normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (group_size * c * h * w)
        # normalized_filter_weight_norm = (normalized_filter_weight.abs().sum(1) / (self.group_size * c * h * w)).detach()
    else:
        n, c = filter_weight.shape
        filter_weight = filter_weight.reshape(n // group_size, group_size * c)
        normalized_filter_weight = filter_weight / torch.max(torch.abs(filter_weight))
        normalized_filter_weight_norm = normalized_filter_weight.abs().sum(1) / (group_size * c)
    return normalized_filter_weight_norm


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_logger_for_distributed(is_master, logger):
    """
    This function disables printing when not in master process
    """

    logger_info = logger.info

    def info(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            logger_info(*args, **kwargs)

    logger.info = info


def setup_tensorboard_logger_for_distributed(is_master, tensorboard_logger):
    """
    This function disables printing when not in master process
    """

    tensorboard_logger_add_scalar = tensorboard_logger.add_scalar
    tensorboard_logger_add_histogram = tensorboard_logger.add_histogram

    def add_scalar(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            tensorboard_logger_add_scalar(*args, **kwargs)

    def add_histogram(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            tensorboard_logger_add_histogram(*args, **kwargs)

    tensorboard_logger.add_scalar = add_scalar
    tensorboard_logger.add_histogram = add_histogram


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.rank = 0
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    setup_for_distributed(args.rank == 0)


# def compute_bops(kernel_size, in_channels, out_channels, h, w, bits_w=32, bits_a=32):
#     nk_square = in_channels * kernel_size * kernel_size
#     log_nk_square = torch.log2(nk_square) if isinstance(nk_square, torch.Tensor) else math.log(nk_square, 2)
#     bop = (
#         out_channels
#         * nk_square
#         * (
#             bits_w * bits_a
#             + bits_w
#             + bits_a
#             + log_nk_square
#         )
#         * h
#         * w
#     )
#     return bop
def compute_bops(kernel_size, in_channels, filter_per_channel, h, w, bits_w=32, bits_a=32):
    conv_per_position_flops = kernel_size * kernel_size * in_channels * filter_per_channel
    active_elements_count = h * w
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bops = overall_conv_flops * bits_w * bits_a
    return bops


def compute_memory_footprint(n, c, h, w, bitwidth=32):
    return n * c * h * w * bitwidth
    

def compute_param(layer):
    total_param = 0
    for param in layer.parameters():
        total_param += param.numel()
    return total_param


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class Dict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

