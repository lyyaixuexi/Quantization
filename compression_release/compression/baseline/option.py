import os
import shutil

from pyhocon import ConfigFactory

from compression.utils.utils import is_main_process


class Option(object):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)

        # ------------- general options ----------------------------------------
        self.save_path = self.conf["save_path"]  # log path
        self.data_path = self.conf["data_path"]  # path for loading data set
        self.dataset = self.conf["dataset"]  # options: imagenet | cifar10
        self.seed = self.conf["seed"]  # manually set RNG seed
        self.gpu = self.conf["gpu"]  # GPU id to use, e.g. "0,1,2,3"
        self.n_gpus = len(self.gpu.split(","))  # number of GPUs to use by default
        self.print_frequency = self.conf["print_frequency"]  # print frequency (default: 10)
        self.dist_url = "env://"

        # ------------- data options -------------------------------------------
        self.n_threads = self.conf["n_threads"]  # number of threads used for data loading
        self.n_classes = self.conf["n_classes"]  # number of classes in the dataset

        # ------------- common optimization options ----------------------------
        self.batch_size = self.conf["batch_size"]  # mini-batch size
        self.momentum = self.conf["momentum"]  # momentum
        self.weight_decay = self.conf["weight_decay"]  # weight decay
        self.lr = self.conf["lr"]  # initial learning rate
        self.n_epochs = self.conf["n_epochs"]  # number of total epochs
        self.step = self.conf["step"]  # multi-step for linear learning rate
        self.lr_scheduler_type = self.conf['lr_scheduler_type']
        self.opt_type = self.conf['opt_type']
        self.quantize_first_last = self.conf['quantize_first_last']
        self.warmup_n_epochs = self.conf['warmup_n_epochs']

        # ------------- model options ------------------------------------------
        self.net_type = self.conf["net_type"]  # options: resnet | preresnet | vgg
        self.experiment_id = self.conf["experiment_id"]  # identifier for experiment
        self.depth = self.conf["depth"]  # resnet depth: (n-2)%6==0

        # ------------- quantization options -----------------------------------
        self.qw = self.conf["qw"]
        self.qa = self.conf["qa"]

        # ---------- resume or pretrained options ---------------------------------
        # path to pretrained model, load model state_dict only
        self.pretrained = None if len(self.conf["pretrained"]) == 0 else self.conf["pretrained"]
        # path to directory containing checkpoint, load state_dicts of model and optimizer, as well as training epoch
        self.resume = None if len(self.conf["resume"]) == 0 else self.conf["resume"]

    def params_check(self):
        if self.dataset in ["cifar10"]:
            self.n_classes = 10
        elif self.dataset in ["cifar100"]:
            self.n_classes = 100
        elif self.dataset == "imagenet":
            self.n_classes = 1000

    def set_save_path(self):
        self.params_check()

        if self.net_type in ["preresnet", "resnet"]:
            self.save_path = (
                self.save_path
                + "log_{}{:d}_{}_w{}a{}_bs{:d}_e{:d}_lr{:.5f}_step{}_{}/".format(
                    self.net_type,
                    self.depth,
                    self.dataset,
                    self.qw,
                    self.qa,
                    self.batch_size,
                    self.n_epochs,
                    self.lr,
                    self.step,
                    self.experiment_id,
                )
            )
        else:
            self.save_path = (
                self.save_path
                + "log_{}_{}_w{}a{}_bs{:d}_e{:d}_lr{:.5f}_step{}_{}/".format(
                    self.net_type,
                    self.dataset,
                    self.qw,
                    self.qa,
                    self.batch_size,
                    self.n_epochs,
                    self.lr,
                    self.step,
                    self.experiment_id,
                )
            )

        if is_main_process():
            if os.path.exists(self.save_path):
                print("{} file exist!".format(self.save_path))
                action = input("Select Action: d (delete) / q (quit):").lower().strip()
                act = action
                if act == "d":
                    shutil.rmtree(self.save_path)
                else:
                    raise OSError("Directory {} exits!".format(self.save_path))

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

    def copy_code(self, logger, src=os.path.abspath("./"), dst="./code/"):
        """
        copy code in current path to a folder
        """

        if is_main_process():
            for f in os.listdir(src):
                if "specific_experiments" in f or "log" in f:
                    continue
                src_file = os.path.join(src, f)
                file_split = f.split(".")
                if len(file_split) >= 2 and file_split[1] == "py":
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    dst_file = os.path.join(dst, f)
                    try:
                        shutil.copyfile(src=src_file, dst=dst_file)
                    except:
                        logger.errro("copy file error! src: {}, dst: {}".format(src_file, dst_file))
                elif os.path.isdir(src_file):
                    deeper_dst = os.path.join(dst, f)
                    self.copy_code(logger, src=src_file, dst=deeper_dst)
