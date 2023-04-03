import os

import torch
import torch.nn as nn

import compression.utils as utils
from compression.utils.utils import is_main_process
from compression.checkpoint import CheckPoint

class SuperCheckPoint(CheckPoint):
    """
    save model state to file
    check_point_params: model, optimizer, epoch
    """

    def __init__(self, save_path, logger):

        self.save_path = os.path.join(save_path, "check_point")
        self.check_point_params = {
            "model": None,
            "optimizer": None,
            "arch_optimizer": None,
            "lr_scheduler": None,
            "arch_lr_scheduler": None,
            "epoch": None,
        }
        self.logger = logger

        # make directory
        if is_main_process():
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

    def load_checkpoint(self, checkpoint_path):
        """
        load checkpoint file
        :params checkpoint_path: path to the checkpoint file
        :return: model_state_dict, optimizer_state_dict, epoch
        """
        if os.path.isfile(checkpoint_path):
            if self.logger:
                self.logger.info("|===>Load resume check-point from: {}".format(checkpoint_path))
            self.check_point_params = torch.load(checkpoint_path, map_location="cpu")
            model_state_dict = self.check_point_params["model"]
            optimizer_state_dict = self.check_point_params["optimizer"]
            arch_optimizer_state_dict = self.check_point_params["arch_optimizer"]
            lr_scheduler = self.check_point_params["lr_scheduler"]
            arch_lr_scheduler = self.check_point_params["arch_lr_scheduler"]
            epoch = self.check_point_params["epoch"]
            return model_state_dict, optimizer_state_dict, arch_optimizer_state_dict, epoch, lr_scheduler, arch_lr_scheduler
        else:
            assert False, "file not exits" + checkpoint_path

    def save_checkpoint(self, model, optimizer, arch_optimizer, lr_scheduler, arch_lr_scheduler, epoch, index=0):
        # get state_dict from model and optimizer
        model = utils.list2sequential(model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.state_dict()
        optimizer = optimizer.state_dict()
        arch_optimizer = arch_optimizer.state_dict()
        lr_scheduler = lr_scheduler.state_dict()
        arch_lr_scheduler = arch_lr_scheduler.state_dict()

        # save information to a dict
        self.check_point_params["model"] = model
        self.check_point_params["optimizer"] = optimizer
        self.check_point_params["arch_optimizer"] = arch_optimizer
        self.check_point_params["lr_scheduler"] = lr_scheduler
        self.check_point_params["arch_lr_scheduler"] = arch_lr_scheduler
        self.check_point_params["epoch"] = epoch

        # save to file
        torch.save(
            self.check_point_params,
            os.path.join(self.save_path, "checkpoint_{:0>3d}.pth".format(index)),
        )