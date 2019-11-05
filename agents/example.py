import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, random_split

from tensorboardX import SummaryWriter

from utils.misc import print_cuda_statistics, get_device
from agents.base import BaseAgent

# from datasets import custom_dataset
# from graphs.model import model
# from utils import utils

cudnn.benchmark = True


class ExampleAgent(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        print_cuda_statistics()
        self.device = get_device()

        # define models
        self.model

        # define data_loader
        train_dataset = custom_dataset(cfg.tr_im_pth, cfg.tr_gt_pth)
        test_dataset = custom_dataset(cfg.te_im_pth, cfg.te_gt_pth)
        
        # train_dataset, test_dataset = random_split(dataset, 
        #                                             [train_size, test_size])

        self.train_loader = DataLoader(train_dataset, batch_size=config.bs, 
                                  shuffle=False, num_workers=config.num_w)
        self.test_loader = DataLoader(test_dataset, batch_size=config.bs, 
                                  shuffle=False, num_workers=config.num_w)

        # define criterion
        self.criterion = Loss()

        # define optimizers for both generator and discriminator
        self.optimizer = None

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.cuda = (self.device == torch.device('cuda')) and self.cfg.cuda

        # set the manual seed for torch
        self.manual_seed = self.cfg.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            if self.config.data_parallel:
                self.model = nn.DataParallel(self.model)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
        else:
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the cfg if not found start from scratch.
        self.load_checkpoint(self.cfg.checkpoint_file)
        # Summary Writer
        self.summary_writer = None


    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            self.logger.info("Loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name, map_location=self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            info = "Checkpoint loaded successfully from "
            self.logger.info(info + "'{}' at (epoch {}) at (iteration {})\n"
              .format(file_name, checkpoint['epoch'], checkpoint['iteration']))
                
        except OSError as e:
            self.logger.info("Checkpoint not found in '{}'.".format(file_name))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current 
                        checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model' : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        # save the state
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        torch.save(state, os.path.join(checkpoint_dir, file_name))

        if is_best:
            shutil.copyfile(os.path.join(checkpoint_dir, file_name),
                            os.path.join(checkpoint_dir, 'best.pt'))

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        pass

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        pass

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        pass

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, 
        the operator and the data loader
        :return:
        """
        pass