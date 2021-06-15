import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import logging
import sys
import matplotlib.pyplot as plt

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_parameters(model):
    return np.sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name and v.requires_grad)/1e6

def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.makedirs(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def set_logger(logger_name, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    log_format = logging.Formatter("%(asctime)s %(message)s", '%Y-%m-%d %H:%M')

    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger

def plot_individual_op_norm(model, filename, normalization="none", normalization_exponent=0):
    """plot the norm of each operation in the given model"""
    
    op_norms = [torch.norm(p) for p in model.parameters() if p.requires_grad]
    op_names = [q for q, p in model.named_parameters() if p.requires_grad]
    op_sizes = [p.numel() for p in model.parameters() if p.requires_grad]
        
    num_ops = len(op_names)
    f = plt.figure(num=None, figsize=(num_ops*0.15, 6), dpi=100, facecolor='w', edgecolor='k')
    if normalization == "none":
        for i, op_norm in enumerate(op_norms):
            plt.semilogy(i, op_norm.item(),"o")
    elif normalization == "mul":
        for i, (op_norm, op_size) in enumerate(zip(op_norms, op_sizes)):
            op_norm_normalized = op_norm * (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(),"o")
    elif normalization == "div":
        for i, (op_norm, op_size) in enumerate(zip(op_norms, op_sizes)):
            op_norm_normalized = op_norm / (op_size ** normalization_exponent)
            plt.semilogy(i, op_norm_normalized.item(),"o")

    plt.xticks(np.arange(num_ops), op_names, rotation=90)
    plt.xlim(-1, num_ops)    
    plt.ylim(1e-5, 1e5)    
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def acc_n_loss(train_loss, test_acc, filename, train_acc=None, test_loss=None, train_loss_reg=None):
    if train_acc is not None and test_loss is not None:
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
        fig.suptitle('Loss and Acc')
        axs[0,0].semilogy(train_loss, label='loss')

        if train_loss_reg is not None:
            axs[0,0].semilogy(train_loss_reg, label='loss+reg')
            
        axs[0,0].grid(True)
        axs[0,0].set_xlabel('Epochs')
        axs[0,0].set_ylabel('Training loss')

        axs[0,1].plot(train_acc)
        axs[0,1].grid(True)
        axs[0,1].set_ylim(0,101)
        axs[0,1].set_yticks(np.arange(0, 101, 5))
        axs[0,1].set_xlabel('Epochs')
        axs[0,1].set_ylabel('Train accuracy (in %)')

        axs[1,0].semilogy(test_loss)
        axs[1,0].grid(True)
        axs[1,0].set_xlabel('Epochs')
        axs[1,0].set_ylabel('Test loss')

        axs[1,1].plot(test_acc)
        axs[1,1].grid(True)
        axs[1,1].set_ylim(0,101)
        axs[1,1].set_yticks(np.arange(0, 101, 5))
        axs[1,1].set_xlabel('Epochs')
        axs[1,1].set_ylabel('Test accuracy (in %)')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)       
        
    elif train_acc is not None and test_loss is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax1.grid(True)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')

        ax2.plot(train_acc)
        ax2.grid(True)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Train accuracy (in %)')

        ax3.plot(test_acc)
        ax3.grid(True)
        ax3.set_ylim(0,101)
        ax3.set_yticks(np.arange(0, 101, 5))
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Test accuracy (in %)')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
    elif train_acc is None and test_loss is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax2.plot(test_acc)
        ax1.grid(True)
        ax2.grid(True)
        #ax1.set_ylim(bottom=0)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax1.title.set_text('Loss')
        ax1.set_xlabel('Epochs')
        ax2.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')
        ax2.set_ylabel('Test accuracy (in %)')
        ax2.title.set_text('Accuracy')
        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
