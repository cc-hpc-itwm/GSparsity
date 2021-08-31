import numpy as np
import matplotlib.pyplot as plt
import sys
import logging
import torch
import torch.nn as nn
import models.resnet as resnet
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
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


def set_logger(logger_name, level=logging.INFO):
        """
        Method to return a custom logger with the given name and level
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        log_format = logging.Formatter("%(asctime)s %(message)s", '%Y-%m-%d %H:%M:%S')

        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        # Creating and adding the file handler
        file_handler = logging.FileHandler(logger_name, mode='w')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        return logger
   
    
def get_param_vec(model):
    #flatten parameters of the network into a numpy array
    rv = torch.zeros(1,0).cuda()
    for p in model.parameters():
        rv_tmp = p.data.view(1,-1)
        rv = torch.cat((rv, rv_tmp),1)
    return rv.view(-1).cpu().numpy()

def compute_cdf(model):
    #save sparsity values
    Num_pars = sum(p.shape[0] for p in model.parameters() if p.requires_grad)
    data = get_param_vec(model)
    values, base = np.histogram(data, bins=1000000)
    cumulative = np.cumsum(values)
    return base[:-1],cumulative/Num_pars

def get_filters_l1norm(model):
    #flatten parameters of the network into a numpy array
    rv = torch.zeros(1,0).cuda()
    total_filters=0
    for index,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            norms=torch.norm((module.weight.data.view(module.weight.data.shape[0],-1)),p=2,dim=1)
            total_filters += module.weight.data.shape[0]
            rv_tmp = norms.data.view(1,-1)
            rv = torch.cat((rv, rv_tmp),1)
    return rv.view(-1).cpu().numpy(),total_filters


def plot_learning_curve(model,data, xlabel, ylabel, filename, ylim=None, cdf_data=False):
    if cdf_data:
        fig, ax = plt.subplots()
        
        #generate a sparsity plot of a given network
        Num_pars = sum(p.shape[0] for p in model.parameters() if p.requires_grad)
        data, Num_pars= get_filters_l1norm(model)
        values, base = np.histogram(data, bins=1000000)
        cumulative = np.cumsum(values)
        plt.plot(base[:-1], cumulative/Num_pars)
    else:
        fig = plt.figure()
        plt.plot(data)            
        
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.yticks(np.arange(ylim[0], ylim[1],10))
        plt.ylim(ylim[0], ylim[1])
        
    if cdf_data:
        '''axins_lower = inset_axes(ax, 1.3,1.3 , loc=2,bbox_to_anchor=(0.6, 0.55),bbox_transform=ax.figure.transFigure) # no zoom
        axins_lower.plot(base[:-1], cumulative/Num_pars)
        axins_lower.set_xlim(-3e-1, 1e-1) # apply the x-limits
        axins_lower.set_ylim(0, 0.4)
        axins_lower.grid(True)
        mark_inset(ax, axins_lower, loc1=2, loc2=4, fc="none", ec="0.5")'''

        axins_upper = inset_axes(ax, 1.6, 1.6, loc=2, bbox_to_anchor=(0.5, 0.6), bbox_transform=ax.figure.transFigure) # no zoom
        axins_upper.plot(base[:-1], cumulative/Num_pars)
        axins_upper.set_xlim(-1e-1, 3e-1) # apply the x-limits
        axins_upper.set_ylim(0.6, 0.8)
        axins_upper.grid(True)
        #mark_inset(ax,axins_upper, loc1=2, loc2=4, fc="none", ec="0.5")
        
    plt.savefig(filename)
    plt.close('all')

def print_nonzeros(model):
    nonzero = total = 0
    
    for name,param in model.named_parameters():
        if "weight" in name and "conv" in name:
            nz_count=param.nonzero().size(0)
            total_params = param.numel()
            nonzero += nz_count
            total += total_params
            #print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'Active: {nonzero}, Pruned : {total - nonzero}, Total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    compression_rate= round(total/nonzero,2)
    percentage_pruned= round(100*(total-nonzero) / total,2)
    return str(compression_rate)+'x'+'('+str(percentage_pruned)+'% pruned)' ,percentage_pruned

def print_nonzeros_filters(model):
    nonzero = total = 0
    for name,param in model.named_parameters():
        if "weight" in name and "conv" in name:
            norms= torch.norm(param.view(param.shape[0],-1),p=2,dim=1)
            nz_count=norms.nonzero().size(0)
            total_params = param.shape[0]
            nonzero += nz_count
            total += total_params
            #print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'Active Filters: {nonzero}, Pruned Filters : {total - nonzero}, Total Filters: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    compression_rate= round(total/nonzero,2)
    percentage_pruned= round(100*(total-nonzero) / total,2)
    return str(compression_rate)+'x'+'('+str(percentage_pruned)+'% pruned)' ,percentage_pruned

def acc_n_loss(train_loss, test_top1, filename, train_top1=None, test_loss=None):
    if train_top1 is not None and test_loss is not None:
        fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2))
        fig.suptitle('Loss and Acc',y=1.0)
        axs[0,0].semilogy(train_loss)
        axs[0,0].grid(True)
#         axs[0,0].title.set_text('Loss')
        axs[0,0].set_xlabel('Epochs')
        axs[0,0].set_ylabel('Training loss')
#         axs[0,0].set_xticks(np.arange(0, len(train_loss)+1, 100))

        axs[0,1].plot(train_top1)
        axs[0,1].grid(True)
        axs[0,1].set_ylim(0,101)
        axs[0,1].set_yticks(np.arange(0, 101, 5))
        axs[0,1].set_xlabel('Epochs')
        axs[0,1].set_ylabel('Train accuracy (in %)')
#         axs[0,1].set_xticks(np.arange(0, len(train_top1)+1, 100))
#         axs[0,1].title.set_text('Accuracy')

        axs[1,0].semilogy(test_loss)
        axs[1,0].grid(True)
        axs[1,0].set_xlabel('Epochs')
        axs[1,0].set_ylabel('Test loss')
#         axs[1,0].set_xticks(np.arange(0, len(test_top1)+1, 100))
#         axs[1,0].title.set_text('Accuracy')

        axs[1,1].plot(test_top1)
        axs[1,1].grid(True)
        axs[1,1].set_ylim(0,101)
        axs[1,1].set_yticks(np.arange(0, 101, 5))
        axs[1,1].set_xlabel('Epochs')
        axs[1,1].set_ylabel('Test accuracy (in %)')
#         axs[1,1].set_xticks(np.arange(0, len(test_top1)+1, 100))
#         axs[1,1].title.set_text('Accuracy')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)       
        
    elif train_top1 is not None and test_loss is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.6, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax1.grid(True)
#         ax1.title.set_text('Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training loss')
#         ax1.set_xticks(np.arange(0, len(train_loss)+1, 100))

        ax2.plot(train_top1)
        ax2.grid(True)
        ax2.set_ylim(0,101)
        ax2.set_yticks(np.arange(0, 101, 5))
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Train accuracy (in %)')
#         ax2.set_xticks(np.arange(0, len(train_top1)+1, 100))
#         ax2.title.set_text('Accuracy')

        ax3.plot(test_top1)
        ax3.grid(True)
        ax3.set_ylim(0,101)
        ax3.set_yticks(np.arange(0, 101, 5))
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Test accuracy (in %)')
#         ax3.set_xticks(np.arange(0, len(test_top1)+1, 100))
#         ax3.title.set_text('Accuracy')

        fig.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
    elif train_top1 is None and test_loss is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.8))
        fig.suptitle('Loss and Acc')
        ax1.semilogy(train_loss)
        ax2.plot(test_top1)
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


def save_checkpoint(epoch, epochs_since_improvement, args,  model, optimizer, is_best, current_loss, current_acc1, train_dir, original_model_acc1, pruned_model_acc1, pruned_threshold, run_id):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param model: model
    :param optimizer: optimizer to update weights
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'state_dict': model.state_dict(),
            'optimizer': optimizer,
            'error' : current_loss,
            'current_acc1': current_acc1,
            'original_model_acc1': original_model_acc1,
            'pruned_model_acc1': pruned_model_acc1, 
             'pruned_threshold': pruned_threshold,
             'run_id' : run_id
            }
    
    torch.save(state, train_dir+args.filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, train_dir+'BEST_' + args.filename)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

if __name__ == '__main__':
    model=resnet.resnet56().cuda()
    trained_model_path=trained_model_path = '/home/shalinis/sparse_nas_channel_pruning/sparse_nas/filter_pruning/save_filter_pruning_cutout/experiment_resnet_CIFAR10_lr_0.0007701_momentum_0.5762_weight_decay_0.02/train/BEST_checkpoint.pth.tar'
    checkpoint = torch.load(trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    
    plot_learning_curve(model,data=model, xlabel="L2 Norm Value of Filters", ylabel="CDF", filename="filter_cdf_plot.png", cdf_data=True)
    