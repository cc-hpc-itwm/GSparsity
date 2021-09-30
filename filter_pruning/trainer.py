import torch
import torch.nn as nn
from torch.autograd import Variable
import utils
import time
from utils import  compute_cdf, print_nonzeros, AverageMeter, accuracy, print_nonzeros_filters


def train(train_loader, model, criterion, optimizer, epoch, weight_reg, logger, args, retrain):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    lossvals = []
    l1loss = []
    
    if retrain == True:
        print('\nIn Retraining...')
    else:
        print('\nIn Training...')
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        #Save the loss value after each iteration into this array
        lossvals.append(loss.item())

        #Accumulate the L1 loss across all layers and save it into the l1loss array
        laccum = 0
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1 = torch.sum(torch.abs(param))
                if weight_reg is not None:
                    loss = loss + (weight_reg * l1)
                laccum += l1.item()
        l1loss.append(laccum)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if retrain ==  True:
            for index,module in enumerate(model.modules()):
                if isinstance(module, nn.Conv2d):
                    '''weight_copy = module.weight.data.abs().clone()
                    mask = weight_copy.gt(0).float().cuda()
                    module.weight.grad.data.mul_(mask)'''
                    
                    weight_copy = module.weight.data.abs().clone()
                    channel_norms = torch.norm((weight_copy.view(weight_copy.shape[0],-1)),p=2,dim=1)
                    channel_mask = channel_norms.gt(0).float().cuda()
                    channel_mask = channel_mask.reshape(channel_mask.shape[0],1)
                    module.weight.grad.data = torch.einsum('ijkl, im -> ijkl', module.weight.grad.data, channel_mask)
                    
        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))

    return top1.avg, losses.avg, lossvals, l1loss

def validate(val_loader, model, criterion, logger, args):
    """
    Run evaluation
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # start time
    
    # switch to evaluate mode
    model.eval()
    end = time.time()
    torch.cuda.synchronize()
    since = int(round(time.time()*1000))
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            torch.cuda.synchronize()
            
            # compute output
            output = model(input_var)
            torch.cuda.synchronize()
            
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))
        torch.cuda.synchronize()
        time_elapsed = int(round(time.time()*1000)) - since

    logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    logger.info('Inference time elapsed {}ms'.format(time_elapsed/len(val_loader)))
    
    filter_compression_rate, filter_percentage_pruned=print_nonzeros_filters(model)
    weight_compression_rate, weight_percentage_pruned=print_nonzeros(model)

    return  losses.avg, top1.avg, filter_compression_rate,weight_compression_rate, weight_percentage_pruned