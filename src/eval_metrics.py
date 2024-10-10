import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc = accuracy_score(binary_truth, binary_preds)
    return acc


def train_eval_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc = accuracy_score(binary_truth, binary_preds)
    return acc


def train_eval_iemo(results, truths):
    test_preds = results.view(-1, 4).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    acc = accuracy_score(test_truth_i, test_preds_i)
    return acc


def train_eval_food(results, truths):
    test_preds = results.view(-1, 101).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    acc = accuracy_score(test_truth_i, test_preds_i)
    return acc

def eval_food(results, truths):
    test_preds = results.view(-1, 101).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)
    return acc


def eval_iemocap(results, truths):
    test_preds = results.view(-1, 4).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_i = np.argmax(test_preds, axis=1)
    test_truth_i = test_truth
    f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
    acc = accuracy_score(test_truth_i, test_preds_i)
    return acc


class SegLoss(nn.Module):
    def __init__(self, n_classes=4, weight=None, alpha=0.5):
        # dice_loss_plus_cetr_weighted
        super(SegLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight.cuda()
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        smooth = 0.01  
        input1 = F.softmax(input, dim=1)
        target1 = F.one_hot(target, self.n_classes)
        input1 = rearrange(input1, 'b n h w -> b n (h w)')
        target1 = rearrange(target1, 'b h w n -> b n (h w)')
        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()
        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input, target, weight=self.weight)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


def Dice(output, target, eps=1e-3):
    inter = torch.sum(output * target,dim=(1,2)) + eps
    union = torch.sum(output,dim=(1,2)) + torch.sum(target,dim=(1,2)) + eps * 2
    x = 2 * inter / union
    dice = torch.mean(x)
    return dice


def train_eval_seg(output, target):
    output = torch.argmax(output, dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())
    return (dice1 + dice2 + dice3) / 3


def cal_dice(output, target):
    output = torch.argmax(output, dim=1)
    dice1 = Dice((output == 3).float(), (target == 3).float())
    dice2 = Dice(((output == 1) | (output == 3)).float(), ((target == 1) | (target == 3)).float())
    dice3 = Dice((output != 0).float(), (target != 0).float())
    print(f'ET: {dice1.item()}, TC: {dice2.item()}, WT: {dice3.item()}.')

    return dice1, dice2, dice3


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0.):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def cal_cos(cls_grad, fusion_grad):
    fgn = fusion_grad.clone().view(-1)
    loss = list()
    for i in range(len(cls_grad)):
        tmp = cls_grad[i].clone().view(-1)
        l = F.cosine_similarity(tmp, fgn, dim=0)
        loss.append(l)

    return loss

