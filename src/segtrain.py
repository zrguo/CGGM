import torch
from torch import nn
import torch.optim as optim
import time
from models.segmodel import DeepLabMultiInput, SegClassifier
from src.eval_metrics import *


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = DeepLabMultiInput(output_stride=16, num_classes=4)

    if hyp_params.modulation != 'none':
        classifier = SegClassifier(num_classes=4)
    else:
        classifier = None
        cls_optimizer = None

    if hyp_params.use_cuda:
        model = model.cuda()
        if hyp_params.modulation != 'none':
            classifier = classifier.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=hyp_params.weight_decay, momentum=hyp_params.momentum)
    if hyp_params.modulation != 'none':
        cls_optimizer = optim.SGD(classifier.parameters(), lr=hyp_params.cls_lr, weight_decay=hyp_params.weight_decay, momentum=hyp_params.momentum)
    criterion = SegLoss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).cuda()
    scheduler = cosine_scheduler(base_value=hyp_params.base_lr, final_value=hyp_params.final_lr, epochs=hyp_params.num_epochs,
                                 niter_per_ep=len(train_loader), warmup_epochs=hyp_params.warmup_epochs, start_warmup_value=hyp_params.start_warmup_value)
    settings = {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'scheduler': scheduler,
                'classifier': classifier, 'cls_optimizer': cls_optimizer}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    classifier = settings['classifier']
    cls_optimizer = settings['cls_optimizer']
    acc1 = [0] * hyp_params.num_mod
    l_gm = None

    def train(model, classifier, optimizer, cls_optimizer, criterion, epoch):
        nonlocal acc1, l_gm
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, batch in enumerate(train_loader):
            it = len(train_loader) * (epoch-1) + i_batch
            param_group = optimizer.param_groups[0]
            param_group['lr'] = scheduler[it]

            flair, t1ce, t1, t2, batch_Y = batch
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    flair, t1ce, t1, t2, batch_Y = flair.cuda(), t1ce.cuda(), t1.cuda(), t2.cuda(), batch_Y.cuda()

            batch_size = flair.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, hf, lf = net(flair, t1ce, t1, t2)
            raw_loss = criterion(preds, batch_Y)
            if hyp_params.modulation == 'cggm' and l_gm is not None:
                raw_loss += hyp_params.lamda * l_gm

            raw_loss.backward()

            if hyp_params.modulation == 'cggm':
                cls_optimizer.zero_grad()
                net2 = nn.DataParallel(classifier) if batch_size > 10 else classifier
                cls_res = net2(hf, lf)
                cls_loss = criterion(cls_res[0], batch_Y)

                for name, para in net.named_parameters():
                    if 'decoder.last_conv.7.weight' in name:
                        fusion_grad = para

                for i in range(1, hyp_params.num_mod):
                    cls_loss += criterion(cls_res[i], batch_Y)
                cls_loss.backward()

                cls_grad = []
                for name, para in net2.named_parameters():
                    if 'last_conv.7.weight' in name:
                        cls_grad.append(para)
                
                llist = cal_cos(cls_grad, fusion_grad)

                acc2 = classifier.cal_coeff(cls_res, batch_Y)
                diff = [acc2[i] - acc1[i] for i in range(hyp_params.num_mod)]

                diff_sum = sum(diff) + 1e-8
                coeff = list()

                for d in diff:
                    coeff.append((diff_sum - d) / diff_sum)
                acc1 = acc2
                
                l_gm = np.sum(np.abs(coeff)) - (coeff[0] * llist[0] + coeff[1] * llist[1] + coeff[2] * llist[2] + coeff[3] * llist[3])
                l_gm /= hyp_params.num_mod

                for name, params in net.named_parameters():
                    if 'vision_encoder' in name:
                        params.grad *= (coeff[0] * hyp_params.rou)
                    if 'text_encoder' in name:
                        params.grad *= (coeff[1] * hyp_params.rou)
                
                cls_optimizer.step()

            optimizer.step()            

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += raw_loss.item() * batch_size

        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch in enumerate(loader):
                flair, t1ce, t1, t2, batch_Y = batch

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        flair, t1ce, t1, t2, batch_Y = flair.cuda(), t1ce.cuda(), t1.cuda(), t2.cuda(), batch_Y.cuda()

                net = model
                preds, _, _ = net(flair, t1ce, t1, t2)

                total_loss += criterion(preds, batch_Y).item()

                # Collect the results into dictionary
                results.append(preds)
                truths.append(batch_Y)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_dice = 0
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, classifier, optimizer, cls_optimizer, criterion, epoch)
        val_loss, r, t = evaluate(model, criterion, test=False)
        d1, d2, d3 = cal_dice(r, t)
        dice = (d1 + d2 + d3) / 3

        end = time.time()
        duration = end - start

        print("-" * 50)
        print(
            'Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}'.format(epoch, duration, val_loss))
        print("-" * 50)

        if dice < best_dice:
            best_dice = dice
    
    print('Dice: ', best_dice)