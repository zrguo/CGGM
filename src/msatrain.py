import torch
from torch import nn
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.msamodel import MSAModel, ClassifierGuided
from src.eval_metrics import eval_iemocap, eval_senti, cal_cos


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = MSAModel(hyp_params.output_dim, hyp_params.orig_dim, hyp_params.proj_dim,
                 hyp_params.num_heads, hyp_params.layers, hyp_params.relu_dropout,
                 hyp_params.embed_dropout, hyp_params.res_dropout, hyp_params.out_dropout,
                 hyp_params.attn_dropout)

    if hyp_params.modulation != 'none':
        classifier = ClassifierGuided(hyp_params.output_dim, hyp_params.num_mod, hyp_params.proj_dim, hyp_params.num_heads,
                                    hyp_params.cls_layers, hyp_params.relu_dropout, hyp_params.embed_dropout,
                                    hyp_params.res_dropout, hyp_params.attn_dropout)
        cls_optimizer = getattr(optim, hyp_params.optim)(classifier.parameters(), lr=hyp_params.cls_lr)
    else:
        classifier, cls_optimizer = None, None

    if hyp_params.use_cuda:
        model = model.cuda()
        if hyp_params.modulation != 'none':
            classifier = classifier.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
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

    def train(model, classifier, optimizer, cls_optimizer, criterion):
        nonlocal acc1, l_gm
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, batch in enumerate(train_loader):
            text, audio, vision, batch_Y = batch['text'], batch['audio'], batch['vision'], batch['labels']
            eval_attr = batch_Y.squeeze(-1)  # if num of labels is 1
            model.zero_grad()

            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemo':
                        eval_attr = eval_attr.long()

            batch_size = text.size(0)
            net = nn.DataParallel(model) if batch_size > 10 else model
            preds, hs = net([text, audio, vision])
            
            if hyp_params.dataset == 'iemo':
                preds = preds.view(-1, 4)
                eval_attr = eval_attr.view(-1)
   
            raw_loss = criterion(preds, eval_attr)
            if hyp_params.modulation == 'cggm' and l_gm is not None:
                raw_loss += hyp_params.lamda * l_gm
            raw_loss.backward()
            
            if hyp_params.modulation == 'cggm':
                cls_optimizer.zero_grad()
                net2 = nn.DataParallel(classifier) if batch_size > 10 else classifier
                cls_res = net2(hs)

                for name, para in net.named_parameters():
                    if 'out_layer.weight' in name:
                        fusion_grad = para

                if hyp_params.dataset == 'iemo':
                    for i in range(len(cls_res)):
                        cls_res[i] = cls_res[i].view(-1, 4)
                
                cls_loss = criterion(cls_res[0], eval_attr)
                for i in range(1, hyp_params.num_mod):
                    cls_loss += criterion(cls_res[i], eval_attr)

                cls_loss.backward()

                cls_grad = []
                for name, para in net2.named_parameters():
                    if 'out_layer.weight' in name:
                        cls_grad.append(para)
                
                llist = cal_cos(cls_grad, fusion_grad)
                
                acc2 = classifier.cal_coeff(hyp_params.dataset, eval_attr, cls_res)
                diff = [acc2[i] - acc1[i] for i in range(hyp_params.num_mod)]

                diff_sum = sum(diff) + 1e-8
                coeff = list()

                for d in diff:
                    coeff.append((diff_sum - d) / diff_sum)
                
                acc1 = acc2
                l_gm = np.sum(np.abs(coeff)) - (coeff[0] * llist[0] + coeff[1] * llist[1] + coeff[2] * llist[2])
                l_gm /= hyp_params.num_mod

                for i in range(hyp_params.num_mod):
                    for name, params in net.named_parameters():
                        if f'encoders.{i}' in name:
                            params.grad *= (coeff[i] * hyp_params.rou)
                
                cls_optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
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
                text, audio, vision, batch_Y = batch['text'], batch['audio'], batch['vision'], batch['labels']
                eval_attr = batch_Y.squeeze(dim=-1)  # if num of labels is 1

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemo':
                            eval_attr = eval_attr.long()

                net = model
                preds, _ = net([text, audio, vision])
                if hyp_params.dataset == 'iemo':
                    preds = preds.view(-1, 4)
                    eval_attr = eval_attr.view(-1)

                total_loss += criterion(preds, eval_attr).item()
                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_acc = 0
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train(model, classifier, optimizer, cls_optimizer, criterion)
        val_loss, val_res, val_truth = evaluate(model, criterion, test=False)

        if hyp_params.dataset == 'iemo':
            acc = eval_iemocap(val_res, val_truth)
        else:
            acc = eval_senti(val_res, val_truth)

        end = time.time()
        duration = end - start
        scheduler.step(val_loss)  # Decay learning rate by validation loss

        print("-" * 50)
        print(
            'Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f}'.format(epoch, duration, val_loss))
        print("-" * 50)

        if best_acc < acc:
            best_acc = acc

    print("Accuracy: ", best_acc)