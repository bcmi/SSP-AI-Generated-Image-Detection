import os
import torch
from utils.util import set_random_seed, poly_lr
from utils.tdataloader import get_loader, get_val_loader
from options import TrainOptions
from networks.ssp import ssp
from utils.loss import bceLoss
from datetime import datetime
import numpy as np
"""Currently assumes jpg_prob, blur_prob 0 or 1"""
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.isVal = True
    # blur
    val_opt.blur_prob = 0
    val_opt.blur_sig = [1]
    # jpg
    val_opt.jpg_prob = 0
    val_opt.jpg_method = ['pil']
    val_opt.jpg_qual = [90]
    # if len(val_opt.blur_sig) == 2:
    #     b_sig = val_opt.blur_sig
    #     val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    # if len(val_opt.jpg_qual) != 1:
    #     j_qual = val_opt.jpg_qual
    #     val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


def train(train_loader, model, optimizer, epoch, save_path):
    model.train()
    global step
    epoch_step = 0
    loss_all = 0
    try:
        for i, (images, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            images = images.cuda()
            preds = model(images).ravel()
            labels = labels.cuda()
            loss1 = bceLoss()
            loss = loss1(preds, labels)
            loss.backward()
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 200 == 0 or i == total_step or i == 1:
                print(
                    f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], Total_loss: {loss.data:.4f}')
        loss_all /= epoch_step
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path +
                       f'Net_epoch_{epoch}.pth')

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')


def val(val_loader, model, epoch, save_path):
    model.eval()
    global best_epoch, best_accu
    total_right_image = total_image = 0
    with torch.no_grad():
        for loader in val_loader:
            right_ai_image = right_nature_image = 0
            name, val_ai_loader, ai_size, val_nature_loader, nature_size = loader['name'], loader[
                'val_ai_loader'], loader['ai_size'], loader['val_nature_loader'], loader['nature_size']
            print("val on:", name)
            # for images, labels in tqdm(val_ai_loader, desc='val_ai'):
            for images, labels in val_ai_loader:
                images = images.cuda()
                labels = labels.cuda()
                res = model(images)
                res = torch.sigmoid(res).ravel()
                right_ai_image += (((res > 0.5) & (labels == 1))
                                   | ((res < 0.5) & (labels == 0))).sum()

            print(f'ai accu: {right_ai_image/ai_size}')
            # for images,labels in tqdm(val_nature_loader,desc='val_nature'):
            for images, labels in val_nature_loader:
                images = images.cuda()
                labels = labels.cuda()
                res = model(images)
                res = torch.sigmoid(res).ravel()
                right_nature_image += (((res > 0.5) & (labels == 1))
                                       | ((res < 0.5) & (labels == 0))).sum()
            print(f'nature accu:{right_nature_image/nature_size}')
            accu = (right_ai_image + right_nature_image) / \
                (ai_size + nature_size)
            total_right_image += right_ai_image + right_nature_image
            total_image += ai_size + nature_size
            print(f'val on:{name}, Epoch:{epoch}, Accuracy:{accu}')
    total_accu = total_right_image / total_image
    if epoch == 1:
        best_accu = total_accu
        best_epoch = 1
        torch.save(model.state_dict(), save_path +
                   'Net_epoch_best.pth')
        print(f'Save state_dict successfully! Best epoch:{epoch}.')
    else:
        if total_accu > best_accu:
            best_accu = total_accu
            best_epoch = epoch
            torch.save(model.state_dict(), save_path +
                       'Net_epoch_best.pth')
            print(f'Save state_dict successfully! Best epoch:{epoch}.')
    print(
        f'Epoch:{epoch},Accuracy:{total_accu}, bestEpoch:{best_epoch}, bestAccu:{best_accu}')


if __name__ == '__main__':
    set_random_seed()
    # train and val options
    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    # load data
    print('load data...')
    train_loader = get_loader(opt)
    total_step = len(train_loader)
    val_loader = get_val_loader(val_opt)

    # cuda config
    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    # load model
    model = ssp().cuda()
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from', opt.load)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    step = 0
    best_epoch = 0
    best_accu = 0
    print("Start train")
    for epoch in range(1, opt.epoch + 1):
        cur_lr = poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        val(val_loader, model, epoch, save_path)
