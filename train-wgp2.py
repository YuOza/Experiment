import torch
from options import TrainOptions
from model_wgp_res0 import CDCGAN
from saver import *
import torchvision
import os
import torchvision.transforms as transforms

import datetime

import pathlib
from PIL import Image
import numpy as np

from fid_score import *
from torchvision import datasets

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transform):
        self.transform = transform
        path = pathlib.Path(img_path)
        files = list(path.glob('**/*.jpg')) + list(path.glob('**/*.png'))
        self.images = []
        i = 0
        for fn in files:
            self.images.append(np.array(Image.open(fn)))
            i += 1
            if i >= 5000:
                break
        # self.images = [Image.open(fn) for fn in files]
        self.length = len(self.images)
    def __len__(self):
        return self.length
    def __getitem__(self, item):
        image = self.images[item]
        label = 0
        image = Image.fromarray(image)
        out_image = self.transform(image)
        return out_image, label

def convergence_tf(loss_list, length, conv):
    if length < 5:
        return False
    cost = 0
    now_length = length-1
    for i in range(0,3):
        cost += abs(loss_list[now_length-i] - loss_list[now_length-i-1])
    if cost < conv*3:
        return True
    else:
        return False

def train_one_time(nz, real_z, n_ep, m_name, dataset_name, value, space):
    print(datetime.datetime.now())     
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    opts.nz = nz            # 次元数の最大値
    opts.real_z = real_z    # 初期潜在変数の次元数
    opts.n_ep = n_ep            # エポック
    opts.name = m_name  # 学習全体の結果フォルダ(画像群とモデル)

    opts.model_save_freq = 50  # モデル保存の頻度
    opts.result_dir = './models/DCGAN-Mode-Seeking' # 上のフォルダを保存する場所
    opts.dataroot = './dataset/' + dataset_name

    opts.batch_size = 64

    opts.class_num = 10
    os.makedirs(opts.dataroot, exist_ok=True)
    os.makedirs(opts.result_dir, exist_ok=True)
    # data loader
    print('\n--- load dataset ---')

    dsn = ""
    if dataset_name == 'CIFAR10':
        opts.img_size = 32
        dataset = torchvision.datasets.CIFAR10(opts.dataroot, train=True, download=True, transform= transforms.Compose([
            transforms.Resize(opts.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif dataset_name == 'STL10':
        opts.img_size = 96
        dataset = torchvision.datasets.STL10(opts.dataroot, split='train', download=True, transform=transforms.Compose([
            transforms.Resize(opts.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    elif dataset_name == 'tower':
        dsn = "tw"
        opts.img_size = 64
        dataset = ImageDataset('./tower', transform=transforms.Compose([
            transforms.Resize((opts.img_size,opts.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    elif dataset_name == 'bedroom':
        dsn = "bd"
        opts.img_size = 64
        dataset = ImageDataset('./bedroom', transform=transforms.Compose([
            transforms.Resize((opts.img_size,opts.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    elif dataset_name == 'Celeb':
        dsn = "ce"
        opts.img_size = 64
        dataset = ImageDataset('./celeb', transform=transforms.Compose([
            transforms.Resize((opts.img_size,opts.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = CDCGAN(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    ep0 += 1
    print('start the training at epoch %d'%(ep0))

    # saver for display and output
    saver = Saver(opts)

    # train
    print('\n--- train ---')

    generator_losses = []
    discriminator_losses = []
    convergence_score = []
    count = 0
    def_direction = -1
    now_direction = -1
    highest_z = 100
    highest_fid = 1000
    now_fid = 1000
    
    Gpath = "./learn/forISimages"
    Tpath = "./learn/forFIDTrue/"+dataset_name
    fid_path = [Tpath, Gpath]
    SavePath = "./learn/forISimages/IS"
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    log = open(m_name + '.txt', 'w')
    model.train()
    imgs = []
    names = []
    fid_list = []
    for ep in range(ep0, opts.n_ep):
        for it, (images, label) in enumerate(train_loader):
            if images.size(0) != opts.batch_size:
                continue
            # input data
            images = images.cuda(opts.gpu).detach()
            # update model
            model.update_D(images, label)
            if it % 5 == 4:
                model.update_G()
            total_it += 1
        # lossを記録
        generator_losses.append(model.save_G_loss())
        discriminator_losses.append(model.save_D_loss())

        if ep % space == 0 and ep != 0:
            #----FID calcurate----
            model.eval()
            total = 0
            imgs.clear()
            names.clear()
            for idx1 in range(10000):
                with torch.no_grad():
                    img = model.test_forward()
                imgs.append(img)
                names.append('img_' + str(total))
                total += 1
            save_imgs(imgs, names, SavePath)
            now_fid = calculate_fid_given_paths(fid_path, 50, True, 2048)
            #--------
            model.train()
            fid_list.append(now_fid)
            if highest_fid > now_fid: # 最良が更新された
                highest_z = model.real_z
                highest_fi = now_fid
            else: # 今回の変更では最良が更新されなかった
                
                if highest_z - model.real_z > 0: # 最良点に近い方向に増減する
                    def_direction = 1
                else:
                    def_direction = -1
                if now_direction != def_direction:
                    now_direction = def_direction
                    value -= 1
                    if value <= 0:
                        value = 1
            model.real_z += def_direction * value
            if model.real_z <= 0 or model.real_z >= model.nz:
                model.real_z -= def_direction * value

            # model.G.change_z_num(model.nz)
            # model.G.cuda(opts.gpu)  # これがないとエラー/モデルを変更するたびにGPUに送る必要がある
            log_str = "z changed to " + str(model.real_z) + " : epoch = "+ str(ep)
            print(log_str)
            log.write(log_str + '\n')
        # save result image
        saver.write_img(ep, model)
        # Save network weights
        if ep == opts.n_ep:
            break
        if ep == 199:
            model.save(m_name + '200'+'-' + str(model.real_z) + '-' + dsn + '.pth', 199, total_it)

    print('Generator Loss print')
    log.write('G loss \n')
    for i in generator_losses:
        log.write(str(i) + '\n')
    print('Discriminator loss print')
    log.write('D loss \n')
    for i in discriminator_losses:
        log.write(str(i) + '\n')
    print('FID print')
    log.write('FID \n')
    for i in islist:
        log.write(str(i) + '\n')
    print('Final z dimension = '+ str(model.real_z))
    log.write('Final z dimension = '+ str(model.real_z) + '\n')
    print(datetime.datetime.now())
    log.write(str(datetime.datetime.now()))
    log.close()

    model.save(m_name + '-' + str(model.real_z) + '-' + dsn + '.pth', 199, total_it)

    return

if __name__ == '__main__':
    train_one_time(nz=200, real_z=100, n_ep=400, m_name='WGPx_xx', dataset_name='Celeb', value=10, space=20)
# dataset name = Celeb, tower, bedroom
