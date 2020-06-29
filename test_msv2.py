import torch
from options import TestOptions
from model_wgp_res0 import CDCGAN
from saver import *
import os
import torchvision
import torchvision.transforms as transforms
import datetime
import pathlib
import re

def main(n_ep, img_size):
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    opts.img_size = img_size
    opts.n_ep = n_ep

    path = pathlib.Path('/models')
    models = list(path.glob('*.pth'))

    for model_path in models:
        opts.resume = str(model_path)
        m = re.search(r'-(\d)+-', str(model_path))
        opts.real_z = int(str(model_path)[m.start()+1 : m.end()-1])
        model = CDCGAN(opts)
        model.eval()
        model.setgpu(opts.gpu)
        model.resume(opts.resume, train=False)

        m = re.search(r'-(\w)+\.', str(model_path))
        dataset = str(model_path)[m.start()+1 : m.end()-1]
        if dataset == 'tw':
            dataset = 'tower'
        elif dataset == 'bd':
            dataset = 'bedroom'
        
        m = re.search(r'\.', str(model_path))
        model_name = str(model_path)[36 : m.end()-1]
        print(model_name)

        result_dir = '/generated_image/LSUN_' + dataset + '/' + model_name
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        imgs = []
        names = []
        total = 0
        for idx1 in range(1000):
            for idx2 in range(10):
                with torch.no_grad():
                    img = model.test_forward()
                imgs.append(img)
                names.append('img_'+ str(total))
                save_imgs(imgs, names, result_dir)
                total += 1

    print(datetime.datetime.now())
    return


if __name__ == '__main__':
    main(n_ep=200,img_size=64)
