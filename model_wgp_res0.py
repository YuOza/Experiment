import networks_wgp_res as networks
import torch
import torch.nn as nn

from torch import autograd

class CDCGAN(nn.Module):
    def __init__(self, opts):
        super(CDCGAN, self).__init__()
        # parameters
        lr = 0.00002

        self.nz = opts.nz
        self.real_z = opts.real_z
        self.class_num = opts.class_num
        self.G = networks.generator(opts)
        self.D = networks.discriminator(opts)

        self.gen_opt = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        self.dis_opt = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)

        self.BCE_loss = torch.nn.BCELoss()


    def initialize(self):
        self.G.weight_init()
        self.D.weight_init()

    def setgpu(self, gpu):
        self.gpu = gpu
        self.D.cuda(self.gpu)
        self.G.cuda(self.gpu)

    def get_z_random(self, batchSize, nz):
        z = torch.cuda.FloatTensor(batchSize, nz)
        z.copy_(torch.randn(batchSize, nz))
        # 次元数変更部分
        work = z.to("cpu").numpy()
        work[:, self.real_z:self.nz] = [0 for i in range(self.nz - self.real_z)]
        z = torch.from_numpy(work)
        z = z.cuda()
        return z

    def onehot_encoding(self, label):
        onehot = torch.zeros(self.class_num, self.class_num)
        index = torch.zeros([self.class_num, 1], dtype= torch.int64)
        for i in range(self.class_num):
            index[i] = i

        onehot = onehot.scatter_(1, index, 1).view(self.class_num, self.class_num, 1, 1)
        label_one_hot = onehot[label]

        # [32, 10, 1, 1]のテンソルを返却 (バッチサイズ * CHWの形式)
        return label_one_hot.cuda(self.gpu).detach()

    def forward(self):
        self.z_random1 = self.get_z_random(self.real_image1.size(0), self.nz)
        self.z_random2 = self.get_z_random(self.real_image2.size(0), self.nz)
        z_conc = torch.cat((self.z_random1, self.z_random2), 0)

        self.fake_image=self.G.forward(z_conc)
        self.fake_image1, self.fake_image2 = torch.split(self.fake_image, self.z_random1.size(0), dim=0)

    def gradient_penalty(self, real, fake):
        self.device = torch.device('cuda')

        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=self.device)

        interpolates = epsilon * real + ((1 - epsilon) * fake)
        interpolates = interpolates.clone().detach().requires_grad_(True)
        gradients = autograd.grad(self.D.forward(interpolates),
                                    interpolates,
                                    grad_outputs=torch.ones(batch_size, device=self.device),
                                    create_graph=True)[0]

        gradients_norm = torch.sqrt(torch.sum(gradients.view(batch_size, -1)**2, dim=1) + 1e-12)
        return ((gradients_norm - 1)**2).mean()

    def update_D(self, image, label):
        self.real_image = image
        self.real_image1, self.real_image2 = torch.split(self.real_image, int(self.real_image.size(0)/2), dim=0)
        self.forward()

        self.dis_opt.zero_grad()
        self.loss_D = self.backward_D(self.D, self.real_image1, self.fake_image1)+ \
                      self.backward_D(self.D, self.real_image2, self.fake_image2)
        gp = self.gradient_penalty(self.real_image1, self.fake_image1)+ \
             self.gradient_penalty(self.real_image2, self.fake_image2)
        self.loss_D_gp = self.loss_D + (gp * 10)
        self.loss_D_gp.backward()
        self.dis_opt.step()

    def save_D_loss(self):
        return self.loss_D.clone().item()

    def save_G_loss(self):
        return self.loss_G_GAN.clone().item()

    def update_G(self):
        self.gen_opt.zero_grad()
        self.loss_G_GAN = self.backward_G(self.D, self.fake_image1)+ \
                        self.backward_G(self.D, self.fake_image2)
        lz = torch.mean(torch.abs(self.fake_image2 - self.fake_image1)) / torch.mean(
            torch.abs(self.z_random2 - self.z_random1))
        eps = 1 * 1e-5
        self.loss_lz = 1 / (lz + eps)

        self.loss_G = self.loss_G_GAN
        self.loss_G.backward()
        self.gen_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = pred_fake.mean() - pred_real.mean()

        return loss_D

    def backward_G(self, netD, fake):
        pred_fake = netD.forward(fake)
        loss_G = - pred_fake.mean()

        return loss_G

    def update_lr(self):
        self.dis_sch.step()
        self.gen_sch.step()

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.D.load_state_dict(checkpoint['dis'])
        self.G.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
                'dis': self.D.state_dict(),
                'gen': self.G.state_dict(),
                'dis_opt': self.dis_opt.state_dict(),
                'gen_opt': self.gen_opt.state_dict(),
                'ep': ep,
                'total_it': total_it
                    }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        image_real = self.normalize_image(self.real_image).detach()
        image_fake = self.normalize_image(self.fake_image).detach()
        return torch.cat((image_real,image_fake),2)

    def normalize_image(self, x):
        return x[:,0:3,:,:]

    def test_forward(self):
        #label_one_hot = self.onehot_encoding(label)
        z_random = self.get_z_random(1, self.nz)
        outputs = self.G.forward(z_random)
        return outputs
