import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.nn import functional as F
from datasets.bmnist import bmnist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from torchvision.utils import save_image

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.layer_1=nn.Linear(784,hidden_dim)
        self.mean_layer=nn.Linear(hidden_dim,z_dim)
        self.std_layer = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        mean, std = None, None
        out_1=F.relu(self.layer_1(input))
        mean=self.mean_layer(out_1)
        logvar=self.std_layer(out_1)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.layer_1=nn.Linear(z_dim,hidden_dim)
        self.layer_2=nn.Linear(hidden_dim,784)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        out_1=F.relu(self.layer_1(input))
        out_2=torch.sigmoid(self.layer_2(out_1))
        ##raise NotImplementedError()

        return out_2


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.encoder=self.encoder.cuda()
        self.decoder = Decoder(hidden_dim, z_dim)
        self.decoder=self.decoder.cuda()

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        average_negative_elbo = None
        mean,logvar=self.encoder(input)
        decoder_data=self.reparametrize(mean,logvar)
        decoder_means=self.decoder(decoder_data)
        average_negative_elbo=self.elbo(decoder_means,input,mean,logvar)
        return average_negative_elbo

    def reparametrize(self,mean,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(ARGS.zdim)
        eps=eps.cuda()
        ##eps = torch.randn_like(std)
        data=torch.mul(eps,std)
        data=torch.add(data,mean)
        data=data.cuda()
        return data

    def elbo(self,x_recons,x,mean,logvar):
        x_recons_temp=x_recons.reshape(x_recons.shape[0],784)
        x_temp = x.reshape(x.shape[0], 784)
        bin_cross_entropy=self.calc_binary_entropy(x_recons_temp,x_temp)
        kl_div=self.calc_kl_div(mean,logvar)
        elbo=bin_cross_entropy+kl_div
        elbo_avg=elbo
        return -elbo_avg

    def calc_binary_entropy(self,x_recon,x_temp):
        re_loss = x_temp * torch.log(x_recon + 1e-9) + (1 - x_temp) * torch.log(1 - x_recon + 1e-9)
        re_loss=torch.sum(re_loss,dim=1)
        return torch.mean(re_loss)

    def calc_kl_div(self,mean,logvar):
        kl = 0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

        kl = torch.sum(kl, dim=1)

        return torch.mean(kl)


    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        sampled_ims, im_means = torch.zeros(n_samples,1,28,28), torch.zeros(n_samples,784)
        for i in range(0,n_samples):
            eps = torch.randn(ARGS.zdim)
            eps = eps.cuda()
            image = self.decoder(eps)
            image = image.cpu()
            image = image.view(1,28, 28)
            im_means[i,:]=image.view(784)
            plt.plot(im_means[i].detach().numpy())
            plt.savefig('vae_means/vae_means_'+str(i)+'.png')
            plt.clf()
            ##image = image.detach().numpy()
            sampled_ims[i,:,:]=image
        save_image(sampled_ims,
                   'vae_images/{}.png'.format(n_samples),
                   nrow=20, normalize=True)


        ##return sampled_ims, im_means



def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    average_epoch_elbo = None
    index=0
    for i_batch,sample_batch in enumerate(data):
        index+=1
        temp=sample_batch.view(sample_batch.shape[0],-1,784)
        temp=temp.cuda()
        avg_elbo=model.forward(temp)
        if(model.training==True):
            optimizer.zero_grad()
            avg_elbo.backward()
            optimizer.step()
        if(average_epoch_elbo is None):
            average_epoch_elbo=avg_elbo.item()
        else:
            average_epoch_elbo=average_epoch_elbo+avg_elbo.item()
    average_epoch_elbo=average_epoch_elbo/index
    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data
    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return -train_elbo, -val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    model=model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    ##save_elbo_plot(train_curve, val_curve, 'elbo.pdf')
    model.sample(100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')

    ARGS = parser.parse_args()

    main()
