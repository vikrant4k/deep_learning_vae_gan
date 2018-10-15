import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self,latent_dim):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 768
        #   Output non-linearity
        self.latent_dim=latent_dim
        self.layer1=nn.Linear(latent_dim,128)
        self.relu_layer_1=nn.LeakyReLU(0.2)

        self.layer2=nn.Linear(128,256)
        self.batch_norm2=nn.BatchNorm1d(256)
        self.relu_layer_2=nn.LeakyReLU(0.2)

        self.layer3 = nn.Linear(256,512)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.relu_layer_3 = nn.LeakyReLU(0.2)

        self.layer4 = nn.Linear(512, 1024)
        self.batch_norm4 = nn.BatchNorm1d(1024)
        self.relu_layer_4 = nn.LeakyReLU(0.2)

        self.layer5 = nn.Linear(1024, 784)
        self.relu_5=nn.Tanh()


    def forward(self, z):
        # Generate images from z

        layer1_out=self.relu_layer_1(self.layer1(z))
        layer2_out=self.relu_layer_2(self.batch_norm2(self.layer2(layer1_out)))
        layer3_out=self.relu_layer_3(self.batch_norm3(self.layer3(layer2_out)))
        layer4_out=self.relu_layer_4(self.batch_norm4(self.layer4(layer3_out)))
        layer_5_out=self.relu_5(self.layer5(layer4_out))
        return layer_5_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        self.layer1=nn.Linear(784,512)
        self.relu_1=nn.LeakyReLU(0.2)
        self.layer2 = nn.Linear(512, 256)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.layer3 = nn.Linear(256,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self, img):
        # return discriminator score for img
        layer_1_out=self.relu_1(self.layer1(img))
        layer_2_out=self.relu_2(self.layer2(layer_1_out))
        layer_3_out=self.sigmoid(self.layer3(layer_2_out))
        return layer_3_out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    loss=nn.BCELoss()
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            imgs=imgs.cuda()
            image_conv=imgs.view(imgs.shape[0],-1, 784)

            real_image=torch.ones(imgs.shape[0])
            fake_image=torch.zeros(imgs.shape[0])
            real_image=real_image.cuda()
            fake_image=fake_image.cuda()

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            latent_variable=torch.randn(imgs.shape[0],args.latent_dim)
            latent_variable = latent_variable.cuda()
            latent_variable=latent_variable.cuda()
            genrated_image=generator(latent_variable)
            result_fake=discriminator(genrated_image)
            image_loss=loss(result_fake,real_image)
            image_loss.backward()
            optimizer_G.step()


            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            latent_variable = torch.randn(imgs.shape[0], args.latent_dim)
            latent_variable=latent_variable.cuda()
            genrated_image = generator(latent_variable)
            genrated_image=genrated_image.detach()
            real_image_state=discriminator(image_conv)
            real_image_loss=loss(real_image_state,real_image)
            fake_image_state=discriminator(genrated_image)
            fake_image_loss=loss(fake_image_state,fake_image)
            total_loss=fake_image_loss+real_image_loss
            total_loss.backward()
            optimizer_D.step()
            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                plt_data = genrated_image[:25]
                plt_data = plt_data.view(plt_data.size(0), 1, 28, 28)
                save_image(plt_data.cpu(),
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
            niter = epoch * len(dataloader) + i
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(dataloader),
                                                                             image_loss.item(), total_loss.item()))



def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    generator.cuda()
    discriminator = Discriminator()
    discriminator.cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    # torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
