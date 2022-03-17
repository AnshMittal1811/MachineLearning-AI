import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")



class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, **kwargs):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        
        hidden_dims = [32, 64, 128, 256, 512, 1024]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        padding_pattern = [1,0,0,0,1]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=padding_pattern[i]),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input):

        result=input
        for layer in self.encoder:
            #print(result.shape)
            #print(layer)
            result = layer(result)
        #print("here")
        result = torch.flatten(result, start_dim=1)
        #print("here2", result.shape)
        
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        #print("here3")
        log_var = self.fc_var(result)
        #print("here4")
        return [mu, log_var]

    def call_decoder(self, x):
        #print("---decode----")
        result = x
        for layer in self.decoder:
            #print(layer)
            #print(result.shape)
            result = layer(result)
        return result

    def decode(self, z):

        result = self.decoder_input(z)
        #print("her5", result.shape)
        #result = result.view(-1, 512, 2, 2)
        result = result.view(-1, 1024, 2, 2)
        #print("here6", result.shape)
        result = self.call_decoder(result)
        #print("here7")
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):


        return self.forward(x)[0]

if __name__ == "__main__":
    # simple dimension debugging

    vae = VAE(in_channels=3, latent_dim=64).to(device)
    # form: B x C x W x H
    x = torch.randn(1, 3, 100,100).to(device)
    y, z_mu, z_logsig, z = vae(x)
    print(y.shape, "ssss")
    #torch.nn.Conv2d()
    #xrec = vae.decode(z)
    #print(z.shape)
    #print(xrec.shape)
    #print(x.shape, z.shape, y.shape)