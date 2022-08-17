import torch
import torch.nn as nn
from torch.nn import init
import torch.autograd.profiler as profiler

from models.losses.gan_loss import DiscriminatorLoss

class BaseModel(nn.Module):
    def __init__(self, model, opt):
        super().__init__()
        self.model = model

        self.opt = opt

        if opt.discriminator_losses is not None:
            self.use_discriminator = True

            self.netD = DiscriminatorLoss(opt)

            if opt.isTrain:
                self.optimizer_D = torch.optim.Adam(
                    list(self.netD.parameters()),
                    lr=opt.lr_d,
                    betas=(opt.beta1, opt.beta2),
                )
                self.optimizer_G = torch.optim.Adam(
                    list(self.model.parameters()),
                    lr=opt.lr_g,
                    betas=(opt.beta1, opt.beta2),
                )
        else:
            depth_model_name = ["mvs_depth_estimator", "pts_regressor"]
            depth_params = list(filter(lambda kv: kv[0].split(".")[1] in depth_model_name, self.model.named_parameters()))
            base_params = list(filter(lambda kv: kv[0].split(".")[1] not in depth_model_name, self.model.named_parameters()))
            base_params = [i[1] for i in base_params]
            depth_params =  [i[1] for i in depth_params]
            self.use_discriminator = False
            self.optimizer_G = torch.optim.Adam(
                [
                    {"params": base_params},
                    {"params": depth_params, "lr": opt.lr_g * opt.depth_lr_scaling}
                ],
                lr=opt.lr_g,
                betas=(0.9, opt.beta2)
            )

        if opt.isTrain:
            self.old_lr = opt.lr

        if opt.init:
            self.init_weights()
        self.gan_loss_weight = opt.gan_loss_weight

    def init_weights(self, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if self.opt.init == "normal":
                    init.normal_(m.weight.data, 0.0, gain)
                elif self.opt.init == "xavier":
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif self.opt.init == "xavier_uniform":
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif self.opt.init == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                elif self.opt.init == "orthogonal":
                    init.orthogonal_(m.weight.data)
                elif self.opt.init == "":  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented"
                        % self.opt.init
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
        for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(self.opt.init, gain)

    def __call__(
        self, dataloader, isval=False, num_steps=1, return_batch=False
    ):
        """
        Main function call
        - dataloader: The sampler that choose data samples.
        - isval: Whether to train the discriminator etc.
        - num steps: not fully implemented but is number of steps in the discriminator for
        each in the generator
        - return_batch: Whether to return the input values
        """
        weight = 1.0 / float(num_steps)
        if isval:
            with profiler.record_function("load data"):
                batch = next(dataloader)
            with profiler.record_function("run model"):
                t_losses, output_images = self.model(batch)

            if self.opt.normalize_image:
                for k in output_images.keys():
                    if "Img" in k:
                        if not torch.is_tensor(output_images[k]):
                            output_images[k] = [0.5 * output_image + 0.5 for output_image in output_images[k]]
                        else:
                            output_images[k] = 0.5 * output_images[k] + 0.5

            if return_batch:
                return t_losses, output_images, batch
            return t_losses, output_images

        self.optimizer_G.zero_grad()
        if self.use_discriminator:
            all_output_images = []
            for j in range(0, num_steps):
                with profiler.record_function("load data"):
                    batch = next(dataloader)
                with profiler.record_function("run model"):
                    t_losses, output_images = self.model(batch)
                    del batch
                g_losses = self.netD.run_generator_one_step(
                    output_images["PredImg"], output_images["OutputImg"]
                )
                (
                    g_losses["Total Loss"] / weight
                    + t_losses["Total Loss"] * self.gan_loss_weight / weight
                ).mean().backward()
                all_output_images += [output_images]
            self.optimizer_G.step()

            self.optimizer_D.zero_grad()
            for step in range(0, num_steps):
                d_losses = self.netD.run_discriminator_one_step(
                    all_output_images[step]["PredImg"],
                    all_output_images[step]["OutputImg"],
                )
                (d_losses["Total Loss"] * self.gan_loss_weight / weight).mean().backward()
            self.optimizer_D.step()

            g_losses.pop("Total Loss")
            d_losses.pop("Total Loss")
            t_losses.update(g_losses)
            t_losses.update(d_losses)
        else:
            for step in range(0, num_steps):
                with profiler.record_function("load data"):
                    batch = next(dataloader)
                with profiler.record_function("run model"):
                    t_losses, output_images = self.model(batch)
                    del batch
                (t_losses["Total Loss"] / weight).mean().backward()
            self.optimizer_G.step()

        if self.opt.normalize_image:
            for k in output_images.keys():
                if "Img" in k:
                    if not torch.is_tensor(output_images[k]):
                        output_images[k] = [0.5 * output_image + 0.5 for output_image in output_images[k]]
                    else:
                        output_images[k] = 0.5 * output_images[k] + 0.5
        return t_losses, output_images
    def lr_annealing(self, scale):
        for g in self.optimizer_G.param_groups:
            g['lr'] = g['lr'] * scale
        if self.use_discriminator:
            for g in self.optimizer_D.param_groups:
                g['lr'] = g['lr'] * scale

