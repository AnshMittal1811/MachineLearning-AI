import torch
from torch import nn

class Encoder_identity(nn.Module):
    def __init__(self, input_size, num_features, num_classes):
        super(Encoder_identity, self).__init__()
        self.n_input = input_size
        self.n_out_id = num_features
        self.n_class_id = num_classes
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_input,1024),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2_id_feature = nn.Sequential(
            nn.Linear(512, self.n_out_id),

        )
        self.fc3_id_label = nn.Sequential(
            nn.Linear(self.n_out_id, self.n_class_id),
        )

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        id_feature = self.fc2_id_feature(x)
        id_label=self.fc3_id_label(id_feature)

        return id_feature,id_label


class Encoder_expression(nn.Module):
    def __init__(self, input_size, num_features, num_classes):
        super(Encoder_expression, self).__init__()
        self.n_input = input_size
        self.n_out_ex = num_features
        self.n_class_ex = num_classes
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_input,1024),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2_ex_feature = nn.Sequential(
                    nn.Linear(512, self.n_out_ex),

                    )
        self.fc3_exp_label = nn.Sequential(
            nn.Linear(self.n_out_ex, self.n_class_ex),

        )

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)

        ex_feature = self.fc2_ex_feature(x)
        ex_label = self.fc3_exp_label(ex_feature)

        return ex_feature,ex_label


class Decoder(nn.Module):
    def __init__(self, num_features, output_size):
        super(Decoder, self).__init__()
        self.n_input = num_features
        self.n_out = output_size
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_input,512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(1024, self.n_out)
                    )

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class Generator(nn.Module):
    def __init__(self, input_size, num_features):
        super(Generator, self).__init__()
        self.n_input = input_size
        self.n_out = num_features
        self.fc0 = nn.Sequential(
                    nn.Linear(self.n_input, 256),
                    nn.LeakyReLU(0.2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.LeakyReLU(0.2)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.LeakyReLU(0.2)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(1024, self.n_out)
                    )
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, n_feat_id, n_feat_ex, n_class_id, n_class_ex):
        super(Discriminator, self).__init__()
        self.n_in_id = n_feat_id
        self.n_in_ex = n_feat_ex
        self.n_out = 1
        self.n_class_id = n_class_id
        self.n_class_ex = n_class_ex

        self.fc0_id = nn.Sequential(
                    nn.Linear(self.n_in_id, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                    )
        self.fc0_ex = nn.Sequential(
            nn.Linear(self.n_in_ex, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.fc0_joint = nn.Sequential(
            nn.Linear(self.n_in_id + self.n_in_ex, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.5)
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.n_out)
                    )
        self.fc4_id = nn.Sequential(
                    nn.Linear(256, self.n_class_id)
                    )
        self.fc4_ex = nn.Sequential(
                    nn.Linear(256, self.n_class_ex)
                    )
    def forward(self, x):
        x_=x[:]
        x_= self.fc0_joint(x_)
        x_ = self.fc1(x_)
        x_ = self.fc2(x_)
        x_= self.fc3(x_)

        #-----------------------------------
        x_id = x[:, 0:self.n_in_id]
        x_id = self.fc0_id(x_id)
        x_id = self.fc1(x_id)
        x_id = self.fc2(x_id)
        x_id=self.fc4_id(x_id)

        #------------------------------
        x_ex = x[:, self.n_in_id:self.n_in_id+self.n_in_ex]
        x_ex = self.fc0_ex(x_ex)
        x_ex = self.fc1(x_ex)
        x_ex = self.fc2(x_ex)
        x_ex=self.fc4_ex(x_ex)

        return x_,x_id,x_ex

class generator_network(nn.Module):
    def __init__(self, z_dim_id, z_dim_ex, n_feat_id, n_feat_ex):
        super(generator_network, self).__init__()
        self.generator_id = Generator(input_size=z_dim_id, num_features=n_feat_id)
        self.generator_exp = Generator(input_size=z_dim_ex, num_features=n_feat_ex)


    def generate_id(self, x):
        return self.generator_id(x)


    def generate_ex(self, x):
        return self.generator_exp(x)


    def forward(self, z_n,z_e):

        id_feature= self.generate_id(z_n)
        ex_feature= self.generate_ex(z_e)
        return id_feature,ex_feature
