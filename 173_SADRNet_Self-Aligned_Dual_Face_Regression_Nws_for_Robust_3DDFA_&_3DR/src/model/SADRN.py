from src.model.modules import ResBlock4, Conv2d_BN_AC, AttentionModel, ConvTranspose2d_BN_AC, VisibilityRebuildModule
from src.model.loss import *


class SADRN(nn.Module):
    def __init__(self):
        super(SADRN, self).__init__()
        self.feature_size = 16
        feature_size = self.feature_size

        self.layer0 = Conv2d_BN_AC(in_channels=3, out_channels=feature_size, kernel_size=4, stride=1, padding=1)
        self.block1 = ResBlock4(in_channels=feature_size, out_channels=feature_size * 2, stride=2,
                                with_conv_shortcut=True)  # 128 x 128 x 32
        self.block2 = ResBlock4(in_channels=feature_size * 2, out_channels=feature_size * 2, stride=1,
                                with_conv_shortcut=False)  # 128 x 128 x 32
        self.block3 = ResBlock4(in_channels=feature_size * 2, out_channels=feature_size * 4, stride=2,
                                with_conv_shortcut=True)  # 64 x 64 x 64
        self.block4 = ResBlock4(in_channels=feature_size * 4, out_channels=feature_size * 4, stride=1,
                                with_conv_shortcut=False)  # 64 x 64 x 64
        self.block5 = ResBlock4(in_channels=feature_size * 4, out_channels=feature_size * 8, stride=2,
                                with_conv_shortcut=True)  # 32 x 32 x 128
        self.block6 = ResBlock4(in_channels=feature_size * 8, out_channels=feature_size * 8, stride=1,
                                with_conv_shortcut=False)  # 32 x 32 x 128
        self.block7 = ResBlock4(in_channels=feature_size * 8, out_channels=feature_size * 16, stride=2,
                                with_conv_shortcut=True)  # 16 x 16 x 256
        self.block8 = ResBlock4(in_channels=feature_size * 16, out_channels=feature_size * 16, stride=1,
                                with_conv_shortcut=False)  # 16 x 16 x 256
        self.block9 = ResBlock4(in_channels=feature_size * 16, out_channels=feature_size * 32, stride=2,
                                with_conv_shortcut=True)  # 8 x 8 x 512
        self.block10 = ResBlock4(in_channels=feature_size * 32, out_channels=feature_size * 32, stride=1,
                                 with_conv_shortcut=False)  # 8 x 8 x 512

        self.attention_branch = AttentionModel(num_features_in=feature_size * 8)
        self.decoder_low = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 32, kernel_size=4,
                                  stride=1),  # 8 x 8 x 512
            ConvTranspose2d_BN_AC(in_channels=feature_size * 32, out_channels=feature_size * 16, kernel_size=4,
                                  stride=2),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4,
                                  stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 16, kernel_size=4,
                                  stride=1),  # 16 x 16 x 256
            ConvTranspose2d_BN_AC(in_channels=feature_size * 16, out_channels=feature_size * 8, kernel_size=4,
                                  stride=2),  # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),
            # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 8, kernel_size=4, stride=1),
            # 32 x 32 x 128
            ConvTranspose2d_BN_AC(in_channels=feature_size * 8, out_channels=feature_size * 4, kernel_size=4, stride=2),
            # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),
            # 64 x 64 x 64
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 4, kernel_size=4, stride=1),
            # 64 x 64 x 64
        )
        self.decoder_kpt = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Sequential())
        )
        self.decoder_offset = nn.Sequential(
            ConvTranspose2d_BN_AC(in_channels=feature_size * 4, out_channels=feature_size * 2, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 2, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 2, out_channels=feature_size * 1, kernel_size=4, stride=2),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=feature_size * 1, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=feature_size * 1, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1),
            ConvTranspose2d_BN_AC(in_channels=3, out_channels=3, kernel_size=4, stride=1, activation=nn.Sequential())
        )

        # self.rebuilder = P2RNRebuildModule()
        self.rebuilder = VisibilityRebuildModule()
        # self.rebuilder=EstimateRebuildModule()

        self.fwrse = FaceWeightedRSE()
        self.bce = BinaryCrossEntropy()
        self.smooth_loss = SmoothLoss()
        self.nme = NME()
        self.frse = FaceRSE()
        self.kptc = KptRSE()
        self.mae = MAE()

    def forward(self, inpt, targets, mode='predict'):
        # torch.autograd.set_detect_anomaly(True)
        img = inpt['img']
        x = self.layer0(img)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        attention = self.attention_branch(x)
        att_feats = torch.stack([x[i] * torch.exp(attention[i]) for i in range(len(x))], dim=0)
        f = self.block7(att_feats)
        # f = self.block7(x)
        f = self.block8(f)
        f = self.block9(f)
        f = self.block10(f)

        f = self.decoder_low(f)
        kpt_uvm = self.decoder_kpt(f)
        offset_uvm = self.decoder_offset(f)

        # from src.util.show_debugger import show_batch
        # show_batch((inpt['img'], targets['face_uvm'], targets['offset_uvm'], targets['attention_mask']))

        if mode == 'train':
            # face_uvm = self.rebuilder(offset_uvm, kpt_uvm)
            face_uvm = kpt_uvm
            # face_uvm=self.rebuilder(targets['offset_uvm'],targets['face_uvm'])
            # show_batch((inpt['img'], face_uvm, targets['offset_uvm'], targets['attention_mask']))
            # show_batch((inpt['img'], targets['face_uvm'], targets['offset_uvm'], targets['attention_mask']))

        else:
            face_uvm = self.rebuilder(offset_uvm, kpt_uvm)

        if mode == 'train':
            loss = {}
            # loss['face_uvm']=self.fwrse(targets['face_uvm'],face_uvm)*FACE_UVM_LOSS_RATE
            loss['offset_uvm'] = self.fwrse(targets['offset_uvm'], offset_uvm) * OFFSET_UVM_LOSS_RATE
            loss['face_uvm'] = loss['offset_uvm'] * 0
            loss['kpt_uvm'] = self.fwrse(targets['face_uvm'], kpt_uvm) * KPT_UVM_LOSS_RATE
            loss['attention_mask'] = self.bce(targets['attention_mask'], attention) * ATT_LOSS_RATE
            loss['smooth'] = self.smooth_loss(offset_uvm) * SMOOTH_LOSS_RATE
            return loss
        elif mode == 'eval':
            metrics = {}
            metrics['offset_uvm'] = self.frse(targets['offset_uvm'], offset_uvm)
            metrics['face_uvm'] = self.nme(targets['face_uvm'], face_uvm)
            metrics['kpt_uvm'] = self.kptc(targets['face_uvm'], kpt_uvm)
            metrics['attention_mask'] = self.mae(targets['attention_mask'], attention)
            return metrics
        elif mode == 'predict':
            out = {}
            out['face_uvm'] = face_uvm
            out['kpt_uvm'] = kpt_uvm
            out['offset_uvm'] = offset_uvm
            out['attention_mask'] = attention
            return out


def get_model():
    return SADRN()
