import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


"""def calc_mean_std_mask(feat, mask, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    #assert (np.unique(mask.cpu().detach().numpy()) == np.array([0,1])).all()
    N, C = size[:2]
    cnt_o = mask.view(N, 1, -1).sum(dim=2).view(N, 1, 1, 1) + eps
    feat_o = (feat*mask).view(N, C, -1)
    feat_mean = feat_o.sum(dim=2).view(N, C, 1, 1)/cnt_o
    print(feat.size(), feat_mean.size(), mask.size())
    t = ((feat-feat_mean)*mask).view(N, C, -1)

    feat_var = torch.sum(torch.pow(t, 2), dim=2)/cnt_o.squeeze(-1).squeeze(-1) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    return feat_mean, feat_std.sqrt()"""


# with mask
def spatially_aware_adaptive_instance_norm(content_feat, style_feat, mask):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    assert (content_feat.size()[-2:] == mask.size()[-2:])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std_mask(content_feat, mask)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)




def calc_mean_std_mask(feat, mask, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    # assert (np.unique(mask.cpu().detach().numpy()) == np.array([0,1])).all()
    N, C = size[:2]
    cnt_o = mask.view(N, 1, -1).sum(dim=2).view(N, 1, 1, 1) + eps
    feat_o = (feat * mask).view(N, C, -1)
    feat_mean = feat_o.sum(dim=2).view(N, C, 1, 1) / cnt_o

    # print(feat.size(), feat_mean.size(), mask.size())
    t = ((feat - feat_mean) * mask).view(N, C, -1)
    # print((feat-feat_mean.squeeze(-1).squeeze(-1)))
    # print(t.size())
    # print(torch.pow(t, 2))
    # print(torch.pow(t, 2).size())
    # print(torch.sum(torch.pow(t, 2), dim=2).size())
    # print(cnt_o.size())
    # print(((cnt_o - 1).view(N, 1)).size())
    feat_var = torch.sum(torch.pow(t, 2), dim=2) / ((cnt_o - 1).view(N, 1)) + eps
    # print(feat_var.size())
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    return feat_mean, feat_std




def adaptive_instance_norm(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class Vgg19(torch.nn.Module):
    def __init__(self, weight_path='vgg19.pth'):
        super(Vgg19, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        # vgg19.load_state_dict(torch.load(weight_path))

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg19.features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg19.features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg19.features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg19.features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        conv1_1 = self.slice1(inputs)
        conv2_1 = self.slice2(conv1_1)
        conv3_1 = self.slice3(conv2_1)
        conv4_1 = self.slice4(conv3_1)

        return conv1_1, conv2_1, conv3_1, conv4_1


"""class SplatBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SplatBlock, self).__init__()
        self.conv_in = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
                nn.ReLU(inplace=True))
        self.conv_short  = nn.Sequential(
                nn.Conv2d(ch_out*16, ch_out, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
                nn.Conv2d(ch_out, ch_out, 1, stride=1),
                nn.ReLU(inplace=True))
        self.conv_out2 = nn.Sequential(
                nn.Conv2d(ch_out, ch_out, 3, padding=1, stride=1),
                nn.ReLU(inplace=True))

    def forward(self, feat_s, feat_c, feat_adain, mask):
        feat_s = self.conv_in(feat_s)
        feat_c = self.conv_in(feat_c)
        #print(feat_c.size(), feat_s.size())
        feat_norm = spatially_aware_adaptive_instance_norm(feat_c, feat_s, mask)
        #print(feat_norm.size(), feat_adain.size())
        shortcut = self.conv_short(feat_adain)
        output = feat_norm + shortcut
        output = self.conv_out(output)
        output = self.conv_out2(output)
        feat_s = self.conv_out2(feat_s)
        return output, feat_s"""


class SplatBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SplatBlock, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(ch_out * 16 + ch_out, ch_out, 1, stride=1),
            nn.ReLU(inplace=True))

    def forward(self, feat_s, feat_c, feat_adain, mask):
        feat_s = self.conv_in(feat_s)
        feat_c = self.conv_in(feat_c)
        # print(feat_c.size(), feat_s.size())
        feat_norm = spatially_aware_adaptive_instance_norm(feat_c, feat_s, mask)
        # print(feat_norm.size(), feat_adain.size())
        output = torch.cat([feat_norm, feat_adain], dim=1)
        output = self.conv_out(output)
        return output, feat_s


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size,
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)]) # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) # norm to [0,1] NxHxWx1
        hg, wg = hg*2-1, wg*2-1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1) # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()
        self.conv1 = ConvBlock(4, 16, kernel_size=3, padding=1, batch_norm=bn)
        self.conv2 = ConvBlock(16, 1, kernel_size=3, padding=1, activation=nn.Tanh)

    def forward(self, inputs, mask):
        output = self.conv1(torch.cat([inputs, mask], dim=1))
        output = self.conv2(output)
        return output


class StyleNetwork(nn.Module):
    def __init__(self, size=256):
        super(StyleNetwork, self).__init__()
        self.size = size
        self.extractor = Vgg19().eval()
        self.splat1 = SplatBlock(64, 8)
        self.splat2 = SplatBlock(8, 16)
        self.splat3 = SplatBlock(16, 32)
        self.feat_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        self.local_layer = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1))

        self.global_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.global_fc = nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64))

        self.final_conv = nn.Conv2d(64, 96, 1, stride=1, padding=0, bias=False)

    def forward(self, content, style, mask):
        lr_c = F.interpolate(content, (self.size, self.size),
                             mode='bilinear', align_corners=True)
        lr_s = F.interpolate(style, (self.size, self.size),
                             mode='bilinear', align_corners=True)
        feats_c = self.extractor(lr_c)
        feats_s = self.extractor(lr_s)

        """feat_adain1 = adaptive_instance_norm(feats_c[1], feats_s[1])
        out_c1, out_s1 = self.splat1(feats_c[0], feats_s[0], feat_adain1)

        feat_adain2 = adaptive_instance_norm(feats_c[2], feats_s[2])
        out_c2, out_s2 = self.splat2(out_c1, out_s1, feat_adain2)

        feat_adain3 = adaptive_instance_norm(feats_c[3], feats_s[3])
        out_c3, out_s3 = self.splat3(out_c2, out_s2, feat_adain3)"""

        mask = F.adaptive_max_pool2d(input=mask, output_size=feats_c[1].size()[-2:])
        feat_adain1 = spatially_aware_adaptive_instance_norm(feats_c[1], feats_s[1], mask)
        out_c1, out_s1 = self.splat1(feats_c[0], feats_s[0], feat_adain1, mask)

        mask = F.adaptive_max_pool2d(input=mask, output_size=feats_c[2].size()[-2:])
        feat_adain2 = spatially_aware_adaptive_instance_norm(feats_c[2], feats_s[2], mask)
        out_c2, out_s2 = self.splat2(out_c1, out_s1, feat_adain2, mask)

        mask = F.adaptive_max_pool2d(input=mask, output_size=feats_c[3].size()[-2:])
        feat_adain3 = spatially_aware_adaptive_instance_norm(feats_c[3], feats_s[3], mask)
        out_c3, out_s3 = self.splat3(out_c2, out_s2, feat_adain3, mask)

        out_feat = self.feat_conv(out_c3)

        local_feat = self.local_layer(out_feat)
        global_feat = self.global_conv(out_feat)
        batch_size = global_feat.size()[0]
        global_feat = global_feat.view(batch_size, -1)
        global_feat = self.global_fc(global_feat)
        fuse_feat = local_feat + global_feat.view(batch_size, -1, 1, 1)
        output = self.final_conv(fuse_feat)
        output = output.view(batch_size, 12, 8, 16, 16)
        return output


class BilateralNetwork(nn.Module):
    def __init__(self, size=256):
        super(BilateralNetwork, self).__init__()
        self.size = size
        self.foreground = StyleNetwork(size)
        self.background = StyleNetwork(size)
        self.guide = GuideNN(bn=False)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, content, foreground_style, background_style, mask):
        foreground_grid = self.foreground(content, foreground_style, mask)
        background_grid = self.background(content, background_style, 1 - mask)
        guide_map = self.guide(content, mask)
        soft_grid_mask = self.compute_soft_grid_mask(guide_map, mask, content.size()[-2:], (8, 16, 16))
        grid = foreground_grid * soft_grid_mask + (1 - soft_grid_mask) * background_grid
        slice_coeffs = self.slice(grid, guide_map)
        output = self.apply_coeffs(slice_coeffs, content)
        output = torch.sigmoid(output)
        predict_mask_f = torch.sigmoid(self.apply_coeffs(self.slice(grid, guide_map), mask.repeat(1, 3, 1, 1)))
        predict_mask_b = torch.sigmoid(self.apply_coeffs(self.slice(grid, guide_map), (1 - mask).repeat(1, 3, 1, 1)))

        return output, foreground_grid, background_grid, guide_map, predict_mask_f, predict_mask_b

    def compute_soft_grid_mask(self, guide_map, mask, image_size, grid_size):
        """
        Input: Learned guide map z, pixel mask M_pxl, image size (w, h), grid size (D, W, H)
        Output: Soft grid mask Mgrid
        """
        w, h = image_size
        D, W, H = grid_size
        batch_size = guide_map.shape[0]
        if torch.cuda.is_available():
            Mgrid = torch.zeros((batch_size, *grid_size)).cuda()
        else:
            Mgrid = torch.zeros((batch_size, *grid_size))
        zD = torch.floor(guide_map * mask * D)
        zD = zD.squeeze(1)
        sw, sh = w // W, h // H
        for x in range(W):  # ←1 to W and y←1 to H do
            for y in range(H):
                patch = zD[:, x * sw:(x + 1) * sw:, y * sh:(y + 1) * sh]
                Mgrid[:, :, x, y] = torch.sum((patch > 0).flatten(start_dim=1), dim=-1).unsqueeze(-1).repeat(1, D)
                for d in range(D):  # d←1 to D do
                    if d in patch:  # then
                        Mgrid[:, d, x, y] = torch.sum((patch == d).flatten(start_dim=1), dim=-1)
        Mgrid = Mgrid / (sw * sh)  # Normalize grid mask: Mgrid←Mgrid/(sw×sh);
        assert Mgrid.max() <= 1
        assert Mgrid.min() >= 0
        return Mgrid.unsqueeze(1).repeat(1, 12, 1, 1, 1)


def get_2_model():
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    state_dict = torch.load("models/best_model_style.pth", map_location=device)
    model = BilateralNetwork()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preproc(content, style1, style2, mask, size=256):
    content = cv2.resize(content, (size, size)).astype('float')
    style1 = cv2.resize(style1, (size, size)).astype('float')
    style2 = cv2.resize(style2, (size, size)).astype('float')
    mask = cv2.resize(mask, (size, size)).astype('float')[:, :, None]

    content = torch.from_numpy(content).permute(2, 0, 1)/255.0
    assert style1.max() > 1
    style1 = torch.from_numpy(style1.copy().astype('float')).permute(2,0,1)/255.0
    assert style2.max() > 1
    style2 = torch.from_numpy(style2.copy().astype('float')).permute(2,0,1)/255.0
    mask = torch.from_numpy(mask.astype('float')).permute(2,0,1)
    return content, style1, style2, mask


def go_2_style(model, content, style1, style2, mask):
    content_shape = content.shape
    content, style1, style2, mask = preproc(content, style1, style2, mask[:,:,None])
    content = content[None]
    style1 = style1[None]
    style2 = style2[None]
    mask = mask[None]

    content = content.float()
    style1 = style1.float()
    style2 = style2.float()
    mask = mask.float()

    with torch.no_grad():
        res = model(content, style1, style2, mask)
        res = res[0]

    result = cv2.resize((res[0] * 255.).detach().numpy().astype(np.uint8).transpose((1, 2, 0)),
                        (content_shape[1], content_shape[0]))

    return Image.fromarray(result)
