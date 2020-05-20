import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=False, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        if dropout > 0:
            classifier += [nn.Dropout(p=dropout)]
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        f = x
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = f.div(f_norm)
        x = self.classifier(x)
        return x, f


class Residual_attention_Net(nn.Module):

    def __init__(self, input_dim=2048):
        super(Residual_attention_Net, self).__init__()
        self.res_attention = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 1, 3),
        )

    def forward(self, x):
        return self.res_attention(x)


class Channel_attention_net(nn.Module):

    def __init__(self, channel=256, reduction=16):
        super(Channel_attention_net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        # y = self.fc(y)
        return y.expand_as(x)


class att_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(att_block, self).__init__()
        self.gap = nn.AvgPool2d((1, 1))
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
        # self.softmax = torch.softmax()

    def forward(self, x):
        x = self.gap(x)
        x = self.conv(x)
        x = torch.sigmoid(x)

        return x


class ds_net(nn.Module):

    def __init__(self, class_num):
        super(ds_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft

        # self.att_block1 = att_block(256, 32)
        # self.att_block2 = att_block(512, 32)
        # self.att_block3 = att_block(1024, 32)
        # self.att_block4 = att_block(2048, 32)
        # self.channel_att = Channel_attention_net(channel=512)
        # self.res_att = Residual_attention_Net(input_dim=512)
        self.channel_att = Channel_attention_net(channel=1024)
        self.res_att = Residual_attention_Net(input_dim=1024)
        self.adjust = nn.Conv2d(2048, 1, 1, 1)
        self.classifier = ClassBlock(2048, class_num, dropout=0.5, relu=False)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        c_att = self.channel_att(x)
        res_att = self.res_att(x)
        x = x * c_att
        x = x * res_att
        x = self.model.layer4(x)
        att_score = self.adjust(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x, f = self.classifier(x)
        return x, att_score, f


# Define a 2048 to 2 Model
class verif_net(nn.Module):
    def __init__(self):
        super(verif_net, self).__init__()
        self.classifier = ClassBlock(512, 2, dropout=0.75, relu=False)

    def forward(self, x):
        x = self.classifier.classifier(x)
        return x
