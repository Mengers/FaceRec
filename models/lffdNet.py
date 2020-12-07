"""
@Descripttion: This is menger's demo,which is only for reference
@version:
@Author: menger
@Date: 2020-11-20 09:32:00
@LastEditors:
@LastEditTime:
"""

import torch
import torch.nn as nn

# 3x3 convolution with padding
def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=True)

# 1x1 convolution
def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 1, bias=True)


class LFFDBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, is_brach=False):
        super(LFFDBlock, self).__init__()
        self.is_brach = is_brach
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out += x
        out = self.relu2(out)
        if self.is_brach:
            branch = out
            return out, branch
        else:
            return out

# loss branch
class LossBranch(nn.Module):
    def __init__(self, planes, num_classes=2):
        super(LossBranch, self).__init__()
        self.conv1 = conv1x1(planes, planes)
        self.conv2_score = conv1x1(planes, planes)
        self.conv3_score = conv1x1(planes, num_classes)
        self.conv2_box = conv1x1(planes, planes)
        self.conv3_box = conv1x1(planes, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        score = self.conv2_score(out)
        score = self.relu(score)
        score = self.conv3_score(score)

        box = self.conv2_box(out)
        box = self.relu(box)
        box = self.conv3_box(box)

        return score, box

# 25-layers Net
# 8 output layers
class LFFDNet(nn.Module):
    def __init__(self):
        super(LFFDNet, self).__init__()

        # tiny part
        self.tiny_part1 = nn.Sequential(
            conv3x3(3, 64, stride=2, padding=0),
            conv3x3(64, 64, stride=2, padding=0),
            LFFDBlock(64, 64, stride=1, is_brach=False),
            LFFDBlock(64, 64, stride=1, is_brach=False),
            LFFDBlock(64, 64, stride=1, is_brach=True),

        )
        self.tiny_part2 = nn.Sequential(
            LFFDBlock(64, 64, stride=1, is_brach=True),
        )
        # small part
        self.small_part1 = nn.Sequential(
            conv3x3(64, 64, stride=2, padding=0),
            LFFDBlock(64, 64, stride=1, is_brach=True)
        )
        self.small_part2 = nn.Sequential(
            LFFDBlock(64, 64, stride=1, is_brach=True)
        )
        # medium part
        self.medium_part1 = nn.Sequential(
            conv3x3(64, 128, stride=2, padding=0),
            LFFDBlock(128, 128, stride=1, is_brach=True)
        )
        # large part
        self.large_part1 = nn.Sequential(
            conv3x3(128, 128, stride=2, padding=0),
            LFFDBlock(128, 128, stride=1, is_brach=True)
        )
        self.large_part2 = nn.Sequential(
            LFFDBlock(128, 128, stride=1, is_brach=True)
        )
        self.large_part3 = nn.Sequential(
            LFFDBlock(128, 128, stride=1, is_brach=True)
        )

        # loss branch
        self.loss_branch1 = LossBranch(64, num_classes=2)
        self.loss_branch2 = LossBranch(64, num_classes=2)
        self.loss_branch3 = LossBranch(64, num_classes=2)
        self.loss_branch4 = LossBranch(64, num_classes=2)
        self.loss_branch5 = LossBranch(128, num_classes=2)
        self.loss_branch6 = LossBranch(128, num_classes=2)
        self.loss_branch7 = LossBranch(128, num_classes=2)
        self.loss_branch8 = LossBranch(128, num_classes=2)

    def forward(self, x):
        x, b1 = self.tiny_part1(x)
        score1, bbox1 = self.loss_branch1(b1)
        x, b2 = self.tiny_part2(x)
        score2, bbox2 = self.loss_branch2(b2)
        x, b3 = self.small_part1(x)
        score3, bbox3 = self.loss_branch3(b3)
        x, b4 = self.small_part2(x)
        score4, bbox4 = self.loss_branch4(b4)
        x, b5 = self.medium_part1(x)
        score5, bbox5 = self.loss_branch5(b5)
        x, b6 = self.large_part1(x)
        score6, bbox6 = self.loss_branch6(b6)
        x, b7 = self.large_part2(x)
        score7, bbox7 = self.loss_branch7(b7)
        x, b8 = self.large_part3(x)
        score8, bbox8 = self.loss_branch8(b8)

        outs = [score1, bbox1, score2, bbox2, score3, bbox3, score4, bbox4, score5, bbox5, score6, bbox6, score7, bbox7,
                score8, bbox8]

        return outs


def get_lffdnet():
    model = LFFDNet()
    return model


if __name__ == '__main__':
    import os
    from torch.autograd import Variable
    #from torchsummary import summary
    #from torchviz import make_dot
    #import tensorwatch

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    x_image = Variable(torch.randn(8, 3, 640, 640))

    net = get_lffdnet()
    print(net)
    y = net(x_image)
    print(y)

    #summary(net.to('cuda'), (3, 640, 640))

    """ 
       If you want to show with torchviz,
       you need to modify the return format of the network.
       """
    # vis_graph = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x_image)]))
    #
    # # vis_graph.format = 'png'
    # # vis_graph.format = 'pdf'
    # vis_graph.format = 'svg'
    #
    # vis_graph.render('lffdNet.gv')
    #
    # vis_graph.view()











