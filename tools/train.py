'''
@Descripttion: This is menger's demo,which is only for reference
@version:
@Author: menger
@Date: 2020-12-8
@LastEditors:
@LastEditTime:
'''
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.lffdNet import LFFDNet
from HicFaceDec.data.dataset import LFFDDataset
from lib.lossfun import LFFDLoss
from tqdm import tqdm
from HicFaceDec.data.augmentor import LFFDAug
from torchvision import transforms

num_classes = 2
num_epochs = 500
image_size = 640
strides = [4,4,8,8,16,32,32,32]
feature_maps = [159,159,79,79,39,19,19,19]
scale_factors = [15, 20, 40, 70, 110, 250, 400, 560]
scales = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)]

if __name__ == '__main__':
    transform = LFFDAug()
    # transform = transforms.Compose([
    #    transforms.Resize(640),
    #    transforms.ToTensor(),
    #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    dataset = LFFDDataset("/home/hp/Data/FaceData/FaceDex", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=6, pin_memory=True)

    net = LFFDNet()
    net.cuda()
    # net.load_state_dict(torch.load("./ckpt/3090.pth"))
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # optimizer = optim.SGD(net.parameters(), lr=1e-1,momentum=0.9,weight_decay=0.)
    # scheduler = optim.lr_scheduler.StepLR(optimizer,100,0.1)
    loss_fn = LFFDLoss()
    num_batches = len(dataloader)
    min_loss = float("inf")
    for epoch in range(num_epochs):
        epoch_loss_cls = 0.
        epoch_loss_reg = 0.

        for img, gt_pos, gt_labels, not_ignored, img_name in tqdm(dataloader):
            print(img_name)
            img = img.cuda()
            gt_pos = gt_pos.cuda()
            not_ignored = not_ignored.cuda()
            gt_labels = gt_labels.cuda()
            cls, loc = net(img)
            reg_loss, cls_loss = loss_fn(cls, loc, gt_labels, gt_pos, not_ignored)
            epoch_loss_cls += cls_loss.item()
            epoch_loss_reg += reg_loss.item()
            loss = (reg_loss + cls_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # scheduler.step()
        print("cls_loss:{}---reg_loss:{}".format(epoch_loss_cls / num_batches, epoch_loss_reg / num_batches))
        if (epoch_loss_cls / num_batches + epoch_loss_reg / num_batches) < min_loss:
            min_loss = epoch_loss_cls / num_batches + epoch_loss_reg / num_batches
            torch.save(net.state_dict(), "./ckpt/{}.pth".format(
                int(epoch_loss_cls / num_batches * 1000 + epoch_loss_reg / num_batches * 1000)))
