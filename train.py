from __future__ import print_function
import torch
import torch.nn.functional as F
import time
from dataprocess.getpath import findpath
from dataprocess.loader import ImageDataLoader
from models.model import PSMNet

assert torch.cuda.is_available()
# train parameters
data_path = '.'
save_path = '.'
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
max_disparity = 192
epochs = 10
batch_size = 5
learning_rate = 0.001


def main():
    # load the data
    
    left_paths, right_paths, disp_paths = findpath(data_path)
    Loader = torch.utils.data.DataLoader(
        ImageDataLoader(left_paths, right_paths, disp_paths, True),
        batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False
    )

    # deploy the model
    model = PSMNet(max_disparity)
    model = torch.nn.DataParallel(model)
    model.cuda()

    # set the  optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    def train(imgL, imgR, disp):
        model.train()
        imgL, imgR, disp = torch.Tensor(imgL).cuda(), torch.Tensor(
            imgR).cuda(), torch.Tensor(disp).cuda()

        # do not consider disparity out of limit
        mask = disp < max_disparity
        mask.detach_()

        optimizer.zero_grad()

        out1, out2, out3 = model(imgL, imgR)
        out1 = torch.squeeze(out1, 1)
        out2 = torch.squeeze(out2, 1)
        out3 = torch.squeeze(out3, 1)
        loss = 0.5 * F.smooth_l1_loss(out1[mask], disp[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            out2[mask], disp[mask], size_average=True) + F.smooth_l1_loss(out3[mask], disp[mask], size_average=True)

        loss.backward()
        optimizer.step()
        return loss.data

    # begin training
    print("Training begin")
    for epoch in range(1, epochs+1):
        print("Epoch {}:".format(epoch))
        total = 0
        for batch_id, (imgL_c, imgR_c, disp_c) in enumerate(Loader):
            batch_begin = time.time()
            loss = train(imgL_c, imgR_c, disp_c)
            total += loss
            print(" batch {}: training loss = {}, time = {}".format(
                batch_id, loss, time.time()-batch_begin))

        # save the model after an epoch
        save_name = save_path + '/checkpoint_' + str(epoch) + '.tar'
        torch.save(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total / len(Loader)
            }, save_name
        )
    print("Training complete")
