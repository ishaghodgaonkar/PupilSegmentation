from utils import PupilDataset
import torch.optim as optim
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp

# Initalize U-Net
model = smp.Unet(in_channels=1)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])


# Initalize trainloader
trainset = PupilDataset(root_dir='PupilDataset', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                          shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-4)


# source: https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


for epoch in range(0, 10):  # loop over the dataset multiple times
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True)
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        print(i, len(data[0]), len(data[1]))
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data

        # zero the parameter gradients
        optimizer.zero_grad()
        print(inputs.shape)
        # forward + backward + optimize
        outputs = model(inputs)

        loss = dice_loss(outputs, targets)
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

torch.save(model.state_dict(), 'model_1epoch.pth')
torch.save(model, 'model_1epoch_whole.pth')