import numpy as np
from utils import PupilDataset
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from deepvog_utils import fit_ellipse_compact
from torchvision import transforms

# Initalize model
model = torch.load('model_1epoch-2.pth')

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Initalize test loader
testset = PupilDataset(root_dir='PupilDataset/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=True)
count = 0

# For all data in test set
for i, data in enumerate(testloader):
    count += 1
    inputs = data[0]
    masks = data[1]
    plt.imshow(inputs.squeeze().numpy())
    outputs = model(inputs)

    out = outputs.detach().squeeze().numpy()

    # Normalize segmentation map to (0,1)
    m = np.amin(out)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = -m + out[i][j]

    m = np.amin(out)
    s = np.sum(out)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] / s

    scale_factor = 1 / np.amax(out)

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] * scale_factor

    plt.imshow(out)
    ellipse = fit_ellipse_compact(out, threshold=0.3)
    print(ellipse)

    # Show fitted ellipse, pixel diameter of pupil
    plt.figure()
    ax = plt.gca()
    plt.imshow((inputs.squeeze().numpy()))
    ell = mpl.patches.Ellipse(tuple(ellipse[0]), ellipse[1], ellipse[2], ellipse[3])
    ax.add_artist(ell)
    plt.show()

