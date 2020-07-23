import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from deepvog_utils import fit_ellipse_compact
from torchvision import transforms
import segmentation_models_pytorch as smp
import cv2
from PIL import Image

# Initalize model

model = torch.load('model_1epoch-2.pth')

# Test transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


# For every 30 frames in demo video
vid = cv2.VideoCapture('demo.mp4')
count = 0
while True:
    count += 30
    vid.set(1, count)
    ret_val, img = vid.read()

    # Show image
    plt.imshow(img)
    img = Image.fromarray(img)
    img = transform(img)
    inputs = img.unsqueeze(0)
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

    # Show output
    out = cv2.resize(out, (320, 240))
    plt.imshow(out)

    ellipse = fit_ellipse_compact(out, threshold=0.3)

    # Show fitted ellipse, pixel diameter of pupil
    plt.figure()
    ax = plt.gca()
    img_resized = cv2.resize(img.squeeze().numpy(), (320, 240))
    plt.imshow(img_resized)
    ell = mpl.patches.Ellipse(tuple(ellipse[0]), ellipse[1], ellipse[2], ellipse[3])
    ax.add_artist(ell)
    plt.show()

    # Estimated diameter - average of height and width
    print("Diameter in pixels:", np.mean([ellipse[1], ellipse[2]]))