import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image


def preprocess(img):
    im = cv2.resize(img, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0)
    return data


def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.numpy().transpose((1, 2, 0))
    inp = 127.5 + inp / 0.0078125
    inp = inp.astype("uint8")
    inp = inp[:, :, ::-1]
    return inp


class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True),
        )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 14 * 2, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 14 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        return x


class STNetInference:
    def __init__(self, weight_path, device=None):
        self.device = device
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        self.STN = STNet().to(self.device)
        self.STN.load_state_dict(
            torch.load(weight_path, map_location=lambda storage, loc: storage)
        )
        self.STN.eval()

    def __call__(self, image: np) -> str:
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()

        data = preprocess(cv_image).to(self.device)

        transfer = self.STN(data)
        transformed_input_tensor = transfer.cpu()
        out_grid = torchvision.utils.make_grid(transformed_input_tensor)
        out_grid = convert_image(out_grid.detach())

        # out_grid = cv2.cvtColor(out_grid, cv2.COLOR_BGR2RGB)
        out_grid = Image.fromarray(out_grid)

        return out_grid
