import torch
import cv2
import torch.nn.functional as F
from torchvision import transforms
from gaze_net.resnet import resnet18

# gaze360 decode config: 90 bins, each 4 degrees wide, spanning -180 to 180
BINS, BINWIDTH, ANGLE, DEVICE = 90, 4, 180, "mps"

# built once by load() and reused every frame (rebuilding per frame would be very slow)
_model = None
_idx = None
_tf = transforms.Compose([
    transforms.ToPILImage(), # transforms need a PIL image, not a numpy array
    transforms.Resize(448), # size that resnet18 was trained on
    transforms.ToTensor(), # HxWxC [0,255] -> CxHxW float [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # standards for this model
])

def load(weight="models/resnet18.pt"):
    global _model, _idx
    _model = resnet18(pretrained=False, num_classes=BINS) # build arch, no imagenet weights (we load our own)
    _model.load_state_dict(torch.load(weight, map_location=DEVICE)) # load the trained weights into the arch
    _model.eval().to(DEVICE) # eval = inference mode (batchnorm/dropout off); to(DEVICE) moves to gpu
    _idx = torch.arange(BINS, device=DEVICE, dtype=torch.float32) # [0,1,...,89], bin indices used in the decode
    with torch.no_grad():
        _model(torch.zeros(1, 3, 448, 448, device=DEVICE)) # warm-up: first mps call compiles kernels (slow), do it now
    return _model


def gaze(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB) # cv2 gives BGR, model wants RGB
    x = _tf(rgb).unsqueeze(0).to(DEVICE) # preprocess, add batch dim -> [1,3,448,448], moves to gpu
    with torch.no_grad(): # inference only, no gradients needed
        yaw, pitch = _model(x) # two [1,90] bin scores (order is yaw, pitch)
    # decode bins -> angle: softmax to probabilities, weight by bin index, sum = expected bin, scale to degrees
    yaw = (F.softmax(yaw, dim=1) * _idx).sum(dim=1) * BINWIDTH - ANGLE
    pitch = (F.softmax(pitch, dim=1) * _idx).sum(dim=1) * BINWIDTH - ANGLE
    return torch.deg2rad(pitch).item(), torch.deg2rad(yaw).item() # radians, order (pitch, yaw)


