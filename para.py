from torchstat import stat
from libraries.CRCNet.models.CRCNet import CRCNetModel
from config import CONFIG

model = CRCNetModel(CONFIG)
stat(model, (3, 256, 256))