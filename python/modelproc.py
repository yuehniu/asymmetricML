import sys
import torch
import torchvision.transforms as transforms
sys.path.insert(0, './')
from python.utils import build_network, infer_memory_size, init_model
from python.models.resnet import BasicBlock

model = build_network('resnet20', num_classes=10)

print( model )
"""
for m in model.children():
    if isinstance( m, torch.nn.Sequential ):
        for mm in m.children():
            if isinstance( mm, BasicBlock ):
                for mmm in mm.children():
                    print( mmm )
                    print("-----------")
            else:
                print(mm)
                print("-----------")
    else:
        print( m )
        print("-----------")
"""
