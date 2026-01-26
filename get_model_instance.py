from torch import nn
from torchvision.models import convnext_base, efficientnet_b0, vgg16, resnet50

from model.GFNet import GFNet
from model.GhostNetV3 import ghostnetv3
from model.MobileNetV3_Tea_0MB import MobileNetV3_Tea_0MB
from model.MobileNetV3_Tea_1MB import MobileNetV3_Tea_1MB
from model.MobileNetV3_Tea_2MB import MobileNetV3_Tea_2MB
from model.MobileNetV3_Tea_3MB import MobileNetV3_Tea_3MB
from model.MobileNetV3_Tea_4MB import MobileNetV3_Tea_4MB
from model.MobileNetV3_Tea_5MB import MobileNetV3_Tea_5MB
from model.MobileNetV3_Tea_6MB import MobileNetV3_Tea_6MB
from model.MobileNetV3_Tea_7MB import MobileNetV3_Tea_7MB
from model.MobileNetV3_Tea_8MB import MobileNetV3_Tea_8MB
from model.MobileNetV3_Tea_9MB import MobileNetV3_Tea_9MB
from model.MobileNetV3_Tea_s1_0MB import MobileNetV3_Tea_s1_0MB
from model.MobileNetV3_Tea_s1_1MB import MobileNetV3_Tea_s1_1MB
from model.MobileNetV3_Tea_s1_2MB import MobileNetV3_Tea_s1_2MB
from model.MobileNetV3_Tea_s1_3MB import MobileNetV3_Tea_s1_3MB
from model.MobileNetV3_Tea_s1_4MB import MobileNetV3_Tea_s1_4MB
from model.MobileNetV3_Tea_s1_5MB import MobileNetV3_Tea_s1_5MB
from model.MobileNetV3_Tea_s1_6MB import MobileNetV3_Tea_s1_6MB
from model.MobileNetV3_Tea_s1_7MB import MobileNetV3_Tea_s1_7MB
from model.MobileNetV3_Tea_s1_8MB import MobileNetV3_Tea_s1_8MB
from model.MobileNetV3_Tea_s1_9MB import MobileNetV3_Tea_s1_9MB
from model.MobileNetV3_Tea_s1s2_0MB import MobileNetV3_Tea_s1s2_0MB
from model.MobileNetV3_Tea_s1s2_1MB import MobileNetV3_Tea_s1s2_1MB
from model.MobileNetV3_Tea_s1s2_2MB import MobileNetV3_Tea_s1s2_2MB
from model.MobileNetV3_Tea_s1s2_3MB import MobileNetV3_Tea_s1s2_3MB
from model.MobileNetV3_Tea_s1s2_4MB import MobileNetV3_Tea_s1s2_4MB
from model.MobileNetV3_Tea_s1s2_5MB import MobileNetV3_Tea_s1s2_5MB
from model.MobileNetV3_Tea_s1s2_6MB import MobileNetV3_Tea_s1s2_6MB
from model.MobileNetV3_Tea_s1s2_7MB import MobileNetV3_Tea_s1s2_7MB
from model.MobileNetV3_Tea_s1s2_8MB import MobileNetV3_Tea_s1s2_8MB
from model.MobileNetV3_Tea_s1s2_9MB import MobileNetV3_Tea_s1s2_9MB
from model.Mobilenetv4 import mobilenetv4_small
from model.ResMLP import ResMLP
from model.StartNet import starnet_s1
from model.SwinTransformer import SwinTransformer
from model.coatnet import coatnet_0
from model.mobilenet_v3 import MobileNetV3_Small
from model.InceptionNeXt import inceptionnext_small

def get_model_instance(model_name, args):
    #对比试验
    if model_name == 'coatnet_0':
        model = coatnet_0()
    elif model_name == 'convnext_base':
        model = convnext_base(pretrained=False)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, args.class_num)
    elif model_name == 'efficientnet_b0':
        model = efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.class_num)
    elif model_name == 'ghostnetv3':
        model = ghostnetv3(pretrained=False, width=1.0)
    elif model_name == 'inceptionnext_small':
        model = inceptionnext_small()
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.class_num)
    elif model_name == 'vgg16':
        model = vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, args.class_num)
    elif model_name == 'mobilenetv4_small':
        model = mobilenetv4_small(pretrained=False, num_classes=args.class_num)
    elif model_name == 'mobilenetv3_small':
        model = MobileNetV3_Small(num_classes=args.class_num)
    elif model_name == 'starnet_s1':
        model = starnet_s1(pretrained=False, num_classes=args.class_num)
    elif model_name == 'resmlp':
        model = ResMLP(in_channels=3, image_size=224, patch_size=16, num_classes=args.class_num,
                     dim=384, depth=12, mlp_dim=384*4)
    elif model_name == 'gfnet':
        model = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=args.class_num)
    elif model_name == 'swintransformer':
        model = SwinTransformer(num_classes=args.class_num)

    #消融实验
    elif model_name == 'mobilenetv3_tea_0mb':
        model = MobileNetV3_Tea_0MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_1mb':
        model = MobileNetV3_Tea_1MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_2mb':
        model = MobileNetV3_Tea_2MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_3mb':
        model = MobileNetV3_Tea_3MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_4mb':
        model = MobileNetV3_Tea_4MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_5mb':
        model = MobileNetV3_Tea_5MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_6mb':
        model = MobileNetV3_Tea_6MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_7mb':
        model = MobileNetV3_Tea_7MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_8mb':
        model = MobileNetV3_Tea_8MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_9mb':
        model = MobileNetV3_Tea_9MB(num_classes=args.class_num)

    elif model_name == 'mobilenetv3_tea_s1_0mb':
        model = MobileNetV3_Tea_s1_0MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_1mb':
        model = MobileNetV3_Tea_s1_1MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_2mb':
        model = MobileNetV3_Tea_s1_2MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_3mb':
        model = MobileNetV3_Tea_s1_3MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_4mb':
        model = MobileNetV3_Tea_s1_4MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_5mb':
        model = MobileNetV3_Tea_s1_5MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_6mb':
        model = MobileNetV3_Tea_s1_6MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_7mb':
        model = MobileNetV3_Tea_s1_7MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_8mb':
        model = MobileNetV3_Tea_s1_8MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1_9mb':
        model = MobileNetV3_Tea_s1_9MB(num_classes=args.class_num)


    elif model_name == 'mobilenetv3_tea_s1s2_0mb':
        model = MobileNetV3_Tea_s1s2_0MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_1mb':
        model = MobileNetV3_Tea_s1s2_1MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_2mb':
        model = MobileNetV3_Tea_s1s2_2MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_3mb':
        model = MobileNetV3_Tea_s1s2_3MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_4mb':
        model = MobileNetV3_Tea_s1s2_4MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_5mb':
        model = MobileNetV3_Tea_s1s2_5MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_6mb':
        model = MobileNetV3_Tea_s1s2_6MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_7mb':
        model = MobileNetV3_Tea_s1s2_7MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_8mb':
        model = MobileNetV3_Tea_s1s2_8MB(num_classes=args.class_num)
    elif model_name == 'mobilenetv3_tea_s1s2_9mb':
        model = MobileNetV3_Tea_s1s2_9MB(num_classes=args.class_num)

    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(args.device)