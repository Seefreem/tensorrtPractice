
'''
Convert the pretrained image segmentation PyTorch model into ONNX
'''
import torch
from torch.autograd import Variable
import torch.onnx as torch_onnx
import onnx
from torchvision.models.resnet import resnet18 as _resnet18
from unet import UNet

def main():
    input_shape = (3, 256, 256)
    model_onnx_path = "unet.onnx"
    dummy_input = Variable(torch.randn(1, *input_shape))
    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #   in_channels=3, out_channels=1, init_features=32, pretrained=True)
    # 当报错：AttributeError: 'collections.OrderedDict' object has no attribute 'eval'时：
    # https://stackoverflow.com/questions/49941426/attributeerror-collections-ordereddict-object-has-no-attribute-eval
    # 有说明pt 只包含模型的参数，所以需要载入模型文件，定义模型，然后加载参数：
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    model.load_state_dict(torch.load('unet-e012d006.pt'))
    model.eval()
    model.train(False)
    
    inputs = ['input.1']
    outputs = ['186']
    dynamic_axes = {'input.1': {0: 'batch'}, '186':{0:'batch'}}
    out = torch.onnx.export(model, dummy_input, model_onnx_path, input_names=inputs, output_names=outputs, dynamic_axes=dynamic_axes)


if __name__=='__main__':
    main()
    