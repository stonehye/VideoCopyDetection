import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from nets.pooling import L2N, GeM, RMAC
import numpy as np
from collections import OrderedDict


class BaseModel(nn.Module):
    def __str__(self):
        return self.__class__.__name__

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            output = self.__class__.__name__ + "\n"
            output += self._summary(self, input_size, batch_size, device)
            return output
        except:
            return self.__repr__()

    @staticmethod
    def _summary(model, input_size, batch_size=-1, device="cuda"):
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size

                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size
                params = 0

                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad

                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))

                summary[m_key]["nb_params"] = params

            if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)
                    and not (module == model)):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if isinstance(input_size, tuple):
            input_size = [input_size]

        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        # print(type(x[0]))

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)
        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()
        output = "---------------------------------------------------------------------------------------\n"
        output += "{:^30}{:^30}{:^15}{:^5}\n".format("Layer (type)", "Output Shape", "Param #", "Grad")
        output += "=======================================================================================\n"
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]

            output += "{:^30}{:^30}{:^15}{:^5}\n".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
                "True" if summary[layer].get("trainable") else "False"
            )

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        output += "=======================================================================================\n"
        output += "Total params: {0:,}\n".format(total_params)
        output += "Trainable params: {0:,}\n".format(trainable_params)
        output += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
        output += "---------------------------------------------------------------------------------------\n"
        output += "Input size (MB): %0.2f\n" % total_input_size
        output += "Forward/backward pass size (MB): %0.2f\n" % total_output_size
        output += "Params size (MB): %0.2f\n" % total_params_size
        output += "Estimated Total Size (MB): %0.2f\n" % total_size
        output += "---------------------------------------------------------------------------------------"

        return output


class MobileNet(BaseModel):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.base = models.mobilenet_v2(pretrained=True)

    def forward(self, x):
        return self.base(x)


class MobileNet_RMAC(BaseModel):
    def __init__(self):
        super(MobileNet_RMAC, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class MobileNet_AVG(BaseModel):
    def __init__(self):
        super(MobileNet_AVG, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class MobileNet_GeM(BaseModel):
    def __init__(self):
        super(MobileNet_GeM, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())]))
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class DenseNet(BaseModel):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.base = models.densenet121(pretrained=True)

    def forward(self, x):
        return self.base(x)

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return super().summary(input_size, batch_size, device)
        except:
            return nn.Module.__repr__()


class DenseNet_RMAC(BaseModel):
    def __init__(self):
        super(DenseNet_RMAC, self).__init__()

        self.base = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                  nn.ReLU(inplace=True))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class DenseNet_AVG(BaseModel):
    def __init__(self):
        super(DenseNet_AVG, self).__init__()

        self.base = nn.Sequential(*list(models.densenet121(pretrained=True).features.children()),
                                  nn.ReLU(inplace=True))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class DenseNet_GeM(BaseModel):
    def __init__(self):
        super(DenseNet_GeM, self).__init__()
        self.base = nn.Sequential(OrderedDict([*list(models.densenet121(pretrained=True).features.named_children())] +
                                              [('relu', nn.ReLU(inplace=True))]))
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class Resnet50_RMAC(BaseModel):
    def __init__(self):
        super(Resnet50_RMAC, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))

        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x


class Resnet50_AVG(BaseModel):
    def __init__(self):
        super(Resnet50_AVG, self).__init__()

        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class TripletNet(BaseModel):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, *x, single=False):
        if single:
            return self.forward_single(x[0])
        else:
            return self.forward_triple(x[0], x[1], x[2])

    def forward_single(self, x):
        output = self.embedding_net(x)
        return output

    def forward_triple(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    def __str__(self):
        return f'[{super(TripletNet, self).__str__()}]{self.embedding_net.__str__()}'


class Segment_Maxpooling(BaseModel):
    def __init__(self):
        super(Segment_Maxpooling, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.norm = L2N()

    def forward(self, x):
        x = self.pool(x).squeeze(-1)
        x = self.norm(x)
        return x


class Local_Maxpooling(BaseModel):
    def __init__(self, group_count):
        super(Local_Maxpooling, self).__init__()
        self.pool = torch.nn.MaxPool1d(group_count)
        self.norm = L2N()

    def forward(self, x):
        x = x.permute(1,2,0)
        x = self.pool(x)
        x = x.permute(2, 0, 1)
        x = self.norm(x)
        return x


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(eps={self.eps})'


class PointwiseConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        out = self.pointwise(x)
        return out


class MobileNet_local(nn.Module):
    def __init__(self):
        super(MobileNet_local, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())])
        )
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.norm(x)
        x = x.reshape(-1, x.shape[1], x.shape[2]*x.shape[2]).squeeze(-1)
        return x


class VGGNet16(BaseModel):
    def __init__(self):
        super(VGGNet16, self).__init__()
        self.base = nn.Sequential(*list(models.vgg16(pretrained=True).features)[:-6][:-2])  # conv5_1
        self.norm = L2N()
    def forward(self, x):
        x = self.base(x)
        return x


class Resnet50_local(BaseModel):
    def __init__(self):
        super(Resnet50_local, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.norm(x)
        x = x.reshape(-1, x.shape[1], x.shape[2] * x.shape[2]).squeeze(-1)
        return x

class Resnet50_intermediate(BaseModel):
    def __init__(self):
        super(Resnet50_intermediate, self).__init__()
        # self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-3]
        #                                       + [('layer4', nn.Sequential(list(models.resnet50(pretrained=True).named_children())[7][1][0]))])) # conv5_1

        # self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-4]
        #                                       + [('layer3', nn.Sequential(
        #     list(models.resnet50(pretrained=True).named_children())[6][1][:2]))])) # conv4_2

        # self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-4]
        #                                       + [('layer3', nn.Sequential(
        #     list(models.resnet50(pretrained=True).named_children())[6][1][:4]))]))  # conv4_4

        # self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-3])) # conv4_6

        # self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-4])) # conv3_4

        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-4]
                                              + [('layer3', nn.Sequential(
            list(models.resnet50(pretrained=True).named_children())[6][1][:5]))]))  # conv4_5

        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.norm(x)
        x = x.reshape(-1, x.shape[1], x.shape[2] * x.shape[2]).squeeze(-1)
        return x


if __name__ == "__main__":
    m = Resnet50_intermediate()
    print(m.summary((3,224,224), device='cpu'))