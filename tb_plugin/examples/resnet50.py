import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models

from torch.autograd.profiler import profile
from torch.autograd import kineto_available

assert(kineto_available())

model = models.resnet50(pretrained=True)
model.cuda()
cudnn.benchmark = True

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0")

model.train()

with profile(use_cuda=True, use_kineto=True, record_shapes=True) as p:
    for _epoch in range(1):
        running_loss = 0.0
        count = 0
        for _i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device=device), data[1].to(device=device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            print("step:", count)
            if count > 5:
                break


try:
    os.mkdir("result")
except Exception:
    pass

p.export_chrome_trace("./result/worker0.pt.trace.json")
