import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models

import torch.profiler

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

def output_fn(p):
    try:
        os.mkdir("result")
    except Exception as e:
        pass
    p.export_chrome_trace("./result/worker0.pt.trace.json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=3,
        active=6),
    on_trace_ready=output_fn,
    record_shapes=True
) as p:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 >= 11:
            break
        p.step()
