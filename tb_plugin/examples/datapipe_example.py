import torch
import torch.nn as nn
import torch.optim
from torch.utils.data.dataloader_experimental import DataLoader2

from torchvision import transforms as T
import torchvision.prototype.datasets as pdatasets
import torchvision.prototype.models as models
from torchvision.prototype.datasets._builtin import Cifar10


if __name__ == "__main__":
    model = models.resnet50(models.ResNet50_Weights.ImageNet1K_V1)
    trainset = Cifar10().to_datapipe(root='./data', decoder=pdatasets.decoder.raw)
    transform = T.Compose([T.Resize(256), T.CenterCrop(224)])
    trainset = trainset.map(transform, input_col="image")
    trainset = trainset.map(fn=T.functional.convert_image_dtype, input_col="image")
    dl = DataLoader2(trainset, batch_size=64)
    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0")
    model.to(device=device).train()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='datapipe0'),
        record_shapes=True,
        profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
        with_stack=True
    ) as p:
        for step, data in enumerate(dl, 0):
            print("step:{}".format(step))
            input_tensors = data['image']
            label_tensors = data['label']
            inputs, labels = input_tensors.to(device=device), label_tensors.to(device=device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step + 1 >= 4:
                break
            p.step()
        print("done")
