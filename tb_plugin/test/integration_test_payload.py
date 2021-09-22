import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as T
import torchvision.models as models

def get_train_func(use_gpu=True):
    model = models.resnet50(pretrained=False, progress=False)
    if use_gpu:
        model.cuda()
    cudnn.benchmark = True

    data_root = os.environ.get("TORCH_PROFILER_TEST_DATA_ROOT", "./data")
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                              shuffle=True, num_workers=0)

    if use_gpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model.train()

    def train(train_step, prof=None):
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            inputs, labels = data[0].to(device=device), data[1].to(device=device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if prof is not None:
                prof.step()
            if step >= train_step:
                break
    return train


if __name__ == "__main__":
    get_train_func(use_gpu=False)(0)
