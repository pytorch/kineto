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

def export_graph(log_dir, model, inputs, text_format=True):
    import os
    os.makedirs(log_dir, exist_ok=True)

    if text_format:
        from torch.utils.tensorboard._pytorch_graph import graph
        from google.protobuf import text_format
        graph_def, _ = graph(model, inputs)
        with open(os.path.join(log_dir, 'model_graph.txt'), 'w') as f:
            f.write(text_format.MessageToString(graph_def))
    else:
        import torch.utils.tensorboard as tb
        with tb.SummaryWriter(log_dir) as graph_writer:
            graph_writer.add_graph(model, inputs)


model = models.resnet50(pretrained=True)
model.cuda()
cudnn.benchmark = True

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0")
model.train()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    model_inputs = None
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)
        if model_inputs is None:
            model_inputs = inputs

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 >= 4:
            break
        p.step()
    export_graph('./result', model, model_inputs, text_format=False)
    export_graph('./result', model, model_inputs, text_format=True)