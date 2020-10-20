import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

xor_data = [
    [[1, 1], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[0, 0], 0]
]


class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 2)
        self.linear1.weight.data = torch.tensor([[0.1, -0.1], [-0.1, 0.1]])
        self.linear2.weight.data = torch.tensor([[-0.1, 0.1], [0.1, -0.1]])
        self.linear1.bias.data = torch.tensor([0.1, -0.1])
        self.linear2.bias.data = torch.tensor([-0.1, 0.1])

    def forward(self, x):
        out1 = self.linear1(x)
        out1 = torch.sigmoid(out1)
        out2 = self.linear2(out1)
        out2 = torch.sigmoid(out2)
        return out2


epoch = 600
log_interval = 100
learning_rate = 1

model = XORNet()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
writer = SummaryWriter()

for e_index in range(1, epoch+1):
    for index, data in enumerate(xor_data):
        _input = torch.tensor(data[0]).to(torch.float)
        target = torch.zeros(2).scatter_(0, torch.tensor(data[1]), 1).to(torch.float)

        output = model(_input)
        loss = F.binary_cross_entropy(output, target)
        optimizer.zero_grad()
        if e_index == 1 and index == 0:
            print(f"\nweight(before): {model.state_dict()}")

        loss.backward()
        optimizer.step()

        if e_index == 1 and index == 0:
            writer.add_graph(model, _input)

            print(f"\nweight(after): {model.state_dict()}]\n")
            for name, parms in model.named_parameters():
                print(f"{name} grad ({parms.requires_grad}): {parms.grad}")

        if not e_index % log_interval:
            print(
                f"epoch: {e_index} input: {_input.data} output: {output.data}, target: {target.data} loss: {loss.data}")

writer.close()
