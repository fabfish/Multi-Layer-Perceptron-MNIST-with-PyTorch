# import libraries
import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)

import matplotlib.pyplot as plt
# %matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)

## Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        return x


# load model weights from file
model = Net()
model.load_state_dict(torch.load('model_mnist_mlp.pth'))

criterion = nn.CrossEntropyLoss()

'''
Test Accuracy (Overall): 92% (9231/10000)
'''

N = 2
M = 4

def get_n_m_sparse_matrix(w):
    length = w.numel()
    group = int(length / M)
    w_tmp = w.t().detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask 

# bimask linear function
class maskedlinear(autograd.Function):
    @staticmethod
    def forward(ctx, weight, inp_unf, forward_mask,  decay = 0.0002):
        ctx.save_for_backward(weight, inp_unf, )
        w_s = weight * forward_mask

        ctx.decay = decay
        ctx.mask = forward_mask

        out_unf = inp_unf.matmul(w_s)
        return out_unf

    @staticmethod
    def backward(ctx, g):
        # print("maskedlinear backward")
        # print("g shape", g.shape)
        weight, inp_unf = ctx.saved_tensors
        # print("weight shape", weight.shape)
        # print("inp_unf shape", inp_unf.shape)
        # print("backward_mask shape", backward_mask.shape)
        w_s = weight.t() 
        g_w_s = inp_unf.t().matmul(g)
        g_w_s = g_w_s + ctx.decay * (1 - ctx.mask) * weight
        g_inp_unf = g.matmul(w_s)
        return g_w_s , g_inp_unf, None, None, None
    

class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = 2
        self.M = 4
        
        self.forward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)
        # self.backward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)

    def forward(self, x):
        w = self.weight.view(self.weight.size(0), -1).t()
        w_s, self.forward_mask = get_n_m_sparse_matrix(w)
        # self.backward_mask = self.forward_mask
        # inp_unf = self.unfold(x)
        inp_unf = x
        out_unf = maskedlinear.apply(w, inp_unf, self.forward_mask)
        # self.weight.data = w_s.t().view(self.weight.size(0), -1) # 这样是不对的，因为正向mask会更新，这一步还是应该放在反向检测。
        # 由于反向接口不能直接修改weight，所以需要在backward完成之后遍历模型手动完成修改。这样比较简便。后续，可以用backward hook来完成。
        # out = self.fold(out_unf.transpose(1, 2))
        out = out_unf
        return out
    


# switch all linear layers to maskedlinear for the original model
for name, module in model.named_modules():
    # import pdb; pdb.set_trace()
    if isinstance(module, nn.Linear):
    # if isinstance(module, nn.Linear) and not "fc3" in name:
        setattr(model, name, MaskedLinear(module.in_features, module.out_features, bias=module.bias is not None))
        # copy the weight and bias
        with torch.no_grad():
            getattr(model, name).weight.copy_(module.weight)
            if module.bias is not None:
                getattr(model, name).bias.copy_(module.bias)

'''
+ forward only mask
Test Accuracy (Overall): 85% (8556/10000)
'''

## Specify loss and optimization functions

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    print(model.fc1.weight)
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)

        # 此处，反向梯度已经更新到各个参数中，但是最终的权重还没算完
        optimizer.step()
        # 还是需要在这里，权重已经更新完成之后，再对权重做裁剪。
        for name, param in model.named_parameters():
            if 'fc' in name and 'weight' in name:
                param.data, _ = get_n_m_sparse_matrix(param)

        print("weight", model.fc1.weight)
                
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
    
    
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

'''
Epoch: 10       Training Loss: 0.313962
Test Loss: 0.315494

Test Accuracy of     0: 97% (959/980)
Test Accuracy of     1: 98% (1115/1135)
Test Accuracy of     2: 87% (904/1032)
Test Accuracy of     3: 89% (905/1010)
Test Accuracy of     4: 91% (898/982)
Test Accuracy of     5: 83% (747/892)
Test Accuracy of     6: 94% (909/958)
Test Accuracy of     7: 91% (944/1028)
Test Accuracy of     8: 86% (841/974)
Test Accuracy of     9: 89% (900/1009)

Test Accuracy (Overall): 91% (9122/10000)
'''