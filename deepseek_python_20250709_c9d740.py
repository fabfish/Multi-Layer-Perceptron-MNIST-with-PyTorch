import argparse
import torch
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import matplotlib.pyplot as plt
import os
from itertools import product

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Unified Sparse Neural Network Experiment')
    
    # 稀疏方式选择 (可以多选)
    parser.add_argument('--use-bi-mask', action='store_true',
                       help='Use bi-directional mask in backward pass')
    parser.add_argument('--use-self-update', action='store_true',
                       help='Apply mask in forward pass (self-update)')
    parser.add_argument('--use-post-update', action='store_true',
                       help='Apply mask after weight update (post-update)')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='Batch size for training')
    
    # 稀疏参数
    parser.add_argument('--n', type=int, default=2,
                       help='N in N:M sparsity (default: 2)')
    parser.add_argument('--m', type=int, default=4,
                       help='M in N:M sparsity (default: 4)')
    parser.add_argument('--decay', type=float, default=0.0002,
                       help='Decay rate for masked weights')
    
    # 实验选项
    parser.add_argument('--grid-search', action='store_true',
                       help='Perform grid search on learning rates')
    parser.add_argument('--ablation', action='store_true',
                       help='Perform ablation study on sparse methods')
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    
    return parser.parse_args()

# 数据加载
def load_data(batch_size):
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
    
    return train_loader, test_loader

# 稀疏矩阵生成
def get_n_m_sparse_matrix(w, N=2, M=4):
    length = w.numel()
    group = int(length / M)
    w_tmp = w.t().detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask

# 神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return x


# 统一的稀疏函数
class UnifiedSparseFunction(autograd.Function):
    @staticmethod
    def forward(ctx, weight, inp_unf, forward_mask, backward_mask, use_bi_mask, decay):
        ctx.save_for_backward(weight, inp_unf,)
        ctx.use_bi_mask = use_bi_mask
        ctx.decay = decay
        ctx.mask = forward_mask
        
        # 应用前向稀疏
        w_s = weight * forward_mask
        out_unf = inp_unf.matmul(w_s)
        
        return out_unf

    @staticmethod
    def backward(ctx, g):
        weight, inp_unf,  = ctx.saved_tensors
        
        # 计算权重梯度
        g_w_s = inp_unf.t().matmul(g)
        g_w_s = g_w_s + ctx.decay * (1 - ctx.mask) * weight
        
        if ctx.use_bi_mask:
            weight, _ = get_n_m_sparse_matrix(weight,)
        
        g_inp_unf = g.matmul(weight.t())
        
        return g_w_s, g_inp_unf, None, None, None, None

# 统一的稀疏线性层
class UnifiedSparseLinear(nn.Linear):
    def __init__(self, use_bi_mask, use_self_update, use_post_update, N, M, decay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_bi_mask = use_bi_mask
        self.use_self_update = use_self_update
        self.use_post_update = use_post_update
        self.N = N
        self.M = M
        self.decay = decay
        
        # 初始化掩码
        self.forward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)
        if self.use_bi_mask:
            self.backward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)

    def forward(self, x):
        w = self.weight.view(self.weight.size(0), -1).t()
        
        # 生成稀疏掩码
        w_s, self.forward_mask = get_n_m_sparse_matrix(w, self.N, self.M)
        
        # 如果使用self-update，直接在forward中应用稀疏
        if self.use_self_update:
            self.weight.data = w_s.t().view(self.weight.size(0), -1)
        
        # 应用稀疏前向传播
        out_unf = UnifiedSparseFunction.apply(
            w, x, 
            self.forward_mask, 
            self.backward_mask if self.use_bi_mask else None,
            self.use_bi_mask,
            self.decay
        )
        
        return out_unf


# 训练和评估函数
def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    train_loss = 0.0
    
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 如果使用post-update，在权重更新后应用稀疏
        if args.use_post_update:
            for name, param in model.named_parameters():
                if 'weight' in name:
                    param.data, _ = get_n_m_sparse_matrix(param, args.n, args.m)
        
        train_loss += loss.item() * data.size(0)
    
    train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f}')
    return train_loss

def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            
            for i in range(target.size(0)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    
    print(f'Test Loss: {test_loss:.6f}')
    print(f'Test Accuracy (Overall): {accuracy:.2f}% ({np.sum(class_correct)}/{np.sum(class_total)})')
    
    return test_loss, accuracy

# 运行单个实验
def run_experiment(args, lr=None):
    if lr is not None:
        args.lr = lr
    
    print("\n=== Running Experiment ===")
    print(f"Sparse Methods: {'Bi-Mask' if args.use_bi_mask else ''} {'Self-Update' if args.use_self_update else ''} {'Post-Update' if args.use_post_update else ''}")
    print(f"Learning Rate: {args.lr}")
    print(f"N:M Sparsity: {args.n}:{args.m}")
    
    # 加载数据
    train_loader, test_loader = load_data(args.batch_size)
    
    # 初始化模型
    model = Net()
    model.load_state_dict(torch.load('model_mnist_mlp.pth'))
    
    # 替换线性层为统一的稀疏线性层
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            setattr(model, name, UnifiedSparseLinear(
                args.use_bi_mask, args.use_self_update, args.use_post_update,
                args.n, args.m, args.decay,
                module.in_features, module.out_features, 
                bias=module.bias is not None
            ))
            with torch.no_grad():
                getattr(model, name).weight.copy_(module.weight)
                if module.bias is not None:
                    getattr(model, name).bias.copy_(module.bias)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    
    # 训练和评估
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, epoch, args)
    
    test_loss, accuracy = evaluate(model, test_loader, criterion)
    
    # 保存结果
    method_str = ""
    if args.use_bi_mask:
        method_str += "bi_"
    if args.use_self_update:
        method_str += "self_"
    if args.use_post_update:
        method_str += "post_"
    method_str = method_str[:-1] if method_str else "none"
    
    result_file = os.path.join(args.save_dir, f"results_{method_str}_lr{args.lr}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Sparse Methods: {method_str}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"N:M Sparsity: {args.n}:{args.m}\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
    
    return accuracy

# 学习率网格搜索
def grid_search(args):
    lr_list = [0.1, 0.01, 0.001, 0.0001]
    best_accuracy = 0
    best_lr = 0
    
    for lr in lr_list:
        accuracy = run_experiment(args, lr)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
    
    print(f"\nBest learning rate: {best_lr} with accuracy: {best_accuracy:.2f}%")

# 消融实验
def ablation_study(args):
    # 所有可能的稀疏方法组合
    methods = [
        {'use_bi_mask': True, 'use_self_update': False, 'use_post_update': False},
        {'use_bi_mask': False, 'use_self_update': True, 'use_post_update': False},
        {'use_bi_mask': False, 'use_self_update': False, 'use_post_update': True},
        {'use_bi_mask': True, 'use_self_update': True, 'use_post_update': False},
        {'use_bi_mask': True, 'use_self_update': False, 'use_post_update': True},
        {'use_bi_mask': False, 'use_self_update': True, 'use_post_update': True},
        {'use_bi_mask': True, 'use_self_update': True, 'use_post_update': True},
    ]
    
    results = []
    
    for method in methods:
        # 更新参数
        args.use_bi_mask = method['use_bi_mask']
        args.use_self_update = method['use_self_update']
        args.use_post_update = method['use_post_update']
        
        # 运行实验
        accuracy = run_experiment(args)
        results.append((method, accuracy))
    
    # 打印结果
    print("\n=== Ablation Study Results ===")
    for method, accuracy in results:
        method_str = []
        if method['use_bi_mask']:
            method_str.append("Bi-Mask")
        if method['use_self_update']:
            method_str.append("Self-Update")
        if method['use_post_update']:
            method_str.append("Post-Update")
        
        print(f"Methods: {', '.join(method_str) if method_str else 'None'} - Accuracy: {accuracy:.2f}%")

# 主函数
def main():
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查是否至少选择了一种稀疏方法
    if not (args.use_bi_mask or args.use_self_update or args.use_post_update):
        print("Warning: No sparse method selected. Running baseline model.")
    
    # 执行网格搜索
    if args.grid_search:
        grid_search(args)
        return
    
    # 执行消融实验
    if args.ablation:
        ablation_study(args)
        return
    
    # 执行单个实验
    run_experiment(args)

if __name__ == '__main__':
    main()