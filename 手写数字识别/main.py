
import torch.utils.data
import torchvision
from torch import nn
from tensorboardX import SummaryWriter
from modle import FewShot

# 创建TensorBoard的SummaryWriter对象
writer = SummaryWriter()


if torch.cuda.is_available():
    print("cuda is available")
    device = torch.device("cuda")
else:
    print("cuda is not available")
    device = torch.device("cpu")

data_train = torchvision.datasets.MNIST(
    "./dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
data_test = torchvision.datasets.MNIST(
    "./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

data_train_size = len(data_train)
data_test_size = len(data_test)
print("------训练集大小{}------".format(data_train_size))
print("------测试集大小{}------".format(data_test_size))

dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=128)
dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=128)




few_shot = FewShot().to(device)

# 损失函数
loss_fun = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(few_shot.parameters(), lr=learning_rate)
# 训练轮数
Epoch = 10

# 记录训练次数
train_step = 0
test_step = 0


# 训练
from tensorboardX import SummaryWriter

# 创建TensorBoard的SummaryWriter对象
writer = SummaryWriter()

for epoch in range(Epoch):
    print("------第{}轮训练开始------".format(epoch + 1))

    # 训练步骤
    few_shot.train()
    train_accuracy = 0
    for data in dataloader_train:
        imgs, label = data[0].to(device), data[1].to(device)
        output = few_shot(imgs)
        loss = loss_fun(output, label)
        writer.add_scalar('Loss', loss, train_step)
        writer.add_image('Input Image', imgs[0], train_step)

        for name, param in few_shot.named_parameters():
            writer.add_histogram(name, param, train_step)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step += 1

        accuracy = (output.argmax(1) == label).sum()
        train_accuracy += accuracy
        if train_step % 100 == 0:
            print("第{}次训练，LOSS值为：{}".format(train_step, loss.item()))

    # 测试
    few_shot.eval()
    loss_test = 0
    test_accuracy = 0
    with torch.no_grad():
        for data in dataloader_test:
            imgs, label = data[0].to(device), data[1].to(device)
            output = few_shot(imgs)
            loss = loss_fun(output, label)
            loss_test += loss.item()
            accuracy = (output.argmax(1) == label).sum()
            test_accuracy += accuracy

    test_step += 1
    print("第{}轮测试，LOSS值为：{}".format(epoch + 1, loss_test))

    print("第{}轮训练，准确率为：{}".format(epoch + 1, train_accuracy / data_train_size))
    print("第{}轮测试，准确率为：{}".format(epoch + 1, test_accuracy / data_test_size))

    model_state_dict = few_shot.state_dict()

    # 保存状态字典到文件
    torch.save(model_state_dict, "model_state_dict_{}.pth".format(epoch))

    print("模型状态字典已保存")

# 关闭SummaryWriter对象
writer.close()