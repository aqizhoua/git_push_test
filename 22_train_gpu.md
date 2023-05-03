- 可以直接调用.cuda()函数，将对应操作转移到gpu上
  - 网络模型
  - 数据（输入、标注）
  - 损失函数

- 使用.to(device)
  - 第一部要实例化device
    - device = torch.device("mps")
    - torch.device("cpu")/torch.device("cuda")/torch.device("mps) 
    - 如果有多张显卡，可以使用torch.device("cuda:0"),torch.device("cuda:1")
      -如果只有一张显卡，则cuda和cuda:0两种写法等价
  - 第二部使用.to(device)
    - model = model.to(device)
  - 需要注意的是，这种方式下，model,loss_function可以不需要再重新赋值,而outputs和targets还是需要的
    - 即model.to(device),而不是像第一种：model=model.to(device)
    - 

在终端terminal输入nvidia-smi可以显示gpu占用信息

