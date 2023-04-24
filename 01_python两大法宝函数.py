import torch

#help
#dir

#命令行查看英伟达驱动程序版本号：nvidia-smi


"""
要更新GTX 1050的英伟达驱动，请按照以下步骤进行操作：


打开英伟达官方网站（https://www.nvidia.com/Download/index.aspx）并选择“驱动程序”选项卡。
在“产品类型”下拉菜单中选择“GeForce”。
在“产品系列”下拉菜单中选择“GeForce 10 Series”。
在“产品”下拉菜单中选择“GeForce GTX 1050”。
在“操作系统”下拉菜单中选择您的操作系统。
点击“搜索”按钮以查找适用于您的系统的最新驱动程序。
下载驱动程序并运行安装程序。
安装程序将指导您完成驱动程序的安装过程。

请注意，在安装新驱动程序之前，最好先卸载旧驱动程序。这可以通过打开Windows设备管理器，找到GTX 1050的驱动程序，右键单击并选择“卸载设备”来完成。
"""
print(torch.device("mps"))
print(torch.decice("cpu"))
# print(torch.device("cuda"))

