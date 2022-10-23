- torch.nn
  - Containers
    - Module
      - Base class for all neural network modules.
      - 即所有构建神经网络的类都要继承它
  - Convolution Layers
    - nn.Conv1d 一维卷积
      - 对由多个输入平面组成的输入信号应用一维卷积。
    - nn.Conv2d 二维卷积 （最常用）
      - 对由多个输入平面组成的输入信号应用二维卷积。
    - nn.Conv3d 三维卷积
      - 对由多个输入平面组成的输入信号应用三维卷积。
    - ...
    
  - torch.nn和torch.nn.functional的关系
    - 前者是后者的封装
    - 后者注重实现细节，前者注重使用
    - 在学习原理时注重后者，使用时注重前者
  
