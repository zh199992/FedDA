class GHDR_FL(nn.Module):#卷积层 (nn.Conv2d)：默认使用Kaiming均匀分布初始化（He初始化）。
#循环神经网络层 (nn.GRU)：默认使用Xavier均匀分布初始化。
#全连接层 (nn.Linear)：默认使用Kaiming均匀分布初始化（He初始化），并且偏置项根据输入特征数量计算一个适当的边界值进行均匀分布初始化。
    def __init__(self,input_size,conv_init='kaiming_uniform', gru_init=None, linear_init='xavier_uniform'):
        super(GHDR_FL, self).__init__()
        self.mode='phase1'
        self.input_size=input_size
        self.filter_num=10
        self.filter_length=10
        self.F=nn.Sequential(
            conv_block(1,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num,self.filter_num,kernel_size=(self.filter_length,1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1))
        )
        self.LHDR=nn.Sequential(
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, self.filter_num, kernel_size=(self.filter_length, 1)),
            conv_block(self.filter_num, 1, kernel_size=(3, 1))
        )
        # self.unique=nn.Sequential(
        #     nn.GRU(input_size=self.input_size,hidden_size=50,num_layers=1,batch_first=True),
        #     LambdaLayer(lambda x: x[0][:, -1, :]),
        #     nn.Linear(50,700),
        #     # nn.Tanh(),
        #     Mish(),
        #     nn.Linear(700,200),
        #     # nn.Tanh(),
        #     Mish(),
        #     nn.Linear(200, 1),
        #
        # )
        self.unique = nn.Sequential(
            nn.GRU(input_size=self.input_size, hidden_size=50, num_layers=1, batch_first=True),
            LambdaLayer(lambda x: x[0][:, -1, :]),
            nn.Dropout(0.3),
            nn.Linear(50, 700),
            # nn.Tanh(),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(700, 200),
            # nn.Tanh(),
            Mish(),
            nn.Dropout(0.5),
            nn.Linear(200, 1),
        )
        conv_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if conv_init in conv_initializers:
            for module in self.LHDR.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)
            for module in self.F.modules():
                if isinstance(module, nn.Conv2d):
                    conv_initializers[conv_init](module.weight)

        # Initialize GRU layers (if specified)
        if gru_init is not None:
            for name, param in self.unique[0].named_parameters():
                if 'weight' in name:
                    if gru_init == 'xavier_uniform':
                        init.xavier_uniform_(param)
                    elif gru_init == 'xavier_normal':
                        init.xavier_normal_(param)
                    elif gru_init == 'kaiming_uniform':
                        init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'kaiming_normal':
                        init.kaiming_normal_(param, mode='fan_in', nonlinearity='leaky_relu')
                    elif gru_init == 'normal':
                        init.normal_(param, mean=0.0, std=0.02)
                    elif gru_init == 'uniform':
                        init.uniform_(param, a=-0.1, b=0.1)
                elif 'bias' in name:
                    init.constant_(param, 0)

        # Initialize Linear layers
        linear_initializers = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': partial(init.kaiming_uniform_, mode='fan_in', nonlinearity='leaky_relu'),
            'kaiming_normal': partial(init.kaiming_normal_, mode='fan_in', nonlinearity='leaky_relu'),
            'normal': partial(init.normal_, mean=0.0, std=0.02),
            'uniform': partial(init.uniform_, a=-0.1, b=0.1)
        }

        if linear_init in linear_initializers:
            for module in self.unique.modules():
                if isinstance(module, nn.Linear):
                    linear_initializers[linear_init](module.weight)
                    init.constant_(module.bias, 0)
    def forward(self, input):
        input=input.unsqueeze(1)
        shallow=self.F(input)
        middle=self.LHDR(shallow)
        middle=middle.squeeze(1)
        output=self.unique(middle)
        return output, shallow, middle