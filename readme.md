- 跑程序的sop 1.确认实验目的-config.yml的expname 2.选择searchSpaceFile 
- 3.保存路径和server和model。 2.确认模块配置true/false 3.确认训练细节 
# utils
# 1.data_utils.py
- visualize函数，实现t-SNE降维+可视化
# models
- GHDR_FL(conv+gru+linear)
- GHDR_FL_new
- GHDR_FL_testeds
- Cloud_GHDR
- Cloud_GHDR_new
- deepCNN
- conv_DANN 1.比GHDR_FL多了个discriminator 2.forward里没有shallow，多了discriminator输出
- conv_DANN2 区别在于discriminator的输入没被压缩
- FedCADA
- Cloud_FedCADA
- Cloud_FedCADA_newda
references:
寻找discriminator的输入怎么处理
- CONTRASTIVE ADVERSARIAL DOMAIN ADAPTATION FOR MACHINE REMAINING USEFUL LIFE PREDICTION：FE输出向量
# system-server(画什么线，为了观察什么？)
通用流程
1.set_clients 返回client列表（client初始化完了dataloader和优化器和loss）
2.train里运行client.train
- serverDANN
1.只有一个client
2.
# system-client(画什么线，为了观察什么？)
- clientDANN
1.初始化source/train/alltrain+loader
2.画mmd来监控S和T特征分布是否靠近
3.self.per_layer用于实现个性化的预测头

- clientCADA
1.在clientDANN的基础上