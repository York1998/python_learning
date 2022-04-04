# 作者：York
# 时间：2022/4/1 17:18
import torch
from torch_geometric.nn import MessagePassing

class NameConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NameConv, self).__init__(**kwargs)
        #...

    def forward(self, x, edge_index):
    	#...
        return self.propagate(edge_index, **kwargs)

    def message(self, **kwargs):
    	#...
"""
    MessagePassing初始化：
        def __init__(self, aggr: Optional[str] = "add",
             flow: str = "source_to_target", node_dim: int = -2,
             decomposed_layers: int = 1):
    aggr：邻域聚合方式，默认为add，还可以是mean，max。
    flow：消息传递方向，默认是从source_to_target，也可以设置target_to_source，
        source_to_target,也就是从节点j传递到节点i。
    node_dim：定义沿着哪个维度进行消息传递，默认-2，因为-1是特征维度。
"""
"""
    MessagePassing.propagate(edge_index, size = None,**kwargs)
    这里实现消息传递，也就是图卷积中三个步骤的地方。propagate会依次调用message，aggregate，update方法。
    如果edge_index是SpareTensor（可以理解为系数矩阵的形式存储边信息），会优先message_and_aggregate来替代
    message和aggregate。
        edge_index：给消息如何传递提供了信息，有两种形式，Tensor和SparseTensor
            Tensor形式下的edge_index的shape是 (2,N).SparseTensor可以理解为是稀疏矩阵形式
        size：当size为None的时候，默认邻接矩阵是方形[N, N]。如果是异构图，比如bipartite图时，图中的
            两类点的特征和index是相互独立的。通过传入size = (N, M)，x = (x_N, x_M)时，propagate可以处理
            这种情况。
        kwargs：图卷积计算过程中的额外所需信息，可以通过此传入
"""
"""
    def message(...)
    这个方法对应公式中的\phi，在flow = "source_to_target"的设置下，计算了邻居节点j到中心节点i的消息。传给
    propagate()素有参数都可以传递给message()，而且传递给propagate()的tensors可以通过加上_i or _j
    的后缀来mapping到对应的节点。
   
   剩下的记录在深度学习框架笔记里面了。
    
"""
"""
    MessagePassing.aggregate(inputs, index, …)
    这个方法实现了邻域的聚合，pytorch_geometric通过scatter共实现了三种方式mean,add,max。
    一般来说，比较通用的图算法，比如说GCN、GraphSAGE、GAT都不需要自己再额外定义aggregate方法。
"""
"""
    MessagePassing.update(aggr_out, …)
    这个方法对应于公式中的\phi，之前传入propagate的参数也都传入update。对应每个中心节点i，根据
    aggregate的邻域结果，以及在传入propagate的参数中选择所需信息，更新节点i的embedding。
"""
"""
    MessagePassing.message_and_aggregate(adj_t, …)
    能矩阵计算就矩阵计算！这是提高计算效率，节省计算资源很重要的一点，在图卷积中也同意适用。
    前面提到pytorch geometric中的边信息有Tensor和SparseTensor两种形式。
    当边是以SparseTensor，也就是我们通常意义上理解的稀疏矩阵的形式存储的时候，会写成adj_t。
    （为什么后面加个t，写成转置的形式？）请参考笔记。

"""

