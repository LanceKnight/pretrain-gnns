from typing import Callable, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear
# from ..inits import reset
from typing import Optional

import copy
import math

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_geometric.nn import inits


# class Linear(torch.nn.Module):
#     r"""Applies a linear tranformation to the incoming data
#     .. math::
#         \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}
#     similar to :class:`torch.nn.Linear`.
#     It supports lazy initialization and customizable weight and bias
#     initialization.
#     Args:
#         in_channels (int): Size of each input sample. Will be initialized
#             lazily in case it is given as :obj:`-1`.
#         out_channels (int): Size of each output sample.
#         bias (bool, optional): If set to :obj:`False`, the layer will not learn
#             an additive bias. (default: :obj:`True`)
#         weight_initializer (str, optional): The initializer for the weight
#             matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
#             or :obj:`None`).
#             If set to :obj:`None`, will match default weight initialization of
#             :class:`torch.nn.Linear`. (default: :obj:`None`)
#         bias_initializer (str, optional): The initializer for the bias vector
#             (:obj:`"zeros"` or :obj:`None`).
#             If set to :obj:`None`, will match default bias initialization of
#             :class:`torch.nn.Linear`. (default: :obj:`None`)
#     """

#     def __init__(self, in_channels: int, out_channels: int, bias: bool = True,
#                  weight_initializer: Optional[str] = None,
#                  bias_initializer: Optional[str] = None):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.weight_initializer = weight_initializer
#         self.bias_initializer = bias_initializer

#         if in_channels > 0:
#             self.weight = Parameter(torch.Tensor(out_channels, in_channels))
#         else:
#             self.weight = torch.nn.parameter.UninitializedParameter()
#             self._hook = self.register_forward_pre_hook(
#                 self.initialize_parameters)

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def __deepcopy__(self, memo):
#         out = Linear(self.in_channels, self.out_channels, self.bias
#                      is not None, self.weight_initializer,
#                      self.bias_initializer)
#         if self.in_channels > 0:
#             out.weight = copy.deepcopy(self.weight, memo)
#         if self.bias is not None:
#             out.bias = copy.deepcopy(self.bias, memo)
#         return out

#     def reset_parameters(self):
#         if self.in_channels > 0:
#             if self.weight_initializer == 'glorot':
#                 inits.glorot(self.weight)
#             elif self.weight_initializer == 'uniform':
#                 bound = 1.0 / math.sqrt(self.weight.size(-1))
#                 torch.nn.init.uniform_(self.weight.data, -bound, bound)
#             elif self.weight_initializer == 'kaiming_uniform':
#                 inits.kaiming_uniform(self.weight, fan=self.in_channels,
#                                       a=math.sqrt(5))
#             elif self.weight_initializer is None:
#                 inits.kaiming_uniform(self.weight, fan=self.in_channels,
#                                       a=math.sqrt(5))
#             else:
#                 raise RuntimeError(
#                     f"Linear layer weight initializer "
#                     f"'{self.weight_initializer}' is not supported")

#         if self.in_channels > 0 and self.bias is not None:
#             if self.bias_initializer == 'zeros':
#                 inits.zeros(self.bias)
#             elif self.bias_initializer is None:
#                 inits.uniform(self.in_channels, self.bias)
#             else:
#                 raise RuntimeError(
#                     f"Linear layer bias initializer "
#                     f"'{self.bias_initializer}' is not supported")

#     def forward(self, x: Tensor) -> Tensor:
#         """"""
#         return F.linear(x, self.weight, self.bias)

#     @torch.no_grad()
#     def initialize_parameters(self, module, input):
#         if isinstance(self.weight, torch.nn.parameter.UninitializedParameter):
#             self.in_channels = input[0].size(-1)
#             self.weight.materialize((self.out_channels, self.in_channels))
#             self.reset_parameters()
#         module._hook.remove()
#         delattr(module, '_hook')

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.in_channels}, '
#                 f'{self.out_channels}, bias={self.bias is not None})')


class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GINEConv(MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None
        # self.reset_parameters()

    # def reset_parameters(self):
    #     reset(self.nn)
    #     self.eps.data.fill_(self.initial_eps)
    #     if self.lin is not None:
    #         self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
