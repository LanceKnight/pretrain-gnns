# from torch_geometric.utils import add_self_loops
import torch
from torch import Tensor


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def add_self_loops(
        edge_index, edge_attr=None,
        fill_value=None,
        num_nodes=None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted or has multi-dimensional edge features
    (:obj:`edge_attr != None`), edge features of self-loops will be added
    according to :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_attr is not None:
        if fill_value is None:
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], 1.)

        elif isinstance(fill_value, float) or isinstance(fill_value, int):
            loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:],
                                           fill_value)
        elif isinstance(fill_value, Tensor):
            loop_attr = fill_value.to(edge_attr.device, edge_attr.dtype)
            if edge_attr.dim() != loop_attr.dim():
                loop_attr = loop_attr.unsqueeze(0)
            sizes = [N] + [1] * (loop_attr.dim() - 1)
            loop_attr = loop_attr.repeat(*sizes)

        elif isinstance(fill_value, str):
            loop_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=N,
                                reduce=fill_value)
        else:
            raise AttributeError("No valid 'fill_value' provided")

        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_attr


edge_index = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]])
edge_attr = torch.tensor([[1, 2], [1, 1], [1, 3]])
edge_index, edge_attr = add_self_loops(edge_index, edge_attr, torch.tensor([1, 1]))
