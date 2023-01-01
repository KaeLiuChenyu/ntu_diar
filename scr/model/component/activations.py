import torch

#---------------------------------------------------------#
class Softmax(torch.nn.Module):
#---------------------------------------------------------#

    def __init__(self, apply_log=False, dim=-1):
        super().__init__()

        if apply_log:
            self.act = torch.nn.LogSoftmax(dim=dim)
        else:
            self.act = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        """Returns the softmax of the input tensor.
        Arguments
        ---------
        x : torch.Tensor
            Input tensor.
        """
        # Reshaping the tensors
        dims = x.shape

        if len(dims) == 3:
            x = x.reshape(dims[0] * dims[1], dims[2])

        if len(dims) == 4:
            x = x.reshape(dims[0] * dims[1], dims[2], dims[3])

        x_act = self.act(x)

        # Retrieving the original shape format
        if len(dims) == 3:
            x_act = x_act.reshape(dims[0], dims[1], dims[2])

        if len(dims) == 4:
            x_act = x_act.reshape(dims[0], dims[1], dims[2], dims[3])

        return x_act