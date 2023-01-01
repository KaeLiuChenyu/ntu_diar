import torch
import torch.nn as nn

#---------------------------------------------------------#
class StatisticsPooling(nn.Module):
#---------------------------------------------------------#

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False \n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                if self.return_mean:
                    mean.append(
                        torch.mean(x[snt_id, 0:actual_size, ...], dim=0)
                    )
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            gnoise = gnoise
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.
        Arguments
        ---------
        shape_of_tensor : tensor
            It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)

        return gnoise
