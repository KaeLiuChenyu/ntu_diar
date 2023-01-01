import torch

from ntu_diar.scr.model.component.linear import Linear
from ntu_diar.scr.model.component.normalization import BatchNorm1d
from ntu_diar.scr.model.component.activations import Softmax
from ntu_diar.scr.model.component.containers import Sequential

#---------------------------------------------------------#
class Classifier(Sequential):
#---------------------------------------------------------#

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=512,
        out_neurons=1211,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(Sequential, layer_name="DNN")

        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
            Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            Softmax(apply_log=True), layer_name="softmax"
        )