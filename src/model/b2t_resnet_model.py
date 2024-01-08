from typing import Optional
from torch import Tensor
from src.args.b2t_resnet_args import B2TResnetArgsModel
from src.model.b2tmodel import B2TModel, ModelOutput
from torchvision.models import resnet50
from torch import nn
import torch


class B2TResnetModel(B2TModel):
    def __init__(
        self, config: B2TResnetArgsModel, num_classes: int, blank_token_id: int | None
    ) -> None:
        super().__init__()
        self.blank_token_id = blank_token_id
        self.config = config
        self.data_transform = nn.Conv2d(
            in_channels=1,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.resnet = resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.num_classes = num_classes
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor, targets: Tensor | None = None) -> ModelOutput:
        batch_size = x.size(0)
        recording_length = x.size(1)

        logits = torch.zeros(size=[batch_size, recording_length, self.num_classes])

        for i in range(recording_length):
            window = x[:, i, :].unsqueeze(0).view(batch_size, 1, 16, 16)
            window = nn.functional.interpolate(window, size=(64, 64), mode="bilinear")
            transformed = self.data_transform(window)
            resnet_output = self.resnet(transformed)
            logits[:, i] = resnet_output

        loss = None
        if targets is not None:
            labels_mask = targets >= 0
            target_lengths = labels_mask.sum(-1)
            input_lengths = torch.tensor([recording_length] * batch_size)
            flattened_targets = targets.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_token_id if self.blank_token_id is not None else 0,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=True,
                )

        return ModelOutput(logits=logits, loss=loss)
