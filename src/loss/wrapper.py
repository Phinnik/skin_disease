import torch


class UncertaintyLossWrapper(torch.nn.Module):
    def __init__(self, losses: list[torch.nn.Module]):
        """
        https://arxiv.org/pdf/1705.07115v3.pdf

        :param losses: list of losses
        """
        super(UncertaintyLossWrapper, self).__init__()
        self.losses = losses
        self.log_vars = torch.nn.Parameter(torch.zeros(len(losses)))

    def forward(self, preds, targets):
        loss_results = []
        for loss, p, t, lv in zip(self.losses, preds, targets, self.log_vars):
            precision = torch.exp(-lv)
            l = precision * loss(p, t) + lv
            loss_results.append(l)

        return sum(loss_results)
