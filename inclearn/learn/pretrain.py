import os.path as osp

import torch
import torch.nn.functional as F
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


def _compute_loss(cfg, logits, targets, device):

    if cfg["train_head"] == "sigmoid":
        n_classes = cfg["start_class"]
        onehot_targets = utils.to_onehot(targets, n_classes).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
    elif cfg["train_head"] == "softmax":
        loss = F.cross_entropy(logits, targets)
    else:
        raise ValueError()

    return loss


def train(cfg, model, optimizer, device, train_loader):
    _loss = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.train()
    for i, (inputs, targets) in enumerate(train_loader, start=1):
        # assert torch.isnan(inputs).sum().item() == 0
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model._parallel_network(inputs)['logit']
        if accu is not None:
            accu.add(logits.detach(), targets)

        loss = _compute_loss(cfg, logits, targets, device)
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()

        loss.backward()
        optimizer.step()
        _loss += loss

    return (
        round(_loss.item() / i, 3),
        round(accu.value()[0], 3),
    )


def test(cfg, model, device, test_loader):
    _loss = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader, start=1):
            # assert torch.isnan(inputs).sum().item() == 0
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model._parallel_network(inputs)['logit']
            if accu is not None:
                accu.add(logits.detach(), targets)
            loss = _compute_loss(cfg, logits, targets, device)
            if torch.isnan(loss):
                import pdb
                pdb.set_trace()

            _loss = _loss + loss
    return round(_loss.item() / i, 3), round(accu.value()[0], 3)


def pretrain(cfg, ex, model, device, train_loader, test_loader, model_path):
    ex.logger.info(f"nb Train {len(train_loader.dataset)} Eval {len(test_loader.dataset)}")
    optimizer = torch.optim.SGD(model._network.parameters(),
                                lr=cfg["pretrain"]["lr"],
                                momentum=0.9,
                                weight_decay=cfg["pretrain"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     cfg["pretrain"]["scheduling"],
                                                     gamma=cfg["pretrain"]["lr_decay"])
    test_loss, test_acc = float("nan"), float("nan")
    for e in range(cfg["pretrain"]["epochs"]):
        train_loss, train_acc = train(cfg, model, optimizer, device, train_loader)
        if e % 5 == 0:
            test_loss, test_acc = test(cfg, model, device, test_loader)
            ex.logger.info(
                "Pretrain Class {}, Epoch {}/{} => Clf Train loss: {}, Accu {} | Eval loss: {}, Accu {}".format(
                    cfg["start_class"], e + 1, cfg["pretrain"]["epochs"], train_loss, train_acc, test_loss, test_acc))
        else:
            ex.logger.info("Pretrain Class {}, Epoch {}/{} => Clf Train loss: {}, Accu {} ".format(
                cfg["start_class"], e + 1, cfg["pretrain"]["epochs"], train_loss, train_acc))
        scheduler.step()
    if hasattr(model._network, "module"):
        torch.save(model._network.module.state_dict(), model_path)
    else:
        torch.save(model._network.state_dict(), model_path)
