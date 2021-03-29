import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from inclearn.tools import factory
from inclearn.convnet.imbalance import BiC, WA
from inclearn.convnet.classifier import CosineClassifier


class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu).to(self.device)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier
