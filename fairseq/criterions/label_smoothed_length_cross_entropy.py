# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('label_smoothed_length_cross_entropy')
class LabelSmoothedLengthCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        # self.sep_idx = task.target_dictionary.sep()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        pp = True
        if pp:
            import torch
            group = torch.distributed.new_group(ranks=[0, 1], backend='nccl')
            updated_sample = self.sync_samples(sample, group)
            if torch.distributed.get_rank() == 1:
                sample = updated_sample
            net_output = model(**sample['net_input'], group=group)
            if torch.distributed.get_rank() == 1:
                loss, nll_loss, length_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            else:
                loss, nll_loss, length_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1)
        else:
            net_output = model(**sample['net_input'])
            loss, nll_loss, length_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'length_loss': utils.item(length_loss.data) if reduce else length_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def sync_samples(self, sample, group):
        import torch
        device = torch.device('cuda:' + str(torch.distributed.get_rank()))
        for key in sample.keys():
            if isinstance(sample[key], dict):
                for inside_key in sample[key].keys():
                    update_data = self.send_recv(sample[key][inside_key], group, device, inside_key)
                    if torch.distributed.get_rank() == 1:
                        sample[key][inside_key] = update_data
            else:
                update_data = self.send_recv(sample[key], group, device, key)
                if torch.distributed.get_rank() == 1:
                    sample[key] = update_data
        if torch.distributed.get_rank() == 1:
            return sample

    def send_recv(self, data, group, device, name):
        import torch
        print('syncing: ', name)
        is_tensor = isinstance(data, torch.Tensor)
        len_size = len(data.shape) if is_tensor else 1
        shape = torch.zeros(len_size, device=device)
        if torch.distributed.get_rank() == 0:
            if is_tensor:
                for i in range(len_size):
                    shape[i] = data.shape[i]
            else:
                shape[0] = 1
            torch.distributed.send(shape, dst=1, group=group)
            if not is_tensor:
                data = torch.tensor([data], device=device, dtype=torch.int64)
            torch.distributed.send(data, dst=1, group=group)
            return None
        else:
            torch.distributed.recv(shape, src=0, group=group)
            shape_np = shape.cpu().detach().numpy().astype('int32')
            size = []
            for i in shape_np:
                size.append(i)
            size = tuple(size)
            data = torch.empty(
                size,
                device=device,
                requires_grad=False,
                dtype=data.dtype if is_tensor else torch.int64)
            torch.distributed.recv(data, src=0, group=group)
            if not is_tensor:
                data = data.cpu().detach().numpy().astype('int64')[0]
            return data

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        sample['padding_idx'] = self.padding_idx
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        # compute length prediction loss
        length_lprobs = net_output[1]['predicted_lengths']
        length_target = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).sum(-1).unsqueeze(-1)
        length_loss = -length_lprobs.gather(dim=-1, index=length_target)

        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
            length_loss = length_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss + 0.1 * length_loss
        return loss, nll_loss, length_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'length_loss': sum(log.get('length_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
