import torch
import numpy as np
from torchmetrics import Metric


class MRMetric(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable: None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: True

    def __init__(self):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += ranks.sum()

    def compute(self):
        return self.rank_sum / self.total
    

class MRRMetric(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += (1. / ranks).sum()

    def compute(self):
        return self.rank_sum / self.total
    
class HitsMetric(Metric):
    higher_is_better = True
    full_state_update = True

    def __init__(self, topk=1):
        super().__init__()
        self.add_state('rank_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.topk = topk

    def update(self, ranks):
        self.total += ranks.size(0)
        self.rank_sum += (ranks <= self.topk).sum()

    def compute(self):
        return self.rank_sum / self.total


def eval(score_fine, score_coarse, mask, answer, top_k, delta):

    filter_mask = mask.clone()
    filter_mask[torch.arange(mask.size(0)), answer] = False
    score_coarse = score_coarse.to(score_fine.device)
    mask_score = score_coarse.masked_fill(filter_mask, float('-inf'))
    filter_argsort = torch.argsort(mask_score, descending=True)

    Hits10 = HitsMetric(10)
    Hits1 = HitsMetric(1)
    Hits3 = HitsMetric(3)
    MRR = MRRMetric()
    MR = MRMetric()

    Hits10 = Hits10.to(score_fine.device)
    Hits1 = Hits1.to(score_fine.device)
    Hits3 = Hits3.to(score_fine.device)
    MRR = MRR.to(score_fine.device)
    MR = MR.to(score_fine.device)

    Hits1.reset()
    Hits10.reset()
    Hits3.reset()
    MRR.reset()
    MR.reset()


    top100_score = torch.gather(score_fine, 1, filter_argsort[:, :top_k])
    other_score = torch.gather(score_fine, 1, filter_argsort[:,top_k:])


    ranks_final = torch.zeros(answer.size(0), dtype=torch.int64, device=answer.device)

    topk_max = top100_score.max(dim=1).values 
    other_max = other_score.max(dim=1).values  
    answer_score = score_fine.gather(1, answer.view(-1, 1)) 

   
    diff = other_max - topk_max


    in_top100 = (answer.unsqueeze(1) == filter_argsort[:, :top_k]).any(dim=1)  


    condition1 = (diff <= delta) & in_top100   #At this point the correct answer was successfully narrowed down to the topk
    condition2 = (diff > delta) & (~in_top100) #At this point the correct answer was successfully narrowed down to the non-topk

   
    ranks_final = torch.where(
    condition1,
    ((top100_score >= answer_score) & (~mask.gather(1, filter_argsort[:, :top_k]))).sum(dim=1) + 1,
    torch.where(
        condition2,
        ((other_score >= answer_score) & (~mask.gather(1, filter_argsort[:, top_k:]))).sum(dim=1) + 1,
        ((score_fine >= answer_score) & (~mask)).sum(dim=1) + 1
    )
)


    Hits10.update(ranks_final)  
    Hits1.update(ranks_final)
    Hits3.update(ranks_final)
    MRR.update(ranks_final)
    MR.update(ranks_final)


    mr = MR.compute()
    mrr = MRR.compute()
    hits1 = Hits1.compute()
    hits3 = Hits3.compute()
    hits10 = Hits10.compute()
    
    return mr, mrr, hits1, hits3, hits10



