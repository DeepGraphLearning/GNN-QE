import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torch_scatter import scatter_add, scatter_mean

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("task.LogicalQuery")
class LogicalQuery(tasks.Task, core.Configurable):
    """
    Logical query task.

    Parameters:
        model (nn.Module): logical query model
        criterion (str, list or dict, optional): training criterion(s). Only ``bce`` is available for now.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mrr``, ``hits@K``, ``mape`` and ``spearmanr``.
        adversarial_temperature (float, optional): temperature for self-adversarial negative sampling.
            Set ``0`` to disable self-adversarial negative sampling.
        sample_weight (bool, optional): whether to weight each query by its number of answers
    """

    _option_members = ["criterion", "metric", "query_type_weight"]

    def __init__(self, model, criterion="bce", metric=("mrr",), adversarial_temperature=0, sample_weight=False):
        super(LogicalQuery, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.adversarial_temperature = adversarial_temperature
        self.sample_weight = sample_weight

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.id2type = dataset.id2type
        self.type2id = dataset.type2id

        self.register_buffer("fact_graph", dataset.fact_graph)
        self.register_buffer("graph", dataset.graph)

        return train_set, valid_set, test_set

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                is_positive = target > 0.5
                is_negative = target <= 0.5
                num_positive = is_positive.sum(dim=-1)
                num_negative = is_negative.sum(dim=-1)
                neg_weight = torch.zeros_like(pred)
                neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        logit = pred[is_negative] / self.adversarial_temperature
                        neg_weight[is_negative] = functional.variadic_softmax(logit, num_negative)
                else:
                    neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            if self.sample_weight:
                sample_weight = target.sum(dim=-1).float()
                loss = (loss * sample_weight).sum() / sample_weight.sum()
            else:
                loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict_and_target(self, batch, all_loss=None, metric=None):
        query = batch["query"]
        type = batch["type"]
        easy_answer = batch["easy_answer"]
        hard_answer = batch["hard_answer"]

        pred = self.model(self.fact_graph, query, all_loss, metric)
        if all_loss is None:
            target = (type, easy_answer, hard_answer)
            ranking = self.batch_evaluate(pred, target)
            # answer set cardinality prediction
            prob = F.sigmoid(pred)
            num_pred = (prob * (prob > 0.5)).sum(dim=-1)
            num_easy = easy_answer.sum(dim=-1)
            num_hard = hard_answer.sum(dim=-1)
            return (ranking, num_pred), (type, num_easy, num_hard)
        else:
            target = easy_answer.float()

        return pred, target

    def batch_evaluate(self, pred, target):
        type, easy_answer, hard_answer = target

        num_easy = easy_answer.sum(dim=-1)
        num_hard = hard_answer.sum(dim=-1)
        num_answer = num_easy + num_hard
        answer2query = functional._size_to_index(num_answer)
        order = pred.argsort(dim=-1, descending=True)
        range = torch.arange(self.num_entity, device=self.device)
        ranking = scatter_add(range.expand_as(order), order, dim=-1)
        easy_ranking = ranking[easy_answer]
        hard_ranking = ranking[hard_answer]
        # unfiltered rankings of all answers
        answer_ranking = functional._extend(easy_ranking, num_easy, hard_ranking, num_hard)[0]
        order_among_answer = functional.variadic_sort(answer_ranking, num_answer)[1]
        order_among_answer = order_among_answer + (num_answer.cumsum(0) - num_answer)[answer2query]
        ranking_among_answer = scatter_add(functional.variadic_arange(num_answer), order_among_answer)

        # filtered rankings of all answers
        ranking = answer_ranking - ranking_among_answer + 1
        ends = num_answer.cumsum(0)
        starts = ends - num_hard
        hard_mask = functional.multi_slice_mask(starts, ends, ends[-1])
        # filtered rankings of hard answers
        ranking = ranking[hard_mask]

        return ranking

    def evaluate(self, pred, target):
        ranking, num_pred = pred
        type, num_easy, num_hard = target

        metric = {}
        for _metric in self.metric:
            if _metric == "mrr":
                answer_score = 1 / ranking.float()
                query_score = functional.variadic_mean(answer_score, num_hard)
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                answer_score = (ranking <= threshold).float()
                query_score = functional.variadic_mean(answer_score, num_hard)
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric == "mape":
                query_score = (num_pred - num_easy - num_hard).abs() / (num_easy + num_hard).float()
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric == "spearmanr":
                type_score = []
                for i in range(len(self.id2type)):
                    mask = type == i
                    score = metrics.spearmanr(num_pred[mask], num_easy[mask] + num_hard[mask])
                    type_score.append(score)
                type_score = torch.stack(type_score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            score = type_score.mean()
            is_neg = torch.tensor(["n" in t for t in self.id2type], device=self.device)
            is_epfo = ~is_neg
            name = tasks._get_metric_name(_metric)
            for i, query_type in enumerate(self.id2type):
                metric["[%s] %s" % (query_type, name)] = type_score[i]
            if is_epfo.any():
                epfo_score = functional.masked_mean(type_score, is_epfo)
                metric["[EPFO] %s" % name] = epfo_score
            if is_neg.any():
                neg_score = functional.masked_mean(type_score, is_neg)
                metric["[negation] %s" % name] = neg_score
            metric[name] = score

        return metric

    def visualize(self, batch):
        query = batch["query"]
        return self.model.visualize(self.fact_graph, self.graph, query)
