import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import data, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .data import Query


class LogicalQueryDataset(data.KnowledgeGraphDataset):
    """Logical query dataset."""

    struct2type = {
        ("e", ("r",)): "1p",
        ("e", ("r", "r")): "2p",
        ("e", ("r", "r", "r")): "3p",
        (("e", ("r",)), ("e", ("r",))): "2i",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
        ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
        (("e", ("r", "r")), ("e", ("r",))): "pi",
        (("e", ("r",)), ("e", ("r", "n"))): "2in",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
        ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
        (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
        (("e", ("r", "r", "n")), ("e", ("r",))): "pni",
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2u-DNF",
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up-DNF",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",)): "2u-DM",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r")): "up-DM",
    }

    def load_pickle(self, path, query_types=None, union_type="DNF", verbose=0):
        """
        Load the dataset from pickle dumps (BetaE format).

        Parameters:
            path (str): path to pickle dumps
            query_types (list of str, optional): query types to load.
                By default, load all query types.
            union_type (str, optional): which union type to use, ``DNF`` or ``DM``
            verbose (int, optional): output verbose level
        """
        query_types = query_types or self.struct2type.values()
        new_query_types = []
        for query_type in query_types:
            if "u" in query_type:
                if "-" not in query_type:
                    query_type = "%s-%s" % (query_type, union_type)
                elif query_type[query_type.find("-") + 1:] != union_type:
                    continue
            new_query_types.append(query_type)
        self.id2type = sorted(new_query_types)
        self.type2id = {t: i for i, t in enumerate(self.id2type)}

        with open(os.path.join(path, "id2ent.pkl"), "rb") as fin:
            entity_vocab = pickle.load(fin)
        with open(os.path.join(path, "id2rel.pkl"), "rb") as fin:
            relation_vocab = pickle.load(fin)
        triplets = []
        num_samples = []
        for split in ["train", "valid", "test"]:
            triplet_file = os.path.join(path, "%s.txt" % split)
            with open(triplet_file) as fin:
                if verbose:
                    fin = tqdm(fin, "Loading %s" % triplet_file, utils.get_line_count(triplet_file))
                num_sample = 0
                for line in fin:
                    h, r, t = [int(x) for x in line.split()]
                    triplets.append((h, t, r))
                    num_sample += 1
                num_samples.append(num_sample)
        self.load_triplet(triplets, entity_vocab=entity_vocab, relation_vocab=relation_vocab)
        fact_mask = torch.arange(num_samples[0])
        # self.graph is the full graph without missing edges
        # self.fact_graph is the training graph
        self.fact_graph = self.graph.edge_mask(fact_mask)

        queries = []
        types = []
        easy_answers = []
        hard_answers = []
        num_samples = []
        max_query_length = 0

        for split in ["train", "valid", "test"]:
            if verbose:
                pbar = tqdm(desc="Loading %s-*.pkl" % split, total=3)
            with open(os.path.join(path, "%s-queries.pkl" % split), "rb") as fin:
                struct2queries = pickle.load(fin)
            if verbose:
                pbar.update(1)
            type2queries = {self.struct2type[k]: v for k, v in struct2queries.items()}
            type2queries = {k: v for k, v in type2queries.items() if k in self.type2id}
            if split == "train":
                with open(os.path.join(path, "%s-answers.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                query2hard_answers = defaultdict(set)
                if verbose:
                    pbar.update(2)
            else:
                with open(os.path.join(path, "%s-easy-answers.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                if verbose:
                    pbar.update(1)
                with open(os.path.join(path, "%s-hard-answers.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
                if verbose:
                    pbar.update(1)

            num_sample = sum([len(q) for t, q in type2queries.items()])
            if verbose:
                pbar = tqdm(desc="Processing %s queries" % split, total=num_sample)
            for type in type2queries:
                struct_queries = sorted(type2queries[type])
                for query in struct_queries:
                    easy_answers.append(query2easy_answers[query])
                    hard_answers.append(query2hard_answers[query])
                    query = Query.from_nested(query)
                    queries.append(query)
                    max_query_length = max(max_query_length, len(query))
                    types.append(self.type2id[type])
                    if verbose:
                        pbar.update(1)
            num_samples.append(num_sample)

        self.queries = queries
        self.types = types
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        self.num_samples = num_samples
        self.max_query_length = max_query_length

    def __getitem__(self, index):
        query = self.queries[index]
        easy_answer = torch.tensor(list(self.easy_answers[index]), dtype=torch.long)
        hard_answer = torch.tensor(list(self.hard_answers[index]), dtype=torch.long)
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "easy_answer": functional.as_mask(easy_answer, self.num_entity),
            "hard_answer": functional.as_mask(hard_answer, self.num_entity),
        }

    def __len__(self):
        return len(self.queries)

    def __repr__(self):
        lines = [
            "#entity: %d" % self.num_entity,
            "#relation: %d" % self.num_relation,
            "#triplet: %d" % self.num_triplet,
            "#query: %d" % len(self.queries),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("dataset.FB15kLogicalQuery")
class FB15kLogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "FB15k-betae")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)


@R.register("dataset.FB15k237LogicalQuery")
class FB15k237LogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "FB15k-237-betae")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)


@R.register("dataset.NELL995LogicalQuery")
class NELL995LogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "NELL-betae")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)
