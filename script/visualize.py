import os
import re
import sys
import math
import pprint
import textwrap

from matplotlib import pyplot as plt
from matplotlib import offsetbox, patches, rcParams

import torch

from torchdrug import core, data, utils
from torchdrug.layers import functional
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gnnqe import dataset, gnn, model, task, util


vocab_file = os.path.join(os.path.dirname(__file__), "../data/fb15k237_entity.txt")
vocab_file = os.path.abspath(vocab_file)


def load_vocab(dataset):
    entity_mapping = {}
    with open(vocab_file, "r") as fin:
        for line in fin:
            k, v = line.strip().split("\t")
            entity_mapping[k] = v
    entity_vocab = [entity_mapping[t] for t in dataset.entity_vocab]
    relation_vocab = []
    for t in dataset.relation_vocab:
        new_t = t[t.rfind("/") + 1:].replace("_", " ")
        if t[0] == "-":
            new_t = new_t + "$^{-1}$"
        relation_vocab.append(new_t)

    return entity_vocab, relation_vocab


def computation_graph_layout(depth, left, right, width=6, height=3, vsep=1, hsep=2, pad=1):
    """Generate the layout of axes for a computation graph."""
    num_local_rows = depth.bincount().tolist()
    # stable sort
    depth_ext = depth * len(depth) + torch.arange(len(depth), device=depth.device)
    by_depth = depth_ext.argsort().tolist()
    depth = depth.tolist()
    left = left.tolist()
    right = right.tolist()

    num_row = max(right)
    num_col = max(depth) + 1
    total_width = width * num_col + hsep * (num_col - 1) + pad * 2
    total_height = height * num_row + vsep * (num_row - 1) + pad * 2
    fig = plt.figure(figsize=(total_width, total_height))

    # generate columns
    widths = [pad]
    widths += [width, hsep] * (num_col - 1)
    widths += [width, pad]
    col_grid = fig.add_gridspec(1, num_col * 2 + 1, width_ratios=widths)
    col_grid.tight_layout(fig, pad=0)

    axes = [None] * len(depth)
    k = 0
    for i, num_local_row in enumerate(num_local_rows):
        # generate local rows for a single column
        last = 0
        local_heights = []
        for j in range(k, k + num_local_row):
            if num_row > 1:
                ratio = (left[by_depth[j]] + right[by_depth[j]] - 1) / 2 / (num_row - 1)
            else:
                ratio = 0.5
            pos = ratio * (total_height - pad * 2 - height) + pad
            local_heights.append(pos - last)
            local_heights.append(height)
            last = pos + height
        local_heights.append(total_height - sum(local_heights))
        local_row_grid = col_grid[i * 2 + 1].subgridspec(num_local_row * 2 + 1, 1, height_ratios=local_heights)

        # generate axes for each node in the computation graph
        for j in range(num_local_row):
            ax = fig.add_subplot(local_row_grid[j * 2 + 1: (j + 1) * 2])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            axes[by_depth[k + j]] = ax
        k += num_local_row

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)

    return fig, axes


def plot_entities(entities, types, fig, axes, size=14):
    """Plot the entities in each axis."""
    type2color = {"easy answer (true)": "C0", "easy answer (false)": "C1",
                  "hard answer (true)": "C2", "hard answer (false)": "C3"}

    for i in range(len(entities)):
        vpack = []
        entity = entities[i]
        type = types[i]
        if isinstance(entity, str):
            entity = [entity]
            type = [type]
        for e, t in zip(entity, type):
            width = int(axes[i].bbox.width / size * 1.2)
            start, end = re.search(r"(?: \(Q\d+\))?: [^:]+$", e).span()
            e = textwrap.shorten(e[:start], width=width - len(e) + start, placeholder="...") + e[start:]
            area = offsetbox.DrawingArea(size, size)
            circle = patches.Circle((size / 2, size / 2), size / 3, color=type2color[t])
            area.add_artist(circle)
            text = offsetbox.TextArea(e, textprops={"size": size})
            hpack = offsetbox.HPacker(children=[area, text], pad=0, sep=size, width=axes[i].bbox.width - size * 2)
            vpack.append(hpack)
        vpack = offsetbox.VPacker(children=vpack, pad=size / 2, sep=size / 2)
        bbox = offsetbox.AnnotationBbox(vpack, (0.5, 0.5), xycoords=axes[i].transAxes,
                                        bboxprops={"boxstyle": "round,rounding_size=1"})
        axes[i].add_artist(bbox)

    circles = []
    for c in type2color.values():
        circle = plt.scatter([], [], c=c, s=size * 6, marker="o")
        circles.append(circle)
    legend = fig.legend(circles, type2color.keys(), bbox_to_anchor=(0.5, 1), loc="upper center",
                        fontsize=size * 1.3, handletextpad=0.2, ncol=len(circles))
    frame = legend.get_frame()
    frame.set_edgecolor((0, 0, 0))
    fig.tight_layout(pad=0)


def plot_operations(pointer, operations, fig, axes, size=14):
    """Plot the operations between axes."""
    pointer = pointer.tolist()

    xyA = (1, 0.5)
    xyB = (0, 0.5)
    for i, ptr in enumerate(pointer[:-1]):
        arrow = patches.ConnectionPatch(xyA, xyB, coordsA=axes[i].transData, coordsB=axes[ptr].transData,
                                        arrowstyle="->", mutation_scale=size * 2, linewidth=size / 6,
                                        shrinkA=size, shrinkB=size)
        ax, ay = axes[i].transData.transform(xyA)
        bx, by = axes[ptr].transData.transform(xyB)
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        angle = math.atan2(by - ay, bx - ax)
        operation = textwrap.fill(operations[i], width=18)
        num_line = operation.count("\n") + 1
        center = (cx - math.sin(angle) * size * (num_line * 2 / 3 + 0.5),
                  cy + math.cos(angle) * size * (num_line * 2 / 3 + 0.5))
        plt.annotate(operation, center, xycoords="figure pixels", va="center", ha="center",
                     rotation=angle / math.pi * 180, size=size)
        fig.add_artist(arrow)


def get_topk_prediction(query, var_probs, easy_answers, all_answers, entity_vocab, num_easy=3, num_hard=6,
                        min_prob=0.1):
    """Get the top-k easy and hard answers from each fuzzy set."""
    easy_probs = var_probs * (easy_answers > 0.5)
    hard_probs = var_probs * (easy_answers <= 0.5)
    num_easy_preds = (easy_probs >= min_prob).sum(dim=-1)
    num_hard_preds = (hard_probs >= min_prob).sum(dim=-1)

    entities = []
    types = []
    for i in range(len(query)):
        easy_pred = easy_probs[i].topk(min(num_easy, num_easy_preds[i]))[1]
        hard_pred = hard_probs[i].topk(min(num_hard, num_hard_preds[i]))[1]
        entity = []
        type = []
        for e in easy_pred:
            entity.append("%s: %.2f" % (entity_vocab[e], easy_probs[i, e]))
            if all_answers[i, e] > 0.5:
                type.append("easy answer (true)")
            else:
                type.append("easy answer (false)")
        for e in hard_pred:
            entity.append("%s: %.2f" % (entity_vocab[e], hard_probs[i, e]))
            if all_answers[i, e] > 0.5:
                type.append("hard answer (true)")
            else:
                type.append("hard answer (false)")
        entities.append(entity)
        types.append(type)

    return entities, types


def get_random_truth(query, pointer, var_probs, easy_answers, all_answers, entity_vocab):
    """
    Get a random truth from each fuzzy set.
    The random truths form a grounding of a hard answer to the query.
    """
    neg_probs = var_probs * (all_answers <= 0.5)
    graph = solver.model.graph

    entities = [None] * len(query)
    types = [None] * len(query)
    for i in reversed(range(len(query))):
        if i == len(query) - 1:
            candidate = (all_answers[i] - easy_answers[i] > 0.5).nonzero().flatten()
        else:
            candidate = (all_answers[i] > 0.5).nonzero().flatten()
        assert len(candidate) >= 1
        index = torch.randint(len(candidate), (1,)).item()
        t = candidate[index]
        rank = (var_probs[i, t] <= neg_probs[i]).sum().item() + 1
        entities[i] = "%s: %.2f (rank = %d)" % (entity_vocab[t], var_probs[i, t], rank)
        if easy_answers[i, t] > 0.5:
            types[i] = "easy answer (true)"
        else:
            types[i] = "hard answer (true)"

        op = query[i]
        union_inputs = []
        for j in range(i):
            if pointer[j] != i:
                continue
            if op.is_projection():
                r = op.get_operand()
                edge_index = graph.match([-1, t, r])[0]
                h_mask = functional.as_mask(graph.edge_list[edge_index, 0], graph.num_node)
                all_answers[j, ~h_mask] = 0
            elif op.is_intersection():
                assert all_answers[j, t] > 0.5
                all_answers[j, :] = 0
                all_answers[j, t] = 1
            elif op.is_union():
                if all_answers[j, t] > 0.5:
                    union_inputs.append(j)
            elif op.is_negation():
                assert all_answers[j, t] < 0.5
            else:
                raise ValueError("Unknown operator `%d`" % op)
        if op.is_union():
            assert len(union_inputs) >= 1
            rand = torch.randint(len(union_inputs), (1,)).item()
            j = union_inputs[rand]
            assert all_answers[j, t] > 0.5
            all_answers[j, :] = 0
            all_answers[j, t] = 1

    return entities, types


def visualize(solver, query, entity_vocab, relation_vocab, mode="prediction", save_file=None):
    """Visualize the prediction / truth of a query."""
    batch = {"query": query.unsqueeze(0)}
    if solver.device.type == "cuda":
        batch = utils.cuda(batch, device=solver.device)

    solver.model.eval()
    var_probs, easy_answers, all_answers = solver.model.visualize(batch)
    var_probs = var_probs.squeeze(0).cpu()
    easy_answers = easy_answers.squeeze(0).cpu()
    all_answers = all_answers.squeeze(0).cpu()
    # remove padded stop operations
    query = query[~query.is_stop()]
    pointer, depth, left, right = query.computation_graph()

    if mode == "prediction":
        entities, types = get_topk_prediction(query, var_probs, easy_answers, all_answers, entity_vocab)
    elif mode == "truth":
        entities, types = get_random_truth(query, pointer, var_probs, easy_answers, all_answers, entity_vocab)
    else:
        raise ValueError("Unknown mode `%s`" % mode)

    operations = []
    for op in query[pointer]:
        if op.is_projection():
            r = op.get_operand()
            operations.append(relation_vocab[r])
        elif op.is_intersection():
            operations.append("intersection")
        elif op.is_union():
            operations.append("union")
        elif op.is_negation():
            operations.append("negation")
        else:
            raise ValueError("Unknown operator `%d`" % op)

    fig, axes = computation_graph_layout(depth, left, right)
    plot_entities(entities, types, fig, axes)
    plot_operations(pointer, operations, fig, axes)

    if save_file:
        fig.savefig(save_file)
    else:
        fig.show()
    plt.close(fig)


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] != "FB15k237LogicalQuery":
        raise ValueError("Visualization is only implemented for FB15k237")

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    entity_vocab, relation_vocab = load_vocab(dataset)

    index = torch.randperm(len(solver.test_set))[:200]
    for i in index.tolist():
        for mode in ["prediction", "truth"]:
            save_file = "%s_%d_%s.png" % (solver.model.id2type[solver.test_set[i]["type"]], i, mode)
            visualize(solver, solver.test_set[i]["query"], entity_vocab, relation_vocab,
                      mode=mode, save_file=save_file)
