"""Smoke-test and inspect the separate STaRK KG loader."""

from __future__ import annotations

import argparse
import json

from stark_loader import STARK_DATASETS, STARK_SPLITS, load_stark, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=STARK_DATASETS, default="prime")
    parser.add_argument("--split", choices=STARK_SPLITS, default="train")
    parser.add_argument("--root", default="data/stark")
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()

    data = load_stark(args.dataset, split=args.split, root=args.root)
    print(json.dumps(summarize(data), indent=2))
    print("first_query:", data.queries.texts[0])
    print("first_answers:", data.queries.answer_ids[0])
    print("first_candidate_ids:", data.graph.candidate_ids[:10].tolist())
    print("first_edge:", data.graph.edge_index[:, 0].tolist())
    print("first_edge_type:", data.graph.edge_type(0))
    if args.show_text:
        node_id = int(data.graph.candidate_ids[0])
        print("first_candidate_text:", data.graph.node_text([node_id])[0])


if __name__ == "__main__":
    main()

