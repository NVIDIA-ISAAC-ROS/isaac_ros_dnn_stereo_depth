#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Duplicate shared constant initializers in an ONNX model, in place.

This mirrors NVIDIA Model Optimizer's ``modelopt.onnx.utils.duplicate_shared_constants``
(the AutoCast graph-sanitizer step), reimplemented with the ``onnx`` Python API
so the only dependency is ``onnx`` rather than ``onnx_graphsurgeon``.

Why this is needed: TensorRT 10.16's weakly-typed FP16 auto-cast pass generates a
cast tensor named ``<weight>_output_casted`` for each consumer of a weight. When a
single initializer feeds more than one node and those consumers land in the same
Myelin region, the generated cast names collide and the engine build fails with
``duplicate tensor name "...weight_output_casted"`` followed by
``Could not find any implementation for node {ForeignNode[...]}``. Giving every
consumer its own uniquely-named copy of the initializer removes the collision.

The transform is semantics-preserving: ONNX initializers are immutable constants,
so duplicating one into byte-identical copies cannot change the computed result.
"""

import argparse
from collections import defaultdict

import onnx


def duplicate_shared_constants(model: onnx.ModelProto) -> int:
    """Give every consumer of a shared initializer its own uniquely-named copy.

    Returns the number of duplicated initializer references.
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    # Map each initializer name to the (node, input_index) pairs that consume it.
    consumers = defaultdict(list)
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name in initializers:
                consumers[name].append((node, idx))

    name_counts = defaultdict(int)
    duplicated = 0
    for name, uses in consumers.items():
        if len(uses) <= 1:
            continue  # not shared, nothing to do

        # Mirror AutoCast: every consumer (including the first) is rewired to a
        # fresh, uniquely-named copy; the original then becomes unreferenced.
        for node, idx in uses:
            name_counts[name] += 1
            new_name = f"{name}_{name_counts[name]}"
            dup = onnx.TensorProto()
            dup.CopyFrom(initializers[name])
            dup.name = new_name
            graph.initializer.append(dup)
            node.input[idx] = new_name
            duplicated += 1

    if duplicated:
        # Drop initializers no longer referenced by any node input (the originals
        # we just rewired), matching graphsurgeon's export which prunes orphans.
        referenced = {inp for node in graph.node for inp in node.input}
        kept = [init for init in graph.initializer if init.name in referenced]
        if len(kept) != len(graph.initializer):
            del graph.initializer[:]
            graph.initializer.extend(kept)

    return duplicated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", help="Path to the ONNX model to rewrite in place.")
    args = parser.parse_args()

    model = onnx.load(args.onnx_path)
    duplicated = duplicate_shared_constants(model)
    onnx.save(model, args.onnx_path)
    print(f"Duplicated {duplicated} shared initializer reference(s) in {args.onnx_path}.")


if __name__ == "__main__":
    main()
