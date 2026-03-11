#!/usr/bin/env python3
"""Generate a small ONNX model for CUDA/ORT structural verification.

The generated model matches the GPU pipeline's assumptions:
- one input:  float32[batch, 16]
- one output: float32[batch]
- dynamic batch dimension

Computation:
    output[i] = dot(input[i], [1, 2, ..., 16])

This is intentionally simple and deterministic so `ort-verify` can compute the
expected outputs exactly and validate the direct `RunAsync` path.

Requires:
    pip install onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


FEATURE_DIM = 16


def build_model() -> onnx.ModelProto:
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        ["batch", FEATURE_DIM],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        ["batch"],
    )

    weights = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [FEATURE_DIM, 1],
        [float(i + 1) for i in range(FEATURE_DIM)],
    )
    axes = helper.make_tensor("axes", TensorProto.INT64, [1], [1])

    matmul = helper.make_node("MatMul", ["input", "weights"], ["matmul_out"])
    squeeze = helper.make_node("Squeeze", ["matmul_out", "axes"], ["output"])

    graph = helper.make_graph(
        [matmul, squeeze],
        "disrust_verify_model",
        [input_info],
        [output_info],
        initializer=[weights, axes],
    )

    opset = helper.make_operatorsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset], producer_name="disrust")
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="tests/models/ort_verify_model.onnx",
        help="Path to write the generated ONNX model",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    onnx.save(model, output)
    print(output)


if __name__ == "__main__":
    main()
