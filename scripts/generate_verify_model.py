#!/usr/bin/env python3
"""Generate a small ONNX model for CUDA/ORT structural verification.

The generated model matches the GPU pipeline's assumptions:
- one input:  float32[batch, 16]
- one output: float32[batch]
- dynamic batch dimension

Computation modes:
    weighted-dot: output[i] = dot(input[i], [1, 2, ..., 16])
    plain-sum:    output[i] = sum(input[i])

This script intentionally has no third-party dependencies. It writes the ONNX
protobuf directly so it can run on minimal hosts.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


FEATURE_DIM = 16
TENSOR_FLOAT = 1
TENSOR_INT64 = 7


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field_key(field_number: int, wire_type: int) -> bytes:
    return _varint((field_number << 3) | wire_type)


def _bytes_field(field_number: int, value: bytes) -> bytes:
    return _field_key(field_number, 2) + _varint(len(value)) + value


def _string_field(field_number: int, value: str) -> bytes:
    return _bytes_field(field_number, value.encode("utf-8"))


def _varint_field(field_number: int, value: int) -> bytes:
    return _field_key(field_number, 0) + _varint(value)


def _packed_varints_field(field_number: int, values: list[int]) -> bytes:
    payload = b"".join(_varint(v) for v in values)
    return _bytes_field(field_number, payload)


def _packed_floats_field(field_number: int, values: list[float]) -> bytes:
    payload = b"".join(struct.pack("<f", v) for v in values)
    return _bytes_field(field_number, payload)


def _tensor_shape(dims: list[int | str]) -> bytes:
    out = bytearray()
    for dim in dims:
        dim_msg = bytearray()
        if isinstance(dim, int):
            dim_msg += _varint_field(1, dim)
        else:
            dim_msg += _string_field(2, dim)
        out += _bytes_field(1, bytes(dim_msg))
    return bytes(out)


def _tensor_type(elem_type: int, dims: list[int | str]) -> bytes:
    out = bytearray()
    out += _varint_field(1, elem_type)
    out += _bytes_field(2, _tensor_shape(dims))
    return bytes(out)


def _type_proto_tensor(elem_type: int, dims: list[int | str]) -> bytes:
    return _bytes_field(1, _tensor_type(elem_type, dims))


def _value_info(name: str, elem_type: int, dims: list[int | str]) -> bytes:
    out = bytearray()
    out += _string_field(1, name)
    out += _bytes_field(2, _type_proto_tensor(elem_type, dims))
    return bytes(out)


def _tensor(name: str, data_type: int, dims: list[int], *, float_data=None, int64_data=None) -> bytes:
    out = bytearray()
    out += _packed_varints_field(1, dims)
    out += _varint_field(2, data_type)
    if float_data is not None:
        out += _packed_floats_field(4, float_data)
    if int64_data is not None:
        out += _packed_varints_field(7, int64_data)
    out += _string_field(8, name)
    return bytes(out)


def _node(op_type: str, inputs: list[str], outputs: list[str]) -> bytes:
    out = bytearray()
    for value in inputs:
        out += _string_field(1, value)
    for value in outputs:
        out += _string_field(2, value)
    out += _string_field(4, op_type)
    return bytes(out)


def build_model(weights_data: list[float], graph_name: str) -> bytes:
    weights = _tensor("weights", TENSOR_FLOAT, [FEATURE_DIM, 1], float_data=weights_data)
    axes = _tensor("axes", TENSOR_INT64, [1], int64_data=[1])

    graph = bytearray()
    graph += _bytes_field(1, _node("MatMul", ["input", "weights"], ["matmul_out"]))
    graph += _bytes_field(1, _node("Squeeze", ["matmul_out", "axes"], ["output"]))
    graph += _string_field(2, graph_name)
    graph += _bytes_field(5, weights)
    graph += _bytes_field(5, axes)
    graph += _bytes_field(11, _value_info("input", TENSOR_FLOAT, ["batch", FEATURE_DIM]))
    graph += _bytes_field(12, _value_info("output", TENSOR_FLOAT, ["batch"]))

    model = bytearray()
    model += _varint_field(1, 9)  # ir_version
    model += _string_field(2, "disrust")
    model += _bytes_field(7, bytes(graph))
    opset = _varint_field(2, 13)
    model += _bytes_field(8, opset)
    return bytes(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="tests/models/ort_verify_model.onnx",
        help="Path to write the generated ONNX model",
    )
    parser.add_argument(
        "--mode",
        choices=("weighted-dot", "plain-sum"),
        default="weighted-dot",
        help="Model computation to generate",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "weighted-dot":
        model = build_model(
            [float(i + 1) for i in range(FEATURE_DIM)],
            "disrust_verify_model_weighted_dot",
        )
    else:
        model = build_model([1.0 for _ in range(FEATURE_DIM)], "disrust_verify_model_plain_sum")
    output.write_bytes(model)
    print(output)


if __name__ == "__main__":
    main()
