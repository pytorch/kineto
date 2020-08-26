load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

# Description:
#  Protobufs used by TensorBoard profile plugin.
package(
    default_visibility = ["//plugin:internal"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "protos_all",
    srcs = [
        "diagnostics.proto",
        "input_pipeline.proto",
        "kernel_stats.proto",
        "overview_page.proto",
        "tf_stats.proto",
        "trace_events.proto",
    ],
    deps = ["@com_google_protobuf//:any_proto"],
)

py_proto_library(
    name = "protos_all_py_pb2",
    srcs = [
        "diagnostics.proto",
        "input_pipeline.proto",
        "kernel_stats.proto",
        "overview_page.proto",
        "tf_stats.proto",
        "trace_events.proto",
    ],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    srcs_version = "PY2AND3",
    deps = ["@com_google_protobuf//:protobuf_python"],
)
