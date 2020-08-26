load("@npm_bazel_typescript//:index.bzl", "ts_library")

package(default_visibility = ["//frontend:internal"])

genrule(
    name = "hlo_proto_ts",
    srcs = ["hlo.proto.ts.gz"],
    outs = ["hlo.proto.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__", "//plugin:__subpackages__",
    ],
)

genrule(
    name = "op_profile_proto_ts",
    srcs = ["op_profile.proto.ts.gz"],
    outs = ["op_profile.proto.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__", "//plugin:__subpackages__",
    ],
)

genrule(
    name = "xla_data_proto_ts",
    srcs = ["xla_data.proto.ts.gz"],
    outs = ["xla_data.proto.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__", "//plugin:__subpackages__",
    ],
)

genrule(
    name = "memory_profile_proto_ts",
    srcs = ["memory_profile.proto.ts.gz"],
    outs = ["memory_profile.proto.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__", "//plugin:__subpackages__",
    ],
)

ts_library(
    name = "op_profile_proto_defs",
    srcs = [
        ":op_profile.proto.ts",
    ],
)

ts_library(
    name = "xla_proto_defs",
    srcs = [
        ":hlo.proto.ts",
        ":xla_data.proto.ts",
    ],
)

ts_library(
    name = "memory_profile_proto_defs",
    srcs = [
        ":memory_profile.proto.ts",
    ],
)

ts_library(
    name = "interfaces",
    srcs = [
        "buffer_allocation_info.ts",
        "capture_profile.ts",
        "data_table.ts",
        "diagnostics.ts",
        "heap_object.ts",
        "navigation_event.ts",
        "op_profile_node.ts",
        "tool.ts",
        "summary_info.ts",
        "window.ts",
    ],
    deps = [
        ":memory_profile_proto_defs",
        ":op_profile_proto_defs",
        ":xla_proto_defs",
    ],
)
