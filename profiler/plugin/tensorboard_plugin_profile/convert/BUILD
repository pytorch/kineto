load("@python_deps//:requirements.bzl", "requirement")

# Description:
#   Converter from protobuf to gviz/json formats.
package(
    default_visibility = ["//plugin:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "all_libs",
    deps = [
        ":diagnostics",
        ":input_pipeline_proto_to_gviz",
        ":kernel_stats_proto_to_gviz",
        ":overview_page_proto_to_gviz",
        ":raw_to_tool_data",
        ":tf_stats_proto_to_gviz",
        ":trace_events_json",
    ],
)

py_library(
    name = "diagnostics",
    srcs = ["diagnostics.py"],
    srcs_version = "PY2AND3",
    deps = [
        requirement("gviz_api"),
    ],
)

py_test(
    name = "diagnostics_test",
    size = "small",
    srcs = ["diagnostics_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":diagnostics",
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "input_pipeline_proto_to_gviz",
    srcs = ["input_pipeline_proto_to_gviz.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":diagnostics",
        requirement("gviz_api"),
    ],
)

py_test(
    name = "input_pipeline_proto_to_gviz_test",
    size = "small",
    srcs = ["input_pipeline_proto_to_gviz_test.py"],
    main = "input_pipeline_proto_to_gviz_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":input_pipeline_proto_to_gviz",
        requirement("absl-py"),
        requirement("gviz_api"),
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "overview_page_proto_to_gviz",
    srcs = ["overview_page_proto_to_gviz.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":diagnostics",
        ":input_pipeline_proto_to_gviz",
        requirement("gviz_api"),
    ],
)

py_test(
    name = "overview_page_proto_to_gviz_test",
    size = "small",
    srcs = ["overview_page_proto_to_gviz_test.py"],
    main = "overview_page_proto_to_gviz_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":overview_page_proto_to_gviz",
        requirement("absl-py"),
        requirement("gviz_api"),
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "tf_stats_proto_to_gviz",
    srcs = ["tf_stats_proto_to_gviz.py"],
    srcs_version = "PY2AND3",
    deps = [requirement("gviz_api")],
)

py_test(
    name = "tf_stats_proto_to_gviz_test",
    size = "small",
    srcs = ["tf_stats_proto_to_gviz_test.py"],
    main = "tf_stats_proto_to_gviz_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":tf_stats_proto_to_gviz",
        requirement("absl-py"),
        requirement("gviz_api"),
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "trace_events_json",
    srcs = ["trace_events_json.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_pythonhosted_six",
    ],
)

py_test(
    name = "trace_events_json_test",
    size = "small",
    srcs = ["trace_events_json_test.py"],
    main = "trace_events_json_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":trace_events_json",
        requirement("absl-py"),
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "kernel_stats_proto_to_gviz",
    srcs = ["kernel_stats_proto_to_gviz.py"],
    srcs_version = "PY2AND3",
    deps = [requirement("gviz_api")],
)

py_test(
    name = "kernel_stats_proto_to_gviz_test",
    size = "small",
    srcs = ["kernel_stats_proto_to_gviz_test.py"],
    main = "kernel_stats_proto_to_gviz_test.py",
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":kernel_stats_proto_to_gviz",
        requirement("absl-py"),
        requirement("gviz_api"),
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "raw_to_tool_data",
    srcs = ["raw_to_tool_data.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":input_pipeline_proto_to_gviz",
        ":kernel_stats_proto_to_gviz",
        ":overview_page_proto_to_gviz",
        ":tf_stats_proto_to_gviz",
        ":trace_events_json",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)
