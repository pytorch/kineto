load("@python_deps//:requirements.bzl", "requirement")

# Description:
#  TensorBoard plugin for the performance profiling.
package(default_visibility = ["//plugin:internal"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "profile_plugin",
    srcs = ["profile_plugin.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_pythonhosted_six",
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_pocoo_werkzeug",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
        "@org_tensorflow_tensorboard//tensorboard/plugins:base_plugin",
        "@org_xprof//plugin/tensorboard_plugin_profile/convert:all_libs",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "profile_plugin_test_utils",
    srcs = ["profile_plugin_test_utils.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
    ],
    deps = [
        ":profile_plugin",
        "@org_pocoo_werkzeug",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:event_multiplexer",
        "@org_tensorflow_tensorboard//tensorboard/plugins:base_plugin",
    ],
)

py_test(
    name = "profile_plugin_test",
    srcs = ["profile_plugin_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":profile_plugin",
        ":profile_plugin_test_utils",
        requirement("absl-py"),
        "@org_tensorflow_tensorboard//tensorboard:expect_tensorflow_installed",
        "@org_tensorflow_tensorboard//tensorboard/backend/event_processing:plugin_asset_util",
        "@org_xprof//plugin/tensorboard_plugin_profile/protobuf:protos_all_py_pb2",
    ],
)

py_library(
    name = "profile_plugin_loader",
    srcs = ["profile_plugin_loader.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":profile_plugin",
        "@org_tensorflow_tensorboard//tensorboard/plugins:base_plugin",
    ],
)

filegroup(
    name = "resources",
    srcs = [
        "__init__.py",
        "static/index.html",
        "static/index.js",
        "static/materialicons.woff2",
    ],
    visibility = ["//plugin:__subpackages__"],
)
