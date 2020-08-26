load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "tf_web_library")

package(default_visibility = ["//plugin:internal"])

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "tf_trace_viewer",
    srcs = [
        "tf-trace-viewer.html",
        "@org_xprof//plugin/third_party/tracing:trace_viewer_full.html",
    ],
    path = "/tf-trace-viewer",
    deps = [
        ":tf-trace-viewer-helper",
    ],
)

tf_web_library(
    name = "demo",
    srcs = ["demo.html"],
    path = "/tf-trace-viewer",
    deps = [
        ":tf_trace_viewer",
        "@org_xprof//plugin/trace_viewer/tf_trace_viewer/data",
    ],
)

tf_web_library(
    name = "tf-trace-viewer-helper",
    srcs = [
        "tf-trace-viewer-helper.js",
    ],
    path = "/tf-trace-viewer",
)
