load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "tf_web_library")
load("@org_tensorflow_tensorboard//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

package(default_visibility = ["//plugin:internal"])

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "trace_viewer",
    srcs = ["trace_viewer.html"],
    path = "/",
    deps = [
        "@org_tensorflow_tensorboard//tensorboard/components/tf_imports:polymer",
        "@org_xprof//plugin/trace_viewer/tf_trace_viewer",
        "@org_xprof//plugin/trace_viewer/webcomponentsjs_polyfill",
    ],
)

tensorboard_html_binary(
    name = "trace_viewer_index",
    input_path = "/trace_viewer.html",
    js_path = "/trace_viewer_index.js",
    output_path = "/trace_viewer_index.html",
    deps = [":trace_viewer"],
)
