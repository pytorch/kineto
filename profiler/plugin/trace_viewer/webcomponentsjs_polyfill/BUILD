load("@org_tensorflow_tensorboard//tensorboard/defs:web.bzl", "tf_web_library")

package(default_visibility = ["//plugin:internal"])

licenses(["notice"])  # Apache 2.0

tf_web_library(
    name = "webcomponentsjs_polyfill",
    srcs = [
        "webcomponentsjs_polyfill.html",
        "@org_xprof//plugin/third_party/webcomponentsjs",
    ],
    path = "/webcomponentsjs_polyfill",
)
