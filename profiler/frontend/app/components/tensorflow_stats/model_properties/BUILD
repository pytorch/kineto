load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "model_properties",
    srcs = [
        "model_properties.ts",
        "model_properties_module.ts",
    ],
    assets = [
        ":model_properties_css",
        "model_properties.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/third_party/google_visualization",
    ],
)



sass_binary(
    name = "model_properties_css",
    src = "model_properties.scss",

    sourcemap = False,
)
