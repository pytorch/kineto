load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary")

package(default_visibility = ["//frontend:internal"])

ng_module(
    name = "empty_page",
    srcs = [
        "empty_page.ts",
        "empty_page_module.ts",
    ],
    assets = [
        ":empty_page_css",
        "empty_page.ng.html",
    ],
    deps = [
        "@npm//@angular/core",
        "@org_xprof//frontend/app/common/interfaces",
        "@org_xprof//frontend/app/components/capture_profile",
    ],
)



sass_binary(
    name = "empty_page_css",
    src = "empty_page.scss",

    sourcemap = False,
    deps = [
        "@org_xprof//frontend/app/styles:common",
    ],
)
