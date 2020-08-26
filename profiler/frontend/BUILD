load("@build_bazel_rules_nodejs//:defs.bzl", "rollup_bundle")
load("@npm_angular_bazel//:index.bzl", "ng_module")
load("@npm_bazel_typescript//:index.bzl", "ts_config", "ts_library")
load("@io_bazel_rules_sass//:defs.bzl", "sass_binary", "sass_library")

package(default_visibility = ["//frontend:__subpackages__", "//plugin:__subpackages__"])

licenses(["notice"])

package_group(
    name = "internal",
    packages = [
      "//frontend/...",
    ],
)

ng_module(
    name = "main",
    srcs = [
        "main.ts",
    ],
    deps = [
        "@npm//@angular/core",
        "@npm//@angular/platform-browser",
        "@org_xprof//frontend/app",
    ],
)

rollup_bundle(
    name = "zone",
    srcs = ["@npm//:node_modules/zone.js/dist/zone.js"],
    entry_point = "@npm//:node_modules/zone.js/dist/zone.js",
)

rollup_bundle(
    name = "bundle",
    entry_point = "main.ts",
    deps = [
        ":main",
        "@npm//@angular/animations",
        "@npm//@angular/cdk",
        "@npm//@angular/material",
        "@npm//@angular/forms",
        "@npm//@angular/core",
        "@npm//@angular/platform-browser",
        "@npm//@angular/router",
        "@npm//@ngrx/store",
    ],
)

sass_library(
    name = "angular_material_theming",
    srcs = [
        "@npm//:node_modules/@angular/material/_theming.scss",
    ]
)

sass_binary(
    name = "css",
    src = "styles.scss",
    output_name = "styles.css",
    sourcemap = False,
    deps = [
        ":angular_material_theming",
    ],
)

py_binary(
    name = "server",
    testonly = True,
    srcs = ["server.py"],
    data = [
        "index.html",
        ":bundle.js",
        ":css",
        ":zone.js",
    ],
    python_version = "PY3",
)
