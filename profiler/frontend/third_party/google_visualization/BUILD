# Description:
#  typings for google_visualization from http://definitelytyped.org/

load("@npm_bazel_typescript//:index.bzl", "ts_library")

licenses(["notice"])

exports_files(["LICENSE"])

genrule(
    name = "index_d_ts",
    srcs = ["index.d.ts.gz"],
    outs = ["index.d.ts"],
    cmd = "gzip -d -c $(SRCS) > $(OUTS)",
    visibility = [
        "//frontend:__subpackages__", "//plugin:__subpackages__",
    ],
)

ts_library(
    name = "google_visualization",
    srcs = ["index.d.ts"],
    visibility = ["//visibility:public"],
)
