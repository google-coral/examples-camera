package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64",
    values = {"cpu": "aarch64"},
)

config_setting(
    name = "armv7a",
    values = {"cpu": "armv7a"},
)

config_setting(
    name = "k8",
    values = {"cpu": "k8"},
)

UNSUPPORTED_CPU_ERROR = "Unsupported target CPU"

cc_library(
    name = "gstreamer",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstreamer-1.0.so*"]),
            ":armv7a": glob(["usr/lib/arm-linux-gnueabihf/libgstreamer-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstreamer-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/*.h",
        ],
    ) + select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/gstreamer-1.0/include/gst/*.h"]),
            ":armv7a": glob(["usr/lib/arm-linux-gnueabihf/gstreamer-1.0/include/gst/*.h"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/gst/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    includes = ["usr/include/gstreamer-1.0"] + select(
        {
            ":aarch64": ["usr/lib/aarch64-linux-gnu/gstreamer-1.0/include"],
            ":armv7a": ["usr/lib/arm-linux-gnueabihf/gstreamer-1.0/include"],
            ":k8": ["usr/lib/x86_64-linux-gnu/gstreamer-1.0/include"],
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    linkstatic = 0,
    deps = [
        ":glib",
    ],
)

cc_library(
    name = "gstallocators",
    srcs = select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/libgstallocators-1.0.so*"]),
            ":armv7a": glob(["usr/lib/arm-linux-gnueabihf/libgstallocators-1.0.so*"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/libgstallocators-1.0.so*"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/gstreamer-1.0/gst/allocators/*.h",
        ],
    ),
    includes = ["usr/include/gstreamer-1.0"],
    linkstatic = 0,
    deps = [
        ":gstreamer",
    ],
)

cc_library(
    name = "glib",
    srcs = select(
        {
            ":aarch64": glob([
                "usr/lib/aarch64-linux-gnu/libgobject-2.0.so.0*",
                "lib/aarch64-linux-gnu/libgobject-2.0.so.0*",
                "usr/lib/aarch64-linux-gnu/libglib-2.0.so.0*",
                "lib/aarch64-linux-gnu/libglib-2.0.so.0*",
            ]),
            ":armv7a": glob([
                "usr/lib/arm-linux-gnueabihf/libgobject-2.0.so.0*",
                "lib/arm-linux-gnueabihf/libgobject-2.0.so.0*",
                "usr/lib/arm-linux-gnueabihf/libglib-2.0.so.0*",
                "lib/arm-linux-gnueabihf/libglib-2.0.so.0*",
            ]),
            ":k8": glob([
                "usr/lib/x86_64-linux-gnu/libgobject-2.0.so.0*",
                "lib/x86_64-linux-gnu/libgobject-2.0.so.0*",
                "usr/lib/x86_64-linux-gnu/libglib-2.0.so.0*",
                "lib/x86_64-linux-gnu/libglib-2.0.so.0*",
            ]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    hdrs = glob(
        [
            "usr/include/glib-2.0/*.h",
            "usr/include/glib-2.0/*/*.h",
            "usr/include/glib-2.0/glib/deprecated/*.h",
            "usr/include/glib-2.0/gobject/*.h",
        ],
    ) + select(
        {
            ":aarch64": glob(["usr/lib/aarch64-linux-gnu/glib-2.0/include/*.h"]),
            ":armv7a": glob(["usr/lib/arm-linux-gnueabihf/glib-2.0/include/*.h"]),
            ":k8": glob(["usr/lib/x86_64-linux-gnu/glib-2.0/include/*.h"]),
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    includes = ["usr/include/glib-2.0"] + select(
        {
            ":aarch64": ["usr/lib/aarch64-linux-gnu/glib-2.0/include"],
            ":armv7a": ["usr/lib/arm-linux-gnueabihf/glib-2.0/include"],
            ":k8": ["usr/lib/x86_64-linux-gnu/glib-2.0/include"],
        },
        no_match_error = UNSUPPORTED_CPU_ERROR,
    ),
    linkstatic = 0,
)
