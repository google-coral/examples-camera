cc_library(
    name = "camerastreamer",
    srcs = ["camerastreamer.cc"],
    hdrs = ["camerastreamer.h"],
    deps = [
	":inferencewrapper",
        "@system_libs//:gstreamer",
        "@system_libs//:gstallocators",
    ],
)

cc_library(
    name = "inferencewrapper",
    srcs = ["inferencewrapper.cc"],
    hdrs = ["inferencewrapper.h"],
    deps = [
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_binary(
    name = "mendelcam",
    srcs = ["main.cc"],
    deps = [
        ":camerastreamer",
        ":inferencewrapper",
    ],
)
