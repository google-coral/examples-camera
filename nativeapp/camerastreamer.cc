#include "camerastreamer.h"

#define GST_MINIMAL_CHECK(x)                                 \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                      \
  }

namespace coral {

namespace {

GstFlowReturn OnNewSample(GstElement *sink, void *data) {
  GstSample *sample;
  GstFlowReturn retval = GST_FLOW_OK;

  g_signal_emit_by_name(sink, "pull-sample", &sample);
  if (sample) {
    GstMapInfo info;
    auto buf = gst_sample_get_buffer(sample);
    if (gst_buffer_map(buf, &info, GST_MAP_READ) == TRUE) {
      // Pass the frame to the user callback
      auto user_data = reinterpret_cast<CameraStreamer::UserData *>(data);
      user_data->f(info.data, info.size, user_data->args);
    } else {
      g_printerr("Error: Couldn't get buffer info\n");
      retval = GST_FLOW_ERROR;
    }

    gst_buffer_unmap(buf, &info);
    gst_sample_unref(sample);
  }
  return retval;
}

gboolean OnBusMessage(GstBus *bus, GstMessage *msg, gpointer data) {
  GMainLoop *loop = reinterpret_cast<GMainLoop *>(data);

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print("End of stream\n");
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError *error;
      gst_message_parse_error(msg, &error, nullptr);
      g_printerr("Error: %s\n", error->message);
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *error;
      gst_message_parse_warning(msg, &error, nullptr);
      g_printerr("Warning: %s\n", error->message);
      g_error_free(error);
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

}  //  namespace

void CameraStreamer::RunPipeline(const gchar *pipeline_string,
                                 UserData user_data) {
  gst_init(nullptr, nullptr);

  // Set up a pipeline based on the pipeline string
  auto loop = g_main_loop_new(nullptr, FALSE);
  GST_MINIMAL_CHECK(loop != nullptr);
  auto pipeline = gst_parse_launch(pipeline_string, nullptr);
  GST_MINIMAL_CHECK(pipeline != nullptr);

  // Add a bus watcher. It's safe to unref the bus immediately after
  auto bus = gst_element_get_bus(pipeline);
  GST_MINIMAL_CHECK(bus != nullptr);

  gst_bus_add_watch(bus, OnBusMessage, loop);
  gst_object_unref(bus);

  // Set up an appsink to pass frames to a user callback
  auto appsink =
      gst_bin_get_by_name(reinterpret_cast<GstBin *>(pipeline), "appsink");
  GST_MINIMAL_CHECK(appsink != nullptr)

  g_object_set(appsink, "emit-signals", true, nullptr);
  g_signal_connect(appsink, "new-sample",
                   reinterpret_cast<GCallback>(OnNewSample), &user_data);

  // Start the pipeline, runs until interrupted, EOS or error
  gst_element_set_state(pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  // Cleanup
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}

}  // namespace coral
