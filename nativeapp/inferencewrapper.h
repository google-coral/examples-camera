#ifndef EDGETPU_CPP_EXAMPLES_UTILS_H_
#define EDGETPU_CPP_EXAMPLES_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/interpreter.h"

namespace coral {

class InferenceWrapper {
 public:
  ~InferenceWrapper() = default;

  InferenceWrapper(const std::string& model_path,
                   const std::string& label_path);

  // InferenceWrapper is neither copyable nor movable
  InferenceWrapper(const InferenceWrapper&) = delete;
  InferenceWrapper& operator=(const InferenceWrapper&) = delete;

  // Runs inference using given `interpreter`
  std::pair<std::string, float> RunInference(const uint8_t *input_data,
                                             int input_size);

 private:
  InferenceWrapper() = default;
  std::vector<std::string> labels_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace coral
#endif  // EDGETPU_CPP_EXAMPLES_UTILS_H_
