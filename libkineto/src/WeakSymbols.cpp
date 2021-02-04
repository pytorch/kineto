#include <stdexcept>

#ifndef _MSC_VER
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library when cuPTI is linked statically
// For more details see https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error("Dummy implementation of acc_get_device_type is not supposed to be called!");
}

} // extern "C"
#endif
