
#include <exception>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

using namespace tensorflow;


REGISTER_OP("OpenFstLoad")
.Attr("filename: string")
.Attr("container: string = ''")
.Attr("shared_name: string = ''")
.Output("handle: resource")
.SetIsStateful()
.SetShapeFn(shape_inference::ScalarShape)
.Doc("OpenFstLoad: loads FST, creates TF resource, persistent across runs in the session");


REGISTER_OP("OpenFstTransition")
.Input("handle: resource")
.Input("states: int32")
.Input("inputs: int32")
.Output("new_states: int32")
.Output("scores: float32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(1));
  c->set_output(1, c->input(1));
  return Status::OK();
})
.Doc("OpenFstTransition: performs a transition");


struct OpenFstInstance : public ResourceBase {
  explicit OpenFstInstance(const string& filename) : filename_(filename) {}

  string DebugString() override {
    return strings::StrCat("OpenFstInstance[", filename_, "]");
  }

  const string filename_;
};


// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/resource_op_kernel.h
// TFUtil.TFArrayContainer
class OpenFstLoadOp : public ResourceOpKernel<OpenFstInstance> {
 public:
  explicit OpenFstLoadOp(OpKernelConstruction* context)
      : ResourceOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filename", &filename_));
  }

 private:
  virtual bool IsCancellable() const { return false; }
  virtual void Cancel() {}

  Status CreateResource(OpenFstInstance** ret) override EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    try {
      *ret = new OpenFstInstance(filename_);
    } catch (std::exception& exc) {
      return errors::Internal("Could not load OpenFst ", filename_, ", exception: ", exc.what());
    }
    if(*ret == nullptr)
      return errors::ResourceExhausted("Failed to allocate");
    return Status::OK();
  }

  Status VerifyResource(OpenFstInstance* fst) override {
    if(fst->filename_ != filename_)
      return errors::InvalidArgument("Filename mismatch: expected ", filename_,
                                     " but got ", fst->filename_, ".");
    return Status::OK();
  }

  string filename_;
};

REGISTER_KERNEL_BUILDER(Name("OpenFstLoad").Device(DEVICE_CPU), OpenFstLoadOp);


class OpenFstTransitionOp : public OpKernel {
 public:
  using OpKernel::OpKernel;

  void Compute(OpKernelContext* context) override {
    OpenFstInstance* fst;
    OP_REQUIRES_OK(context, GetResourceFromContext(context, "handle", &fst));
    core::ScopedUnref unref(fst);

    const Tensor& states_tensor = context->input(1);
    auto states_flat = states_tensor.flat<int32>();

    const Tensor& inputs_tensor = context->input(2);
    auto inputs_flat = inputs_tensor.flat<int32>();

    OP_REQUIRES(
      context,
      TensorShapeUtils::IsVector(states_tensor.shape()) &&
      TensorShapeUtils::IsVector(inputs_tensor.shape()) &&
      states_flat.size() == inputs_flat.size(),
      errors::InvalidArgument(
        "Shape mismatch. states ", states_tensor.shape().DebugString(),
        " vs inputs ", inputs_tensor.shape().DebugString()));

    Tensor* output_new_states_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, states_tensor.shape(), &output_new_states_tensor));
    auto output_new_states_flat = output_new_states_tensor->flat<int32>();
    Tensor* output_scores_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, states_tensor.shape(), &output_scores_tensor));
    auto output_scores_flat = output_scores_tensor->flat<float>();

    for(int i = 0; i < inputs_flat.size(); ++i) {
      output_new_states_flat(i) = -1;  // TODO
      output_scores_flat(i) = -1.;  // TODO
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("OpenFstTransition").Device(DEVICE_CPU), OpenFstTransitionOp);
