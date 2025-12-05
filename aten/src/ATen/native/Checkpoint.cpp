#include <ATen/ATen.h>
#include <ATen/CheckpointTensorImpl.h>
#include <iostream>

namespace at {
namespace native {

Tensor checkpoint_add(const Tensor& a, const Tensor& b, const c10::Scalar& c) {
  std::cout << "checkpoint_add \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::add(vec.at(0), vec.at(1), c)};
  };
  // //std::coutrunning checkpoint implementation tensor " << std::endl;
  return CheckpointTensorImpl::make("add", rt, {a, b})[0];
}

Tensor checkpoint_bmm(const Tensor& a, const Tensor& b) {
  std::cout << "checkpoint_bmm \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::bmm(vec.at(0), vec.at(1))};
  };
  // //std::coutrunning checkpoint implementation tensor " << std::endl;
  return CheckpointTensorImpl::make("bmm", rt, {a, b})[0];
}

Tensor checkpoint_unsafe_view(const Tensor& a, IntArrayRef b) {
  std::cout << "checkpoint unsafe view  \n";
  auto b_ = b.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::_unsafe_view(vec.at(0), b_)};
  };
  // //std::coutrunning checkpoint implementation tensor " << std::endl;
  return CheckpointTensorImpl::make("_unsafe_view", rt, {a})[0];
}

Tensor checkpoint_cat(const ITensorListRef& qxs, int64_t dim) {
  std::cout << "checkpoint cat\n";
  std::vector<Tensor> qxs_copy;
  qxs_copy.reserve(qxs.size());
  for (const auto& t : qxs) {
    qxs_copy.push_back(t);
  }

  std::cout << "checkpoint_cat\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::cat(vec, dim)};
  };
  // //std::coutrunning checkpoint implementation tensor " << std::endl;
  return CheckpointTensorImpl::make("cat", rt, qxs_copy)[0];
}

Tensor checkpoint_mean(
    const Tensor& self,
    c10::optional<c10::ScalarType> dtype) {
  std::cout << "checkpint mean scalar\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::mean(vec[0], dtype)};
  };
  return CheckpointTensorImpl::make("mean", rt, {self})[0];
}

Tensor checkpoint_permute(const Tensor& self, IntArrayRef dim) {
  std::cout << "checkpint permute\n";
  auto dim_ = dim.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::permute(vec[0], dim)};
  };
  return CheckpointTensorImpl::make("permute", rt, {self})[0];
}

Tensor checkpoint_t(const Tensor& a) {
  std::cout << "checkpoint_t\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::t(vec.at(0))};
  };
  return CheckpointTensorImpl::make("t", rt, {a})[0];
}

Tensor checkpoint_index(
    const Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices) {
  std::cout << "checkpoint index\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    auto self_local = vec.at(0);

    torch::List<c10::optional<Tensor>> local_indices;
    for (size_t i = 1; i < vec.size(); ++i) {
      local_indices.push_back(vec[i]);
    }
    return {at::index(self_local, local_indices)};
  };

  std::vector<Tensor> inputs;
  inputs.push_back(self);
  for (auto i : c10::irange(indices.size())) {
    auto index = indices.get(i);
    if (index.has_value() && index->numel() > 0)
      inputs.push_back(*index);
  }

  return CheckpointTensorImpl::make("index", rt, inputs)[0];
}

Tensor checkpoint_mean(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<c10::ScalarType> dtype) {
  std::cout << "checkpoint mean\n";
  at::OptionalIntArrayRef dim_vec;
  if (dim.has_value()) {
    dim_vec = dim.value().vec();
  }
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::mean(vec[0], dim, keepdim, dtype)};
  };
  return CheckpointTensorImpl::make("mean.dim", rt, {self})[0];
}
Tensor checkpoint_unsqueeze(const Tensor& self, int64_t dim) {
  std::cout << "checkpoint unsqueez\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::unsqueeze(vec[0], dim)};
  };
  return CheckpointTensorImpl::make("unsqueeze", rt, {self})[0];
}
Tensor checkpoint_neg(at::Tensor const& a) {
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::neg(vec.at(0))};
  };
  return CheckpointTensorImpl::make("neg", rt, {a})[0];
}
Tensor checkpoint_transpose(at::Tensor const& a, int64_t b, int64_t c) {
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::transpose(vec.at(0), b, c)};
  };
  return CheckpointTensorImpl::make("transpose_int", rt, {a})[0];
}
Tensor checkpoint_baddbmm(
    at::Tensor const& a,
    at::Tensor const& b,
    at::Tensor const& c,
    const c10::Scalar& d,
    const c10::Scalar& e) {
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::baddbmm(vec.at(0), vec.at(1), vec.at(2), d, e)};
  };
  return CheckpointTensorImpl::make("baddbmm", rt, {a, b, c})[0];
}

Tensor& checkpoint_add_(Tensor& a, const Tensor& b, const c10::Scalar& c) {
  std::cout << "checkpoint add_\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    vec.at(0).add_(vec.at(1), c);
  };
  CheckpointTensorImpl::mutate("add_", mt, {a, b}, {0});
  return a;
}

Tensor checkpoint_add(
    at::Tensor const& a,
    const c10::Scalar& b,
    const c10::Scalar& c) {
  std::cout << "checkpoint add\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::add(vec.at(0), b, c)};
  };
  return CheckpointTensorImpl::make("add", rt, {a})[0];
}

Tensor checkpoint_rand_like(
    at::Tensor const& a,
    c10::optional<ScalarType> b,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> mem_format) {
  std::cout << "checkpoint rand like\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {
        at::rand_like(vec.at(0), b, layout, device, pin_memory, mem_format)};
  };
  return CheckpointTensorImpl::make("rand_like", rt, {a})[0];
}

Tensor checkpoint_thnn_conv2d_forward(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias_opt,
    IntArrayRef stride,
    IntArrayRef padding) {
  std::cout << "checkpoint conv forward\n";
  bool has_bias = bias_opt.has_value();
  auto kernel_size_vec = kernel_size.vec();
  auto stride_vec = stride.vec();
  auto padding_vec = padding.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    c10::optional<Tensor> bias;
    if (has_bias && vec.size() > 2 && vec.at(2).numel() > 0) {
      bias = vec.at(2);
    }
    return {at::_slow_conv2d_forward(
        vec.at(0), vec.at(1), kernel_size_vec, bias, stride_vec, padding_vec)};
  };

  std::vector<Tensor> args;

  args.push_back(self);
  args.push_back(weight);
  if (bias_opt.has_value() && bias_opt->numel() > 0)
    args.push_back(*bias_opt);
  return CheckpointTensorImpl::make("thnn_conv2d", rt, args)[0];
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps) {
  std::cout << "checkpoint norm native\n";

  // Store which optionals actually exist
  const bool has_weight = weight_opt.has_value();
  const bool has_bias = bias_opt.has_value();
  const bool has_running_mean = running_mean_opt.has_value();
  const bool has_running_var = running_var_opt.has_value();

  // Create device-correct empty tensors for missing args

  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    TORCH_INTERNAL_ASSERT(vec.size() == 5);

    c10::optional<Tensor> weight;
    if (has_weight && vec.at(1).numel() > 0)
      weight = vec.at(1);

    c10::optional<Tensor> bias;
    if (has_bias && vec.at(2).numel() > 0)
      bias = vec.at(2);

    c10::optional<Tensor> running_mean;
    if (has_running_mean && vec.at(3).numel() > 0)
      running_mean = vec.at(3);

    c10::optional<Tensor> running_var;
    if (has_running_var && vec.at(4).numel() > 0)
      running_var = vec.at(4);

    auto ret = at::native_batch_norm(
        vec.at(0),
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);

    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto make_tensor = [&](const c10::optional<Tensor>& opt) -> Tensor {
    if (opt.has_value() && opt->numel() > 0)
      return *opt;
    else
      return at::empty({0}, input.options());
  };

  std::vector<Tensor> args;
  args.reserve(5);
  args.push_back(input);
  args.push_back(make_tensor(weight_opt));
  args.push_back(make_tensor(bias_opt));
  args.push_back(make_tensor(running_mean_opt));
  args.push_back(make_tensor(running_var_opt));
  auto out = CheckpointTensorImpl::make("native_batch_norm", rt, args);
  return {out[0], out[1], out[2]};
}

std::tuple<Tensor&, Tensor&, Tensor&> checkpoint_native_batch_norm_out(
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_invstd) {
  std::cout << "checkpoint norm out\n";

  const bool has_weight = weight_opt.has_value();
  const bool has_bias = bias_opt.has_value();
  const bool has_running_mean = running_mean_opt.has_value();
  const bool has_running_var = running_var_opt.has_value();
  auto make_tensor = [&](const c10::optional<Tensor>& opt, const Tensor& ref) {
    return opt.has_value() ? *opt
                           : at::empty({0}, ref.options()); // safe empty tensor
  };

  std::vector<Tensor> vec_args = {
      input,
      make_tensor(weight_opt, input),
      make_tensor(bias_opt, input),
      make_tensor(running_mean_opt, input),
      make_tensor(running_var_opt, input),
      out,
      save_mean,
      save_invstd};

  mutate_function_t mt = [=](const Tensors& vec) {
    TORCH_INTERNAL_ASSERT(vec.size() == 8);

    Tensor out_ = vec.at(5);
    Tensor save_mean_ = vec.at(6);
    Tensor save_invstd_ = vec.at(7);

    c10::optional<Tensor> weight;
    if (has_weight && vec.at(1).numel() > 0)
      weight = vec.at(1);

    c10::optional<Tensor> bias;
    if (has_bias && vec.at(2).numel() > 0)
      bias = vec.at(2);

    c10::optional<Tensor> running_mean;
    if (has_running_mean && vec.at(3).numel() > 0)
      running_mean = vec.at(3);

    c10::optional<Tensor> running_var;
    if (has_running_var && vec.at(4).numel() > 0)
      running_var = vec.at(4);

    at::native_batch_norm_out(
        out_,
        save_mean_,
        save_invstd_,
        vec.at(0),
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
  };

  CheckpointTensorImpl::mutate(
      "native_batch_norm_out", mt, vec_args, {5, 6, 7});

  return {out, save_mean, save_invstd};
}

Tensor checkpoint_view(const Tensor& a, IntArrayRef b) {
  std::cout << "view\n";
  auto b_ = b.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {vec.at(0).view(b_)};
  };
  return CheckpointTensorImpl::make("view", rt, {a})[0];
}

Tensor checkpoint_detach(const Tensor& a) {
  std::cout << "checkpoint detach\n";
  return Tensor(get_cpti(a)->shallow_copy_and_detach(
      // NB: The ADInplaceOrView logic will overwrite these with the
      // appropriate values if it runs; otherwise these are the values.
      /*version_counter=*/0,
      /*allow_tensor_metadata_change=*/false));
}

Tensor checkpoint_abs(const Tensor& a) {
  std::cout << "abs\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::abs(vec.at(0))};
  };
  return CheckpointTensorImpl::make("abs", rt, {a})[0];
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_thnn_conv2d_backward(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    std::array<bool, 3ul> output_mask) {
  auto kernel_size_vec = kernel_size.vec();
  auto stride_vec = stride.vec();
  auto padding_vec = padding.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    auto ret = at::_slow_conv2d_backward(
        grad_output,
        self,
        weight,
        kernel_size_vec,
        stride_vec,
        padding_vec,
        output_mask);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make(
      "_slow_conv2d_backward", rt, {grad_output, self, weight});
  return {ret[0], ret[1], ret[2]};
}
Tensor checkpoint_as_strided(
    const at::Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  // std::cout << "as strided\n";

  auto stride_vec = stride.vec();
  auto size_vec = size.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::as_strided(vec.at(0), size_vec, stride_vec, storage_offset)};
  };
  return CheckpointTensorImpl::make("as_strided", rt, {self})[0];
}
Tensor checkpoint_div(const Tensor& a, const Tensor& b) {
  // std::cout << "checkpoint_div";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::div(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("div", rt, {a, b})[0];
}

Tensor& checkpoint_div_(Tensor& a, const Tensor& b) {
  std::cout << "checkpoint_div_ \n";
  mutate_function_t mt = [=](const Tensors& vec) { vec.at(0).div_(vec.at(1)); };
  CheckpointTensorImpl::mutate("div_", mt, {a, b}, {0});
  return a;
}
Tensor checkpoint_kl_div(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    bool log_target) {
  std::cout << "kl_div \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::kl_div(vec.at(0), vec.at(1), reduction, log_target)};
  };
  return CheckpointTensorImpl::make("kl_div", rt, {self, target})[0];
}
Tensor checkpoint_mul(at::Tensor const& a, at::Tensor const& b) {
  std::cout << "checkpoint mul\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::mul(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("mul", rt, {a, b})[0];
}

Tensor& checkpoint_mul_(at::Tensor& a, at::Tensor const& b) {
  std::cout << "checkpoint mul_ \n";
  mutate_function_t mt = [=](const Tensors& vec) { vec.at(0).mul_(vec.at(1)); };
  CheckpointTensorImpl::mutate("mul_", mt, {a, b}, {0});
  return a;
}

Tensor& checkpoint_mul_(at::Tensor& a, const c10::Scalar& b) {
  std::cout << "checkpoint mul scalar\n";
  mutate_function_t mt = [=](const Tensors& vec) { vec.at(0).mul_(b); };
  CheckpointTensorImpl::mutate("mul_", mt, {a}, {0});
  return a;
}
Tensor checkpoint_mul(at::Tensor const& a, const c10::Scalar& b) {
  std::cout << "checkpoint mul scalar\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::mul(vec.at(0), b)};
  };
  return CheckpointTensorImpl::make("mul", rt, {a})[0];
}
Tensor checkpoint_sum(const Tensor& a, c10::optional<ScalarType> b) {
  std::cout << "checkpoint sum\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::sum(vec.at(0), b)};
  };
  return CheckpointTensorImpl::make("sum", rt, {a})[0];
}

Tensor checkpoint_sum_dim_IntList(
    const Tensor& a,
    at::OptionalIntArrayRef b,
    bool c,
    c10::optional<ScalarType> d) {
  std::cout << "checkpoint sum int list \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::sum(vec.at(0), b, c, d)};
  };
  return CheckpointTensorImpl::make("sum_dim_IntList", rt, {a})[0];
}

Tensor checkpoint_cudnn_convolution(
    const Tensor& a,
    const Tensor& b,
    IntArrayRef c,
    IntArrayRef d,
    IntArrayRef e,
    int64_t f,
    bool g,
    bool h,
    bool i) {
  std::cout << "checkpoint_cudnn_convolution\n";
  auto c_ = c.vec();
  auto d_ = d.vec();
  auto e_ = e.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {
        at::cudnn_convolution(vec.at(0), vec.at(1), c_, d_, e_, f, g, h, i)};
  };
  return CheckpointTensorImpl::make("cudnn_convolution", rt, {a, b})[0];
}

Tensor checkpoint_cudnn_convolution_transpose(
    const Tensor& a,
    const Tensor& b,
    IntArrayRef c,
    IntArrayRef d,
    IntArrayRef e,
    IntArrayRef f,
    int64_t g,
    bool h,
    bool i,
    bool j) {
  std::cout << "checkpoint_cudnn_convolution_transpose\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::cudnn_convolution_transpose(
        vec.at(0), vec.at(1), c, d, e, f, g, h, i, j)};
  };
  return CheckpointTensorImpl::make(
      "cudnn_convolution_transpose", rt, {a, b})[0];
}

Tensor checkpoint_relu(const Tensor& a) {
  std::cout << "checkpoint relu\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::relu(vec.at(0))};
  };
  return CheckpointTensorImpl::make("relu", rt, {a})[0];
}

Tensor& checkpoint_relu_(Tensor& a) {
  std::cout << "checkpoint relu_ \n";
  mutate_function_t mt = [=](const Tensors& vec) { vec.at(0).relu_(); };
  CheckpointTensorImpl::mutate("relu_", mt, {a}, {0});
  return a;
}
Tensor checkpoint_ne_Scalar(const Tensor& a, const c10::Scalar& b) {
  std::cout << "scalar\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::ne(vec.at(0), b)};
  };
  return CheckpointTensorImpl::make("ne_Scalar", rt, {a})[0];
}
Tensor checkpoint_eq_Tensor(const Tensor& a, const Tensor& b) {
  std::cout << "eq tensor\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::eq(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("eq_Tensor", rt, {a, b})[0];
}
void annotate_log() {}
Tensor& checkpoint_eq_Tensor_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  std::cout << "eq tensor out\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    at::eq_out(const_cast<Tensor&>(out), self, other);
  };
  CheckpointTensorImpl::mutate("eq_Tensor_out", mt, {self, other, out}, {0});
  return out;
}

Tensor checkpoint_eq_Scalar(const Tensor& a, const c10::Scalar& b) {
  std::cout << "eq scalar\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::eq(vec.at(0), b)};
  };
  return CheckpointTensorImpl::make("eq_Scalar", rt, {a})[0];
}

Tensor& checkpoint_eq_Scalar_out(
    const Tensor& self,
    const c10::Scalar& other,
    Tensor& out) {
  std::cout << "checkpoint scalar out\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    at::eq_out(const_cast<Tensor&>(out), self, other);
  };
  CheckpointTensorImpl::mutate("eq_Scalar_out", mt, {self, out}, {0});
  return out;
}
Tensor& checkpoint_bitwise_and_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  std::cout << "bitwaise and out \n";
  mutate_function_t mt = [=](const Tensors& vec) {
    at::bitwise_and_out(const_cast<Tensor&>(out), self, other);
  };
  CheckpointTensorImpl::mutate("bitwise_and_out", mt, {self, other, out}, {0});
  return out;
}

Tensor& checkpoint_bitwise_and_out(
    const Tensor& self,
    const c10::Scalar& other,
    Tensor& out) {
  std::cout << "bit wise and out with scalar\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor self = vec.at(0);
    at::bitwise_and_out(const_cast<Tensor&>(out), self, other);
  };
  CheckpointTensorImpl::mutate("bitwise_and_out", mt, {self, out}, {0});
  return out;
}
void checkpoint_print(c10::string_view s) {
  std::cout << "printing:\n";
  std::cout << s << "\n";
}

Tensor checkpoint_bitwise_and(at::Tensor const& a, at::Tensor const& b) {
  return {at::bitwise_and(a, b)};
}

Tensor checkpoint_clone(
    at::Tensor const& a,
    c10::optional<c10::MemoryFormat> b) {
  std::cout << "checkpoint clone\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::clone(vec.at(0), b)};
  };
  return CheckpointTensorImpl::make("clone", rt, {a})[0];
}

Tensor checkpoint_addmm(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const c10::Scalar& d,
    const c10::Scalar& e) {
  std::cout << "checkpoint addmm\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::addmm(vec.at(0), vec.at(1), vec.at(2), d, e)};
  };
  return CheckpointTensorImpl::make("addmm", rt, {a, b, c})[0];
}

Tensor& checkpoint_addmm_out(
    const Tensor& b,
    const Tensor& c,
    const Tensor& d,
    const c10::Scalar& e,
    const c10::Scalar& f,
    Tensor& a) {
  std::cout << "checkpoint addmm out\n";

  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor a_ = vec.at(0);
    at::addmm_out(a_, vec.at(1), vec.at(2), d, e, f);
  };

  CheckpointTensorImpl::mutate("addmm_out", mt, {a, b, c}, {0});

  return a;
}

Tensor& checkpoint_addmm_(
    Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const c10::Scalar& d,
    const c10::Scalar& e) {
  std::cout << "checkpoint addmm_\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor a_ = vec.at(0);
    a.addmm_(vec.at(1), vec.at(2), d, e);
  };
  CheckpointTensorImpl::mutate("addmm_", mt, {a, b, c}, {0});
  return a;
}
Tensor checkpoint__log_softmax(const Tensor& a, int64_t b, bool c) {
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::_log_softmax(vec.at(0), b, c)};
  };
  return CheckpointTensorImpl::make("_log_softmax", rt, {a})[0];
}

std::tuple<Tensor, Tensor> checkpoint_nll_loss_forward(
    const Tensor& a,
    const Tensor& b,
    const c10::optional<Tensor>& weight_opt,
    int64_t d,
    int64_t e) {
  std::cout << "checkpoint nll forward\n";
  const bool has_weight = weight_opt.has_value();
  auto make_tensor = [&](const c10::optional<Tensor>& opt) -> Tensor {
    if (opt.has_value()) {
      const Tensor& t = opt.value();
      if (t.numel() == 0) {
        return at::empty(t.sizes(), a.options());
      } else {
        return t;
      }
    } else {
      return at::empty({0}, a.options());
    }
  };
  std::vector<Tensor> args;
  args.reserve(3);
  args.push_back(a);
  args.push_back(b);
  args.push_back(make_tensor(weight_opt));
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    c10::optional<Tensor> weight;
    if (has_weight && vec.at(2).numel() > 0)
      weight = vec.at(2);

    auto ret = at::nll_loss_forward(vec.at(0), vec.at(1), weight, d, e);
    return {std::get<0>(ret), std::get<1>(ret)};
  };

  auto ret = CheckpointTensorImpl::make("nll_loss_forward", rt, args);
  return {ret[0], ret[1]};
}

std::tuple<Tensor&, Tensor&> checkpoint_nll_loss_forward_out(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  std::cout << "checkpoint nll loss forward out\n";
  const bool has_weight = weight_opt.has_value();
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor output_ = vec.at(3);
    Tensor total_weight_ = vec.at(4);
    c10::optional<Tensor> weight;
    if (has_weight && vec.at(2).numel() > 0)
      weight = vec.at(2);
    at::nll_loss_forward_out(
        output_,
        total_weight_,
        vec.at(0),
        vec.at(1),
        weight,
        reduction,
        ignore_index);
  };

  auto make_tensor = [&](const c10::optional<Tensor>& opt) -> Tensor {
    if (opt.has_value())
      return *opt;
    else
      return at::empty({0}, self.options());
  };

  std::vector<Tensor> args;
  args.push_back(output);
  args.push_back(total_weight);
  args.push_back(self);
  args.push_back(target);
  args.push_back(make_tensor(weight_opt));

  CheckpointTensorImpl::mutate("nll_loss_forward_out", mt, args, {0, 1});

  return {output, total_weight};
}
Tensor checkpoint_ones_like(
    const Tensor& a,
    c10::optional<ScalarType> b,
    c10::optional<c10::Layout> layout,
    c10::optional<c10::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> mem_format) {
  std::cout << "checkpoint ones_like\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {
        at::ones_like(vec.at(0), b, layout, device, pin_memory, mem_format)};
  };
  return CheckpointTensorImpl::make("ones_like", rt, {a})[0];
}

Tensor checkpoint_nll_loss_backward(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const c10::optional<Tensor>& weight_opt,
    int64_t e,
    int64_t f,
    const Tensor& g) {
  std::cout << "checkpoint nll loss backward\n";
  const bool has_weight = weight_opt.has_value();
  auto make_tensor = [&](const c10::optional<Tensor>& opt) -> Tensor {
    if (opt.has_value()) {
      const Tensor& t = opt.value();
      if (t.numel() == 0) {
        return at::empty({0}, a.options());
      } else {
        return t;
      }
    } else {
      return at::empty({0}, a.options());
    }
  };

  std::vector<Tensor> args;
  args.reserve(5);
  args.push_back(a);
  args.push_back(b);
  args.push_back(c);
  args.push_back(make_tensor(weight_opt));
  args.push_back(g);

  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    c10::optional<Tensor> weight;
    if (has_weight && vec.at(3).numel() > 0)
      weight = vec.at(3);
    return {at::nll_loss_backward(
        vec.at(0), vec.at(1), vec.at(2), weight, e, f, vec.at(4))};
  };

  return CheckpointTensorImpl::make("nll_loss_backward", rt, args)[0];
}

Tensor& checkpoint_nll_loss_backward_grad_input(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const c10::optional<Tensor>& weight_opt,
    int64_t f,
    int64_t g,
    const Tensor& total_weight,
    Tensor& h) {
  std::cout << "checkpoint nll loss backward grad input \n";
  const bool has_weight = weight_opt.has_value();
  auto make_tensor = [&](const c10::optional<Tensor>& opt) -> Tensor {
    if (opt.has_value())
      return *opt;
    else
      return at::empty({0}, a.options());
  };

  std::vector<Tensor> args;
  args.push_back(h);
  args.push_back(a);
  args.push_back(b);
  args.push_back(c);
  args.push_back(make_tensor(weight_opt));
  args.push_back(total_weight);

  mutate_function_t mt = [=](const Tensors& vec) {
    c10::optional<Tensor> weight;
    if (has_weight && vec.at(4).numel() > 0)
      weight = vec.at(4);

    Tensor h_ = vec.at(0);
    at::nll_loss_backward_out(
        h_, vec.at(1), vec.at(2), vec.at(3), weight, f, g, vec.at(5));
  };

  CheckpointTensorImpl::mutate("nll_loss_backward_grad_input", mt, args, {0});

  return h;
}

Tensor checkpoint_log_softmax_backward_data(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    c10::ScalarType input_dtype) {
  std::cout << "log softmax backward data\n";
  auto rt = [=](const Tensors& vec) -> Tensors {
    return {
        at::_log_softmax_backward_data(vec.at(0), vec.at(1), dim, input_dtype)};
  };
  return CheckpointTensorImpl::make(
      "_log_softmax_backward_data", rt, {grad_output, output})[0];
}
Tensor checkpoint_mm(const Tensor& a, const Tensor& b) {
  std::cout << "checkpoint mm\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::mm(vec.at(0), vec.at(1))};
  };
  return CheckpointTensorImpl::make("mm", rt, {a, b})[0];
}

Tensor& checkpoint_mm_out(const Tensor& b, const Tensor& c, Tensor& a) {
  std::cout << "checkpoint mm out\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor a_ = vec.at(2);
    at::mm_out(a_, vec.at(0), vec.at(1));
  };
  CheckpointTensorImpl::mutate("mm_out", mt, {a, b, c}, {0});
  return a;
}
Tensor checkpoint_expand(const Tensor& a, IntArrayRef b, bool c) {
  std::cout << "checkpoint expand\n";
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {vec.at(0).expand(b_, c)};
  };
  return CheckpointTensorImpl::make("expand", rt, {a})[0];
}
// IntArrayRef
// const c10::Scalar& c
Tensor checkpoint_div(const Tensor& a, const c10::Scalar& b) {
  // std::coutcheckpoint div \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::div(vec.at(0), b)};
  };
  return CheckpointTensorImpl::make("div", rt, {a})[0];
}

Tensor checkpoint_threshold_backward(
    const Tensor& a,
    const Tensor& b,
    const c10::Scalar& c) {
  std::cout << "checkpoint threshold backward\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::threshold_backward(vec.at(0), vec.at(1), c)};
  };
  return CheckpointTensorImpl::make("threshold_backward", rt, {a, b})[0];
}
Tensor& checkpoint_threshold_out(
    const Tensor& b,
    const c10::Scalar& c,
    const c10::Scalar& d,
    Tensor& a) {
  std::cout << "checkpoint threshold out";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor a_ = vec.at(1);
    at::threshold_out(a_, b, c, d);
  };
  CheckpointTensorImpl::mutate("threshold_out", mt, {a, b}, {0});
  return a;
}
Tensor& checkpoint_threshold_backward_out(
    const Tensor& grad_output,
    const Tensor& self,
    const c10::Scalar& c,
    Tensor& gradInput) {
  std::cout << "checkpoint threshold backward out\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor grad_input_ = vec.at(2);
    at::threshold_backward_out(grad_input_, vec.at(0), vec.at(1), c);
  };
  CheckpointTensorImpl::mutate(
      "threshold_out", mt, {gradInput, grad_output, self}, {0});
  return gradInput;
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_native_batch_norm_backward(
    const Tensor& grad_out,
    const Tensor& input,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& running_mean_opt,
    const c10::optional<Tensor>& running_var_opt,
    const c10::optional<Tensor>& save_mean,
    const c10::optional<Tensor>& save_invstd,
    bool train,
    double eps,
    std::array<bool, 3ul> output_mask) {
  std::cout << "checkpoint norm out backward \n";

  const bool has_weight = weight_opt.has_value();
  const bool has_running_mean = running_mean_opt.has_value();
  const bool has_running_var = running_var_opt.has_value();
  const bool has_save_mean = save_mean.has_value();
  const bool has_save_invstd = save_invstd.has_value();
  auto make_tensor = [&](const c10::optional<Tensor>& opt, const Tensor& ref) {
    return opt.has_value() ? *opt : at::empty({0}, ref.options());
  };
  std::vector<Tensor> args = {
      grad_out,
      input,
      make_tensor(weight_opt, input),
      make_tensor(running_mean_opt, input),
      make_tensor(running_var_opt, input),
      make_tensor(save_mean, input),
      make_tensor(save_invstd, input)};
  // //std::cout weight " << make_tensor(weight_opt, input).device() << "\n";
  // //std::cout running mean " << make_tensor(running_mean_opt,
  // input).device()
  //           << "\n";
  // //std::cout running var " << make_tensor(running_var_opt,
  // input).device()
  //           << "\n";
  // //std::cout save mean " << make_tensor(save_mean, input).device() <<
  // "\n"; //std::cout save invstd " << make_tensor(save_invstd,
  // input).device()
  //           << "\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    c10::optional<Tensor> weight;
    if (has_weight && vec.at(2).numel() > 0)
      weight = vec.at(2);
    c10::optional<Tensor> running_mean;
    if (has_running_mean && vec.at(3).numel() > 0)
      running_mean = vec.at(3);

    c10::optional<Tensor> running_var;
    if (has_running_var && vec.at(4).numel() > 0)
      running_var = vec.at(4);
    c10::optional<Tensor> save_mean_;
    if (has_save_mean && vec.at(5).numel() > 0)
      save_mean_ = vec.at(5);
    c10::optional<Tensor> save_invstd_;
    if (has_save_invstd && vec.at(6).numel() > 0)
      save_invstd_ = vec.at(6);
    auto ret = at::native_batch_norm_backward(
        vec.at(0),
        vec.at(1),
        weight,
        running_mean,
        running_var,
        save_mean_,
        save_invstd_,
        train,
        eps,
        output_mask);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("native_batch_norm_backward", rt, args);
  return {ret[0], ret[1], ret[2]};
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_convolution_backward(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const at::OptionalIntArrayRef bias_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3ul> output_mask) {
  std::cout << "checkpoint convolution backward \n";

  auto padding_vec = padding.vec();
  auto stride_vec = stride.vec();
  auto dilation_vec = dilation.vec();
  auto output_padding_vec = output_padding.vec();

  at::OptionalIntArrayRef bias_size_vec;
  if (bias_size.has_value()) {
    bias_size_vec = bias_size.value().vec();
  }
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    auto ret = at::convolution_backward(
        vec.at(0).contiguous(),
        vec.at(1).contiguous(),
        vec.at(2).contiguous(),
        bias_size_vec,
        stride_vec,
        padding_vec,
        dilation_vec,
        transposed,
        output_padding_vec,
        groups,
        output_mask);

    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto ret = CheckpointTensorImpl::make(
      "convolution_backward", rt, {grad_out, input, weight});
  return {ret[0], ret[1], ret[2]};
}

Tensor checkpoint_slice(
    const Tensor& a,
    int64_t b,
    c10::optional<int64_t> c,
    c10::optional<int64_t> d,
    int64_t e) {
  std::cout << "checkpoint slice\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::slice(vec.at(0), b, c, d, e)};
  };
  return CheckpointTensorImpl::make("slice", rt, {a})[0];
}

Tensor checkpoint_slice_backward(
    const Tensor& a,
    c10::ArrayRef<long> b,
    int64_t c,
    int64_t d,
    int64_t e,
    int64_t f) {
  std::cout << "checkpoint slice backward\n";
  std::vector<long> b_ = b.vec();
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::slice_backward(vec.at(0), b_, c, d, e, f)};
  };
  return CheckpointTensorImpl::make("slice_backward", rt, {a})[0];
}

std::tuple<Tensor, Tensor> checkpoint_max_pool2d_with_indices(
    const Tensor& a,
    IntArrayRef b,
    IntArrayRef c,
    IntArrayRef d,
    IntArrayRef e,
    bool f) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  std::cout << "checkpoint max pool2d\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    auto ret = at::max_pool2d_with_indices(vec.at(0), b_, c_, d_, e_, f);
    return {std::get<0>(ret), std::get<1>(ret)};
  };
  auto ret = CheckpointTensorImpl::make("max_pool2d", rt, {a});
  return {ret[0], ret[1]};
}

Tensor& checkpoint_max_pool2d_with_indices_backward_grad_input(
    const Tensor& b,
    const Tensor& c,
    IntArrayRef d,
    IntArrayRef e,
    IntArrayRef f,
    IntArrayRef g,
    bool h,
    const Tensor& i,
    Tensor& a) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec(), g_ = g.vec();
  std::cout << "max pool2d backward grad input \n";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor a_ = vec.at(3);
    at::max_pool2d_with_indices_backward_out(
        a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, vec.at(2));
  };
  CheckpointTensorImpl::mutate(
      "max_pool2d_with_indices_backward_grad_input", mt, {a, b, c, i}, {0});
  return a;
}

Tensor checkpoint_max_pool2d_with_indices_backward(
    const Tensor& a,
    const Tensor& b,
    IntArrayRef c,
    IntArrayRef d,
    IntArrayRef e,
    IntArrayRef f,
    bool g,
    const Tensor& h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  std::cout << "checkpoin max2d pool backward";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::max_pool2d_with_indices_backward(
        vec.at(0), vec.at(1), c_, d_, e_, f_, g, vec.at(2))};
  };
  return CheckpointTensorImpl::make(
      "max_pool2d_with_indices_backward", rt, {a, b, h})[0];
}
Tensor checkpoint_avg_pool2d(
    const Tensor& a,
    IntArrayRef b,
    IntArrayRef c,
    IntArrayRef d,
    bool e,
    bool f,
    c10::optional<long> g) {
  std::vector<long> b_ = b.vec(), c_ = c.vec(), d_ = d.vec();
  std::cout << "checkpoint avg pool2d \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::avg_pool2d(vec.at(0), b_, c_, d_, e, f, g)};
  };
  return CheckpointTensorImpl::make("avg_pool2d", rt, {a})[0];
}

Tensor checkpoint_avg_pool2d_backward(
    const Tensor& a,
    const Tensor& b,
    IntArrayRef c,
    IntArrayRef d,
    IntArrayRef e,
    bool f,
    bool g,
    c10::optional<long> h) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  std::cout << "checkpoint avg pool backward \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::avg_pool2d_backward(vec.at(0), vec.at(1), c_, d_, e_, f, g, h)};
  };
  return CheckpointTensorImpl::make("avg_pool2d_backward", rt, {a, b})[0];
}

Tensor& checkpoint_avg_pool2d_out(
    const Tensor& b,
    IntArrayRef c,
    IntArrayRef d,
    IntArrayRef e,
    bool f,
    bool g,
    c10::optional<long> h,
    Tensor& a) {
  std::vector<long> c_ = c.vec(), d_ = d.vec(), e_ = e.vec();
  std::cout << "checkpoint avg pool2d out\n";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor a_ = vec.at(1);
    at::avg_pool2d_out(a_, vec.at(0), c_, d_, e_, f, g, h);
  };
  CheckpointTensorImpl::mutate("avg_pool2d_out", mt, {a, b}, {0});
  return a;
}
Tensor checkpoint_embedding(
    const Tensor& a,
    const Tensor& b,
    int64_t c,
    bool d,
    bool e) {
  std::cout << "checkpoint embedding\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::embedding(vec.at(0), vec.at(1), c, d, e)};
  };
  return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
}

Tensor checkpoint_embedding_backward(
    const Tensor& a,
    const Tensor& b,
    int64_t c,
    int64_t d,
    bool e,
    bool f) {
  std::cout << "checkpoint embedding backward\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::embedding_backward(vec.at(0), vec.at(1), c, d, e, f)};
  };
  return CheckpointTensorImpl::make("embedding", rt, {a, b})[0];
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_layer_norm(
    const Tensor& input,
    IntArrayRef ns,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    double eps) {
  const bool has_weight = weight_opt.has_value();
  const bool has_bias = bias_opt.has_value();
  auto ns_vec = ns.vec();
  std::cout << "checkpoint layer norm \n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    c10::optional<Tensor> weight;
    if (has_weight && vec.size() > 1 && vec.at(1).numel() > 0)
      weight = vec.at(1);

    c10::optional<Tensor> bias;
    if (has_bias && vec.size() > 2 && vec.at(2).numel() > 0)
      bias = vec.at(2);

    auto ret = at::native_layer_norm(vec.at(0), ns_vec, weight, bias, eps);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  auto make_tensor = [&](const c10::optional<Tensor>& opt) -> Tensor {
    if (opt.has_value() && opt->numel() > 0)
      return *opt;
    else
      return at::empty({0}, input.options());
  };

  std::vector<Tensor> args;
  args.reserve(3);
  args.push_back(input);
  if (weight_opt.has_value() && weight_opt->numel() > 0)
    args.push_back(*weight_opt);
  if (bias_opt.has_value() && bias_opt->numel() > 0)
    args.push_back(*bias_opt);
  auto ret = CheckpointTensorImpl::make("native_layer_norm", rt, args);
  return {ret[0], ret[1], ret[2]};
}
Tensor checkpoint_where(
    at::Tensor const& a,
    at::Tensor const& b,
    at::Tensor const& c) {
  std::cout << "checkpoint where\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::where(vec.at(0), vec.at(1), vec.at(2))};
  };
  return CheckpointTensorImpl::make("where", rt, {a, b, c})[0];
}

Tensor checkpoint__softmax(const Tensor& self, long dim, bool half_to_float) {
  std::cout << "checkpoint softmax\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::_softmax(vec.at(0), dim, half_to_float)};
  };
  return CheckpointTensorImpl::make("_softmax", rt, {self})[0];
}

Tensor checkpoint__softmax_backward_data(
    const Tensor& grad_output,
    const Tensor& output,
    long dim,
    c10::ScalarType dtype) {
  std::cout << "checkpoint softwamx backward data\n";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    return {at::_softmax_backward_data(vec.at(0), vec.at(1), dim, dtype)};
  };
  return CheckpointTensorImpl::make(
      "_softmax_backward_data", rt, {grad_output, output})[0];
}

std::tuple<Tensor, Tensor, Tensor> checkpoint_layer_norm_backward(
    const Tensor& grad_out,
    const Tensor& input,
    IntArrayRef ns,
    const Tensor& mean,
    const Tensor& rstd,
    const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt,
    std::array<bool, 3ul> output_mask) {
  auto ns_vec = ns.vec();
  const bool has_weight = weight_opt.has_value();
  const bool has_bias = bias_opt.has_value();
  std::cout << "checkpoint layer norm backard";
  rematerialize_function_t rt = [=](const Tensors& vec) -> Tensors {
    c10::optional<Tensor> weight;
    if (has_weight && vec.size() > 4 && vec.at(4).numel() > 0)
      weight = vec.at(4);

    c10::optional<Tensor> bias;
    if (has_bias && vec.size() > 5 && vec.at(5).numel() > 0)
      bias = vec.at(5);

    auto ret = at::native_layer_norm_backward(
        vec.at(0),
        vec.at(1),
        ns_vec,
        vec.at(2),
        vec.at(3),
        weight,
        bias,
        output_mask);
    return {std::get<0>(ret), std::get<1>(ret), std::get<2>(ret)};
  };
  std::vector<Tensor> args;
  args.reserve(3);
  args.push_back(grad_out);
  args.push_back(input);
  args.push_back(mean);
  args.push_back(rstd);
  if (weight_opt.has_value() && weight_opt->numel() > 0)
    args.push_back(*weight_opt);
  if (bias_opt.has_value() && bias_opt->numel() > 0)
    args.push_back(*bias_opt);
  auto ret = CheckpointTensorImpl::make("native_layer_norm_backward", rt, args);
  return {ret[0], ret[1], ret[2]};
}

Tensor& checkpoint_avg_pool2d_backward_grad_input(
    const Tensor& b,
    const Tensor& c,
    IntArrayRef d,
    IntArrayRef e,
    IntArrayRef f,
    bool g,
    bool h,
    c10::optional<long> i,
    Tensor& a) {
  std::vector<long> d_ = d.vec(), e_ = e.vec(), f_ = f.vec();
  std::cout << "checkpoint avg pool backward grad input \n";
  mutate_function_t mt = [=](const Tensors& vec) {
    Tensor a_ = vec.at(2);
    at::avg_pool2d_backward_out(a_, vec.at(0), vec.at(1), d_, e_, f_, g, h, i);
  };
  CheckpointTensorImpl::mutate(
      "avg_pool2d_backward_grad_input", mt, {a, b, c}, {0});
  return a;
}
} // namespace native
} // namespace at
