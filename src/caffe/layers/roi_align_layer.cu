// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/fast_rcnn_layers.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int aligned_height, const int aligned_width,
    const Dtype* bottom_rois, Dtype* top_data, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int pw = index % aligned_width;
    int ph = (index / aligned_width) % aligned_height;
    int c = (index / aligned_width / aligned_height) % channels;
    int n = index / aligned_width / aligned_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w, 1.0);
    Dtype roi_height = max(roi_end_h - roi_start_h, 1.0);
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(aligned_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(aligned_width);

    // Add roi offsets and clip to input boundaries
    const Dtype x = (static_cast<Dtype>(pw) + 0.5) * bin_size_w - 0.5;
    const Dtype y = (static_cast<Dtype>(ph) + 0.5) * bin_size_h - 0.5;

    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = min(x0 + 1, width);
    const int y1 = min(y0 + 1, height);
    Dtype u = x - static_cast<Dtype>(x0);
    Dtype v = y - static_cast<Dtype>(y0);
    
    // Define an empty align region to be zero
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    Dtype p00 = bottom_data[y0 * width + x0];
    Dtype p01 = bottom_data[y1 * width + x0];
    Dtype p10 = bottom_data[y0 * width + x1];
    Dtype p11 = bottom_data[y1 * width + x1];

    top_data[index] = (1-v)*((1-u)*p00 + u*p10) + v*((1-u)*p01 + u*p11);
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, spatial_scale_, channels_, height_, width_,
      aligned_height_, aligned_width_, bottom_rois, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int num_rois, const Dtype spatial_scale,
    const int channels, const int height, const int width,
    const int aligned_height, const int aligned_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that aligned this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
      Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
      Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
      Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;

      // Skip if ROI doesn't include (h, w), consider 1 px outside.
      const bool in_roi = (w >= (roi_start_w + 1) && w <= (roi_end_w - 1) &&
                           h >= (roi_start_h + 1) && h <= (roi_end_h - 1));
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * aligned_height * aligned_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const int* offset_argmax_data = argmax_data + offset;

      // Compute feasible set of aligned units that could have aligned
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w, 1.0);
      int roi_height = max(roi_end_h - roi_start_h, 1.0);

      Dtype bin_size_h = static_cast<Dtype>(roi_height)
                         / static_cast<Dtype>(aligned_height);
      Dtype bin_size_w = static_cast<Dtype>(roi_width)
                         / static_cast<Dtype>(aligned_width);

      const Dtype x = (static_cast<Dtype>(aligned_width) + 0.5) * bin_size_w - 0.5;
      const Dtype y = (static_cast<Dtype>(aligned_width) + 0.5) * bin_size_h - 0.5;

      const int x0 = static_cast<int>(x);
      const int y0 = static_cast<int>(y);
      const int x1 = min(x0 + 1, width);
      const int y1 = min(y0 + 1, height);
      Dtype u = x - static_cast<Dtype>(x0);
      Dtype v = y - static_cast<Dtype>(y0);

      if(x0 == w && y0 == h){
          gradient += (1-v) * (1-u) * offset_top_diff[h * width + w];
      }
      else if(x1 == w && y0 == h){
          gradient += (1-v) * u * offset_top_diff[h * width + w];
      }
      else if(x0 == w && y1 == h){
          gradient += (1-u) * v * offset_top_diff[h * width + w];
      }
      else if(x1 == w && y1 == h){
          gradient += v * u * offset_top_diff[h * width + w];
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), spatial_scale_, channels_,
      height_, width_, aligned_height_, aligned_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe
