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
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIAlignParameter roi_align_param = this->layer_param_.roi_align_param();
  CHECK_GT(roi_align_param.aligned_h(), 0)
      << "aligned_h must be > 0";
  CHECK_GT(roi_align_param.aligned_w(), 0)
      << "aligned_w must be > 0";
  aligned_height_ = roi_align_param.aligned_h();
  aligned_width_ = roi_align_param.aligned_w();
  spatial_scale_ = roi_align_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, aligned_height_,
      aligned_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, aligned_height_,
      aligned_width_);
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_rois = bottom[1]->cpu_data();
  // Number of ROIs
  int num_rois = bottom[1]->num();
  int batch_size = bottom[0]->num();
  int top_count = top[0]->count();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int* argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);

  // For each ROI R = [batch_index x1 y1 x2 y2]: max align over R
  for (int n = 0; n < num_rois; ++n) {
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = bottom_rois[1] * spatial_scale_;
    Dtype roi_start_h = bottom_rois[2] * spatial_scale_;
    Dtype roi_end_w = bottom_rois[3] * spatial_scale_;
    Dtype roi_end_h = bottom_rois[4] * spatial_scale_;
    CHECK_GE(roi_batch_ind, 0);
    CHECK_LT(roi_batch_ind, batch_size);

    int roi_height = max(roi_end_h - roi_start_h, static_cast<Dtype>(1.0f));
    int roi_width = max(roi_end_w - roi_start_w, static_cast<Dtype>(1.0f));
    const Dtype bin_size_h = static_cast<Dtype>(roi_height)
                             / static_cast<Dtype>(aligned_height_);
    const Dtype bin_size_w = static_cast<Dtype>(roi_width)
                             / static_cast<Dtype>(aligned_width_);

    const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);

    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < aligned_height_; ++ph) {
        for (int pw = 0; pw < aligned_width_; ++pw) {
          // User bilinear interpolation to get the roi data.
          const Dtype x = (static_cast<Dtype>(pw) + 0.5) * bin_size_w - 0.5;
          const Dtype y = (static_cast<Dtype>(ph) + 0.5) * bin_size_h - 0.5;

          const int x0 = static_cast<int>(x);
          const int y0 = static_cast<int>(y);
          const int x1 = min(x0 + 1, width_);
          const int y1 = min(y0 + 1, height_);
          Dtype u = x - static_cast<Dtype>(x0);
          Dtype v = y - static_cast<Dtype>(y0);
         
          Dtype p00 = batch_data[y0 * width_ + x0];
          Dtype p01 = batch_data[y1 * width_ + x0];
          Dtype p10 = batch_data[y0 * width_ + x1];
          Dtype p11 = batch_data[y1 * width_ + x1];
          
          const int align_index = ph * aligned_width_ + pw;
          top_data[align_index] = (1-v)*((1-u)*p00 + u*p10) + v*((1-u)*p01 + u*p11);
        }
      }
      // Increment all data pointers by one channel
      batch_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
    // Increment ROI data pointer
    bottom_rois += bottom[1]->offset(1);
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignLayer);
#endif

INSTANTIATE_CLASS(ROIAlignLayer);
REGISTER_LAYER_CLASS(ROIAlign);

}  // namespace caffe
