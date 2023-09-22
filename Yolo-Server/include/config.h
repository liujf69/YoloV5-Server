#pragma once

// you can set them to whatever you want.
const static char* kInputTensorName = "data"; // input tensor name
const static char* kOutputTensorName = "prob"; // output tensor name

// number of classes
constexpr static int kNumClass = 80; // 0 represents person

constexpr static int kBatchSize = 1;

// Yolo's input width and height must by divisible by 32
constexpr static int kInputH = 640;
constexpr static int kInputW = 640;

// Maximum number of output bounding boxes from yololayer plugin.
// That is maximum number of output bounding boxes before NMS.
constexpr static int kMaxNumOutputBbox = 1000;

constexpr static int kNumAnchor = 3;

// The bboxes whose confidence is lower than kIgnoreThresh will be ignored in yololayer plugin.
constexpr static float kIgnoreThresh = 0.1f;

// NMS overlapping thresh and final detection confidence thresh
const static float kNmsThresh = 0.45f;
const static float kConfThresh = 0.5f;

// If your image size is larger than 4096 * 3112, please increase this value
const static int kMaxInputImageSize = 4096 * 3112;

