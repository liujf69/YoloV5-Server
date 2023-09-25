/**
    @brief A demo of postprocess the infer result of Yolov5 
    @author Jinfu Liu
    @date 2023.09.15
    @version 1.0  
*/

#pragma once
#include "config.h"
#include <opencv2/opencv.hpp>

// 检测结果的结构体
struct alignas(float) Detection {
    float bbox[4];  // center_x center_y w h 坐标
    float conf;  // bbox_conf * cls_conf 置信度
    float class_id; // 类别
    float mask[32]; // segmentation 32 channel feature
};

// get rect
cv::Rect get_rect(cv::Mat& img, float bbox[4]){
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0); // kInputW = 640
    float r_h = kInputH / (img.rows * 1.0);
    if(r_h > r_w){
        l = bbox[0] - bbox[2] / 2.f; // left_top_x
        r = bbox[0] + bbox[2] / 2.f; // left_top_y
        t = bbox[1] - bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (kInputH - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } 
    else{
        l = bbox[0] - bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (kInputW - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    // return (left_top_x, left_top_y, width, height)
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

// draw bounding box
void draw_bbox(cv::Mat& img, std::vector<Detection>& res) {
    for (size_t j = 0; j < res.size(); j++){
        cv::Rect r = get_rect(img, res[j].bbox); // get (left_top_x, left_top_y, width, height)
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);  // draw box
        // draw label
        cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}

// iou
static float iou(float lbox[4], float rbox[4]){
    // 计算交集的左上角坐标和右下角坐标
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]); // 计算交集面积
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS); // 计算IOU
}

static bool cmp(const Detection& a, const Detection& b){
    return a.conf > b.conf; // 按预测置信度从大到小排序
}

// nms
void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh){
    /*
        conf_thresh 置信度阈值: 0.5
        nms_thresh iou阈值: 0.45
    */
    int det_size = sizeof(Detection) / sizeof(float); // 38
    std::map<float, std::vector<Detection>> m; // 存储预测类别对应的预测结果
    for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; i++) { // 遍历预测框
        // output[0] 表示检测框的数目，因此下面要 + 1
        if (output[1 + det_size * i + 4] <= conf_thresh) continue; // 过滤预测置信度低于置信度阈值的预测框
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for(auto it = m.begin(); it != m.end(); it++){ // 遍历所有预测类别
        auto& dets = it->second; // 取当前类别的所有预测结果
        std::sort(dets.begin(), dets.end(), cmp); // 按预测置信度从大到小排序预测结果
        for (size_t m = 0; m < dets.size(); ++m){ // 遍历当前类别的所有预测结果
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n){
                if (iou(item.bbox, dets[n].bbox) > nms_thresh){ // 根据 iou 过滤重叠的预测框
                    dets.erase(dets.begin() + n); // 删除重叠的预测框
                    --n;
                }
            }
        }
    }
}
