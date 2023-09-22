/**
    @brief A demo of postprocess the infer result of Yolov5 
    @author Jinfu Liu
    @date 2023.09.15
    @version 1.0  
*/

#pragma once
#include "config.h"
#include <opencv2/opencv.hpp>

struct alignas(float) Detection {
    float bbox[4];  // center_x center_y w h
    float conf;  // bbox_conf * cls_conf
    float class_id;
    float mask[32];
};

// get rect
cv::Rect get_rect(cv::Mat& img, float bbox[4]){
    float l, r, t, b;
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    if(r_h > r_w){
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
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
    float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

static bool cmp(const Detection& a, const Detection& b){
    return a.conf > b.conf;
}

// nms
void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh){
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;
    for (int i = 0; i < output[0] && i < kMaxNumOutputBbox; i++) {
        if (output[1 + det_size * i + 4] <= conf_thresh) continue;
        Detection det;
        memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
        m[det.class_id].push_back(det);
    }
    for(auto it = m.begin(); it != m.end(); it++){
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m){
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n){
                if (iou(item.bbox, dets[n].bbox) > nms_thresh){
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}


