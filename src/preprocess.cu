#include "preprocess.h"
#include "cuda_utils.h"

static uint8_t* img_buffer_host = nullptr;
static uint8_t* img_buffer_device = nullptr;

// 仿射变换的矩阵 2*3
struct AffineMatrix{
    float value[6];
};

// 利用双线性插值实现仿射变换
__global__ void warpaffine_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, uint8_t const_value_st, AffineMatrix d2s, int edge){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    // d2s逆仿射变换矩阵
    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f; // 通过逆仿射变换矩阵求解对应在原图像的坐标
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    } 
    else {
        int y_low = floorf(src_y); // 向下取整获取左上角坐标
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1; // 右下角坐标

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        // w1表示右下角坐标的权重, w2表示右上角坐标
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; // 四个点的权重，用面积来表示
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        // 计算上下左右四个点对应在内存的位置
        if (y_low >= 0){
            if (x_low >= 0) v1 = src + y_low * src_line_size + x_low * 3; 
            if (x_high < src_width) v2 = src + y_low * src_line_size + x_high * 3;
        }
        if (y_high < src_height){
            if (x_low >= 0) v3 = src + y_high * src_line_size + x_low * 3;
            if (x_high < src_width) v4 = src + y_high * src_line_size + x_high * 3;
        }

        // 利用上下四个点三通道的值，以及对应的权重来实现双线性插值
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // 将bgr转换为rgb，直接交换通道即可 
    float t = c2;
    c2 = c0;
    c0 = t;
    // 归一化
    c0 = c0 / 255.0f; // r
    c1 = c1 / 255.0f; // g
    c2 = c2 / 255.0f; // b

    // rgbrgbrgb to rrrgggbbb 
    // 展开为一维来存储，因为NHWC排列时C在最内层，所以是RGB连续存储
    // 在GPU中，使用NCHW格式来计算卷积，速度会更快，此时C在外层，所以存储格式为rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx; // r
    float* pdst_c1 = pdst_c0 + area; // 在c0的基础上跳过一个通道，即跳过rrrr...g之间的area个数量的r，这样才能取到g
    float* pdst_c2 = pdst_c1 + area; // 同理在c1的基础上再跳过一个通道，这样才能取到b
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

// 通过仿射变换将图片大小处理为640×640
void cuda_preprocess(uint8_t* src, int src_width, int src_height, float* dst, int dst_width, int dst_height, cudaStream_t stream) {
    int img_size = src_width * src_height * 3;
    // 复制输入数据到申请的锁页内存中
    memcpy(img_buffer_host, src, img_size);
    // host->device
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

    AffineMatrix s2d, d2s; // s2d 表示将src变换为dst图像的仿射矩阵，d2s则表示将dst变换为src的逆变换矩阵
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width); // 缩放比例: dst/src，取较小值
    // 对仿射变换矩阵赋值
    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width  * 0.5  + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value); // 仿射变换矩阵
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value); // 仿射逆变换矩阵
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s); // 通过s2d计算逆变换矩阵d2s

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256; // 线程块中使用256个线程
    int blocks = ceil(jobs / (float)threads);

    warpaffine_kernel<<<blocks, threads, 0, stream>>>(
        img_buffer_device, src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2s, jobs);
}

// 准备内存
void cuda_preprocess_init(int max_image_size) {
  // 使用cudaMallocHost为输入数据申请页锁定内存（pinned memory）（始终存放在物理内存，不会发生页交换）
  CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
  // 使用cudaMalloc为输入数据申请pageable memory
  CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}
