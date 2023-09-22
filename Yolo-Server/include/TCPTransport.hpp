/**
    @brief A demo of send and write img_Msg
    @author Jinfu Liu
    @date 2023.09.15
    @version 1.0  
*/

#include <iostream>
#include <arpa/inet.h>
#include <unistd.h>
#include "opencv2/opencv.hpp"

class TransportMsg{
public:
    TransportMsg(){};
    ~TransportMsg(){};

    bool Write_Msg(const int fd, cv::Mat &img){
        int row = img.rows, col = img.cols;
        img = img.reshape(0, 1);  // turn Mat into a vector
        int rgb_size = img.total() * img.elemSize();  // total bytes of the rgb image

        int net_row = htonl(row), net_col = htonl(col), net_size = htonl(rgb_size); // host endian to network endian
        unsigned char send_Msg[rgb_size + 12]; 
        memset(send_Msg, 0, sizeof(send_Msg));
        memcpy(send_Msg, &net_row, 4); // four bytes store row
        memcpy(send_Msg+4, &net_col, 4); // four bytes store col
        memcpy(send_Msg+8, &net_size, 4); // four bytes store the total bytes of image
        memcpy(send_Msg+12, img.data, rgb_size); // Remaining Bytes store img

        if(write(fd, send_Msg, rgb_size+12) == -1){ // send image to server
            std::cerr << "send() error!" << std::endl;
            exit(1);
        }
        return true;
    }

    bool Read_Msg(const int fd, cv::Mat& recv_img){
        // read six bytes
        int bytes = 0;
        unsigned char row_buf[4], col_buf[4], num_bytes_buf[4];
        if((readn(fd, &row_buf[0], 4) && readn(fd, &col_buf[0], 4) && readn(fd, &num_bytes_buf[0], 4)) == false){
            // std::cerr << "read zero byte" << std::endl;
            return false;
        }

        int i_row[4], i_col[4], i_num_bytes[4];
        memcpy(i_row, row_buf, 4);
        memcpy(i_col, col_buf, 4);
        memcpy(i_num_bytes, num_bytes_buf, 4);
        // network endian to host endian
        int row = ntohl(*i_row);
        int col = ntohl(*i_col);
        int num_bytes = ntohl(*i_num_bytes); 

        // recv image from server
        unsigned char Recv_Data[num_bytes];
        if(readn(fd, &Recv_Data[0], num_bytes) == false){
            // std::cerr << "read zero byte" << std::endl;
            return false;
        }
        recv_img = cv::Mat(row, col, CV_8UC3, Recv_Data); // uchar*->cv::Mat
        return true;
    }

    // read n bytes
    bool readn(const int fd, unsigned char* buf, int nbytes){
        int read_bytes = 0;
        for(int i = 0; i < nbytes; i += read_bytes){ 
            if((read_bytes = read(fd, buf + i, nbytes - i)) <= 0){
                break;
            }
        }
        if(read_bytes == -1){ // read error
            std::cout << "recv() error" << std::endl;
            exit(1);
        }
        else if(read_bytes == 0){
            return false;
        }
        return true;
    }
};