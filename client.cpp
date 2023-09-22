/**
    @brief A demo of client, you can run by ./client 127.0.0.1 9190
    @author Jinfu Liu
    @date 2023.09.17
    @version 1.0  
*/

#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <arpa/inet.h>
#include <unistd.h> 
#include <vector>
#include <string>
 
#include "opencv2/opencv.hpp"
#include "TCPTransport.hpp"

#define BUF_SIZE 1024
typedef unsigned char uchar;
 
class Client{
private:
    // Addr
    std::string IP;
    int Port;
    // Socket
    int serv_sock;
    struct sockaddr_in serv_adr;
    // Data
    cv::Mat send_rgb;
    cv::Mat recv_rgb;
    TransportMsg Msg;

public:
    // Constructor
    Client(std::string ip, int port) : IP(ip), Port(port) {};
    // Destructor 
    ~Client(){close(serv_sock);}
    // init client, bind server ip and port
    void init_Client();
    // run client, send and recv data
    void run_Client();
    // get cur time, return the string type
    std::string get_time_string();
};

void Client::init_Client(){
    serv_sock = socket(AF_INET, SOCK_STREAM, 0); // create tcp socket
    memset(&serv_adr, 0, sizeof(serv_adr)); // set serv_adr
    serv_adr.sin_family = AF_INET;
    serv_adr.sin_addr.s_addr = inet_addr(IP.c_str());
    serv_adr.sin_port = htons(Port);
}

void Client::run_Client(){
    // connect to server
    if(connect(serv_sock, (struct sockaddr*)&serv_adr, sizeof(serv_adr)) == -1){
        std::cerr << "connect() error" << std::endl; // connect failed
        exit(1);
    }

    while(1){
        std::cout << "Input the image path (Q to quit): " << std::endl;
        std::string str;
        std::cin >> str;
        if(str == "q" || str == "Q"){ // close client
            close(serv_sock);
            break;
        } 

        cv::Mat send_img = cv::imread(str);
        bool send_flag = Msg.Write_Msg(serv_sock, send_img);  // send image to server
        cv::Mat recv_img;
        Msg.Read_Msg(serv_sock, recv_img); // recv image from server

        // save infer image
        std::string save_path = "../images/in_";
        cv::imwrite(save_path + get_time_string() + ".jpg", recv_img);
    }
}

std::string Client::get_time_string(){
    time_t setTime;
    time(&setTime);
    tm* ptm = localtime(&setTime);
    std::string cur_time = std::to_string(ptm->tm_year + 1900) + "_"
                        + std::to_string(ptm->tm_mon + 1) + "_"
                        + std::to_string(ptm->tm_mday) + "_"
                        + std::to_string(ptm->tm_hour) + "_"
                        + std::to_string(ptm->tm_min) + "_"
                        + std::to_string(ptm->tm_sec);
    return cur_time;
}

int main(int argc, char* argv[]){
    if(argc != 3){
        printf("Usage : %s <IP> <port>\n", argv[0]);
        exit(1);
    }
    Client client1(argv[1], atoi(argv[2]));
    client1.init_Client();
    client1.run_Client();
	return 0;
}