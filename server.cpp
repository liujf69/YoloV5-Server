/**
    @brief A demo of server, you can run by ./server 9190
    @author Jinfu Liu
    @date 2023.09.17
    @version 1.0  
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <string.h>
#include <memory>

#include "opencv2/opencv.hpp"
#include "ThreadPool.hpp"
#include "infer.hpp"
#include "TCPTransport.hpp"

#define EPOLL_SIZE 10
#define BUF_SIZE 100

// RGBMessage
class RGBMessage{
public:
    cv::Mat Img;
    RGBMessage(cv::Mat& img){
        this->Img = img;
    }
};

class Server{
private:
    // Socket
    int serv_sock; 
    int Port;
    struct sockaddr_in serv_adr;

    // Epoll
    int epConnect, epRead1, epRead2;
    struct epoll_event *connect_events, *read_ep_events1, *read_ep_events2; 
    struct epoll_event event;
    char buf[BUF_SIZE];
    
    // ThreadPool
    LIU::ThreadPool TPool{5};  
    std::atomic<int> connect1{0}; // the num of current connected_client in read thread1
    std::atomic<int> connect2{0}; // the num of current connected_client in read thread2

    // Data
    TransportMsg Msg;
    
public:
    Server(){}
    ~Server(){close(serv_sock), close(epConnect), close(epRead1), close(epRead2);}

    bool init_Server(int port){
        this->Port = port;
        serv_sock = socket(PF_INET, SOCK_STREAM, 0);
        memset(&serv_adr, 0, sizeof(serv_adr));
        serv_adr.sin_family = AF_INET;
        serv_adr.sin_addr.s_addr = htonl(INADDR_ANY);
        serv_adr.sin_port = htons(Port);

        // avoid timewait
        socklen_t option = 1;
        int optlen = sizeof(option);
        setsockopt(serv_sock, SOL_SOCKET, SO_REUSEADDR, (void*)&option, optlen);

        if(bind(serv_sock, (struct sockaddr*)&serv_adr, sizeof(serv_adr)) == -1){
            std::cout << "bind() error" << std::endl;
            exit(1);
        }
        if(listen(serv_sock, 5) == -1){
            std::cout << "listen() error" << std::endl;
            exit(1);
        }
        return true;
    }

    void run_Server(){
        // Three Epoll
        epConnect = epoll_create(1); 
        epRead1 = epoll_create(EPOLL_SIZE);
        epRead2 = epoll_create(EPOLL_SIZE);
        connect_events = (struct epoll_event *)malloc(sizeof(struct epoll_event));
        read_ep_events1 = (struct epoll_event *)malloc(sizeof(struct epoll_event) * EPOLL_SIZE);
        read_ep_events2 = (struct epoll_event *)malloc(sizeof(struct epoll_event) * EPOLL_SIZE);

        event.events = EPOLLIN; // read
        event.data.fd = serv_sock; // fd
        epoll_ctl(epConnect, EPOLL_CTL_ADD, serv_sock, &event); // add serv_sock to epoll
        
        // create two read threads
        std::thread readthread1(&Server::Read_data, this, epRead1, read_ep_events1, 1); // use 1 and 2 to distinguish two threads
        std::thread readthread2(&Server::Read_data, this, epRead2, read_ep_events2, 2);

        while(1){
            int con_event_num = epoll_wait(epConnect, connect_events, EPOLL_SIZE, -1); // wait connect event
            if(con_event_num == -1){
                std::cout << "epoll_wait() error";
                break;
            }
            for(int i = 0; i < con_event_num; i++){ // check event
                if(connect_events[i].data.fd == serv_sock){ // servsock means connection event
                    struct sockaddr_in clnt_adr;
                    socklen_t adr_sz = sizeof(clnt_adr);
                    int clnt_sock = accept(serv_sock, (struct sockaddr*)&clnt_adr, &adr_sz);

                    // add clnt_sock to epRead
                    event.events = EPOLLIN;
                    event.data.fd = clnt_sock;
                    
                    // Least Connections for Load Balance 
                    if(connect1 <= connect2){
                        connect1++; // atomic
                        epoll_ctl(epRead1, EPOLL_CTL_ADD, clnt_sock, &event);
                    }
                    else{
                        connect2++; // atomic
                        epoll_ctl(epRead2, EPOLL_CTL_ADD, clnt_sock, &event);
                    }
                    printf("connected client: %d \n", clnt_sock);
                }
            }
        }
    }

    // read rgb data
    void Read_data(int epRead, struct epoll_event* read_ep_events, int id){
        while(1){
            int num_events = epoll_wait(epRead, read_ep_events, EPOLL_SIZE, -1);
            if(num_events == -1){
                std::cout << "epoll_wait() error";
                break;
            }
            for(int i = 0; i < num_events; i++){ // check event
                int fd = read_ep_events[i].data.fd;
                cv::Mat recv_img;
                bool flag = Msg.Read_Msg(fd, recv_img); // recv EOF
                if(flag == false){
                    std::cout << "closed client: " << read_ep_events[i].data.fd << std::endl;
                    epoll_ctl(epRead, EPOLL_CTL_DEL, read_ep_events[i].data.fd, NULL); // delete connection
                    close(read_ep_events[i].data.fd);
                    if(id == 1) connect1--; // atomic
                    else connect2--;
                }
                else{ // process data and send to client
                    std::shared_ptr<RGBMessage> RGBMsg_ptr = std::make_shared<RGBMessage>(recv_img);
                    TPool.enqueue(&Server::Process_Write, this, RGBMsg_ptr, (int)read_ep_events[i].data.fd);
                }
            }
        }
    }

    // process rgb data and send to client
    void Process_Write(std::shared_ptr<RGBMessage> RGBMsg, int fd){
        cv::Mat read_img = RGBMsg->Img; 
        YoloV5_Infer Yolo_Infer; // Yolo
        cv::Mat send_img = Yolo_Infer.run(read_img);
        Msg.Write_Msg(fd, send_img);  // send image to server
    }
};

int main(int argc, char* argv[]){
    if(argc != 2){
        printf("Usage : %s <port>\n", argv[0]);
        exit(1);
    }
    Server myServer;
    myServer.init_Server(atoi(argv[1]));
    myServer.run_Server();
    return 0;
}