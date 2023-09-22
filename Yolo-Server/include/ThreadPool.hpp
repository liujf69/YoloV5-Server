/**
    @brief A demo of C++ thread pool
    @author Jinfu Liu
    @date 2023.09.15
    @version 1.0  
*/

#ifndef THREADPOOL_H_
#define THREADPOOL_H_

#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <future>
#include <memory>
#include <functional>
#include <stdexcept>

namespace LIU{
    class ThreadPool{
    public:
        /**
            @brief Constructor
            @param threads Number of started threads
        */
        ThreadPool(size_t threads);

        /// @brief Destructor
        ~ThreadPool();

        /**
            @brief Enqueue task into the tasks queue
            @tparam _Callable: The type of callable object
            @tparam ...Args: The parameter types of callable object
            @param _f: The callable object
            @param ...args: The parameters of callable object
            @return std::future<_Callable>
        */
        template<class _Callable, class... Args>
        auto enqueue(_Callable&& _f, Args&&... args) 
            -> std::future<typename std::result_of<_Callable(Args...)>::type>;

    private:
        std::vector<std::thread> workers; // threads array
        std::queue<std::function<void()>> tasks; // tasks queue
        std::mutex queue_mutex; // synchronized variable
        std::condition_variable condition; // condition variable
        bool stop; // stop flag
    };   

    // Constructor
    ThreadPool::ThreadPool(size_t threads): stop(false){
        for(size_t i = 0; i < threads; ++i){
            workers.emplace_back([this](){
                while (true){
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, 
                                            [this](){
                                                return this->stop || !this->tasks.empty(); // if stop or the tasks queue is not empty.
                                            });
                        if (this->stop && this->tasks.empty()){
                            return;
                        }
                        std::function<void()> task = std::move(this->tasks.front()); // move a task from tasks queue
                        this->tasks.pop();
                        lock.unlock(); // manual unlock
                        task(); // run the task
                    }
                }
            });
        }
    }

    template<class _Callable, class... Args>
    auto ThreadPool::enqueue(_Callable&& _f, Args&&... args)
        -> std::future<typename std::result_of<_Callable(Args...)>::type>{

        using return_type = typename std::result_of<_Callable(Args...)>::type; // get the return type

        std::shared_ptr<std::packaged_task<return_type()>> task = 
            std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<_Callable>(_f), std::forward<Args>(args)...)
            );

        std::future<return_type> res_future = task->get_future(); // get the future of the task
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task](){
                               (*task)(); 
                            });
        }
        condition.notify_one(); // wake up one worker thread
        return res_future;
    }

    // Destructor
    ThreadPool::~ThreadPool(){
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all(); // wake up all threads
        for(std::thread& worker : workers){
            worker.join();
        }
    }
}
#endif