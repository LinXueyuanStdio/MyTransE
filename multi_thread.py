import _thread
import time

action_list = []


# 为线程定义一个函数
def print_time(threadName, delay):
    global action_list
    while 1:
        time.sleep(delay)
        action = "%s: %s" % (threadName, time.ctime(time.time()))
        action_list.append(action)
        print(action, action_list)


# 创建两个线程
try:
    _thread.start_new_thread(print_time, ("Thread-1", 2,))
    _thread.start_new_thread(print_time, ("Thread-2", 4,))
except:
    print("Error: 无法启动线程")

while 1:
    pass
