#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/inotify.h>
#include <csignal>
#include <chrono>
#include <thread>

// Global flag to indicate if the program should exit
bool shouldExit = false;

// Signal handler to catch termination signals
void signalHandler(int signum) {
    shouldExit = true;
}


#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))

void executeCPlusPlusProgram(const std::string& Path) {
    std::string command = Path;
    system(command.c_str());
}

int main() {
    int fd, wd1, wd2, wd3;
    char buffer[BUF_LEN];

    // Initialize inotify
    fd = inotify_init();
    if (fd < 0) {
        perror("inotify_init");
        return 1;
    }

    // Watch for file creation events in the specified directory
    wd1 = inotify_add_watch(fd, "../images", IN_CREATE | IN_MOVED_TO);
    if (wd1 < 0) {
        perror("inotify_add_watch");
        return 1;
    }
    wd2 = inotify_add_watch(fd, "../detections", IN_CREATE | IN_MOVED_TO);
    if (wd2 < 0) {
        perror("inotify_add_watch");
        return 1;
    }
    wd2 = inotify_add_watch(fd, "../landmarks", IN_CREATE | IN_MOVED_TO);
    if (wd2 < 0) {
        perror("inotify_add_watch");
        return 1;
    }

    while (!shouldExit) {
        int length = read(fd, buffer, BUF_LEN);
        if (length < 0) {
            perror("read");
            return 1;
        }

        int i = 0;
        while (i < length) {
            struct inotify_event *event = (struct inotify_event *)&buffer[i];
            if (event->len) {
                if (event->mask & IN_CREATE | IN_MOVED_TO) {
                    if (!(event->mask & IN_ISDIR)) {
                        // Check if the created file is an image
                        if (strstr(event->name, ".jpg") || strstr(event->name, ".png")) {
                            std::string Path = "./obj_det ../images/" + std::string(event->name);
                            executeCPlusPlusProgram(Path);
                        }
                        if (strstr(event->name, ".txt") || strstr(event->name, ".png")) {
                            std::string Path = "./EPnP ../landmarks/" + std::string(event->name);
                            executeCPlusPlusProgram(Path);
                        }

                    }
                }
            }
            i += EVENT_SIZE + event->len;
        }
    }

    // Clean up
    inotify_rm_watch(fd, wd1);
    inotify_rm_watch(fd, wd2);
    inotify_rm_watch(fd, wd2);
    close(fd);

    return 0;
}
