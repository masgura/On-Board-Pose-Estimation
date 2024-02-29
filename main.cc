#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/inotify.h>
#include <csignal>

// Global flag to indicate if the program should exit
bool shouldExit = false;

// Signal handler to catch termination signals
void signalHandler(int signum) {
    std::cout << "Received signal " << signum << ". Exiting..." << std::endl;
    shouldExit = true;
}


#define EVENT_SIZE  (sizeof(struct inotify_event))
#define BUF_LEN     (1024 * (EVENT_SIZE + 16))

void executeCPlusPlusProgram(const std::string& imagePath) {
    std::string command = "./obj_det " + imagePath;
    system(command.c_str());
}

int main() {
    int fd, wd;
    char buffer[BUF_LEN];

    // Initialize inotify
    fd = inotify_init();
    if (fd < 0) {
        perror("inotify_init");
        return 1;
    }

    // Watch for file creation events in the specified directory
    wd = inotify_add_watch(fd, "../images", IN_CREATE | IN_MOVED_TO);
    if (wd < 0) {
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
                        std::cout << "File created: " << event->name << std::endl;
                        // Check if the created file is an image
                        if (strstr(event->name, ".jpg") || strstr(event->name, ".png")) {
                            std::string imagePath = "../images/" + std::string(event->name);
                            executeCPlusPlusProgram(imagePath);
                        }
                    }
                }
            }
            i += EVENT_SIZE + event->len;
        }
    }

    // Clean up
    inotify_rm_watch(fd, wd);
    close(fd);

    return 0;
}
