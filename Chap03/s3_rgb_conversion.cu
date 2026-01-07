#include <iostream>
#include <fstream>
#include <stdexcept>

int main() {
    try {
        std::ifstream imageFile("sample.jpg", std::ios::binary);

        if (!imageFile.is_open()) {
            throw std::runtime_error("Failed to open sample.jpg");
        }

        std::cout << "Success: sample.jpg opened successfully." << std::endl;
        imageFile.close();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
