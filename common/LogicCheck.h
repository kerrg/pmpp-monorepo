#pragma once

#include <cstdlib>
#include <iostream>

#define LOGIC_CHECK(cond)                                                      \
  if (!(cond)) {                                                               \
    std::cerr << "Logic check failed at " << __FILE__ << ":" << __LINE__       \
              << std::endl;                                                    \
    exit(EXIT_FAILURE);                                                        \
  }
