cmake_minimum_required(VERSION 3.24)
project(AP_DTRL LANGUAGES CXX)

# =========================
# Global Settings
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AP_DTRL_ENABLE_TORCH "Enable LibTorch integration" ON)
option(AP_DTRL_BUILD_APP "Build application entrypoint" ON)

if(MSVC)
  add_compile_options(/W3 /EHsc)
endif()

# =========================
# Dependencies
# =========================
find_package(OpenGL REQUIRED)
find_package(Bullet REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

# =========================
# Torch (optional)
# =========================
if(AP_DTRL_ENABLE_TORCH)
  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

# =========================
# Sources
# =========================
file(GLOB AP_DTRL_UTIL_SRC   util/*.cpp)
file(GLOB AP_DTRL_ANIM_SRC   anim/*.cpp)
file(GLOB AP_DTRL_SIM_SRC    sim/*.cpp)
file(GLOB AP_DTRL_RENDER_SRC render/*.cpp)

# =========================
# Core Library
# =========================
add_library(ap_dtrl_core
  ${AP_DTRL_UTIL_SRC}
  ${AP_DTRL_ANIM_SRC}
  ${AP_DTRL_SIM_SRC}
  ${AP_DTRL_RENDER_SRC}
  learning/NeuralNet.cpp
)

target_include_directories(ap_dtrl_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${BULLET_INCLUDE_DIRS}
)

# =========================
# Link Libraries
# =========================
target_link_libraries(ap_dtrl_core PUBLIC
  ${BULLET_LIBRARIES}
  glfw
  OpenGL::GL
  jsoncpp_lib
)

# =========================
# Torch Integration (FIXED)
# =========================
if(AP_DTRL_ENABLE_TORCH AND Torch_FOUND)
  message(STATUS "LibTorch enabled")

  target_link_libraries(ap_dtrl_core PUBLIC ${TORCH_LIBRARIES})
  target_compile_definitions(ap_dtrl_core PUBLIC AP_DTRL_HAS_TORCH=1)
else()
  message(STATUS "LibTorch not found or disabled — building without it")
  target_compile_definitions(ap_dtrl_core PUBLIC AP_DTRL_HAS_TORCH=0)
endif()

# =========================
# Optional App
# =========================
if(AP_DTRL_BUILD_APP)
  add_executable(ap_dtrl_app Main.cpp)
  target_link_libraries(ap_dtrl_app PRIVATE ap_dtrl_core)
endif()