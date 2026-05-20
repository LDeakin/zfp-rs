cmake_minimum_required(VERSION 3.16)

project(zfp_rs_ffi_c_dropin C)

include(CTest)

set(ZFP_DROPIN_BACKEND "c" CACHE STRING "Backend providing the zfp C ABI: c or rust")
set_property(CACHE ZFP_DROPIN_BACKEND PROPERTY STRINGS c rust)

if(ZFP_DROPIN_BACKEND STREQUAL "c")
  if(NOT ZFP_SOURCE_DIR)
    message(FATAL_ERROR "ZFP_SOURCE_DIR must point at the upstream zfp source tree")
  endif()

  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_UTILITIES OFF CACHE BOOL "" FORCE)
  set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(ZFP_WITH_OPENMP OFF CACHE BOOL "" FORCE)

  add_subdirectory("${ZFP_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/zfp-c" EXCLUDE_FROM_ALL)
  set(ZFP_DROPIN_TARGET zfp)
elseif(ZFP_DROPIN_BACKEND STREQUAL "rust")
  if(NOT ZFP_RS_FFI_LIBRARY)
    message(FATAL_ERROR "ZFP_RS_FFI_LIBRARY must point at libzfp_rs_ffi.a")
  endif()
  if(NOT ZFP_INCLUDE_DIR)
    message(FATAL_ERROR "ZFP_INCLUDE_DIR must point at the zfp-rs-ffi generated include directory")
  endif()

  find_package(Threads REQUIRED)

  add_library(zfp_rs_ffi_dropin STATIC IMPORTED GLOBAL)
  set_target_properties(zfp_rs_ffi_dropin PROPERTIES
    IMPORTED_LOCATION "${ZFP_RS_FFI_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ZFP_INCLUDE_DIR}"
  )
  target_link_libraries(zfp_rs_ffi_dropin INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
  if(UNIX AND NOT APPLE)
    target_link_libraries(zfp_rs_ffi_dropin INTERFACE m)
  endif()

  set(ZFP_DROPIN_TARGET zfp_rs_ffi_dropin)
else()
  message(FATAL_ERROR "ZFP_DROPIN_BACKEND must be either 'c' or 'rust'")
endif()

add_executable(zfp_dropin_compress compress.c)
target_compile_definitions(zfp_dropin_compress
  PRIVATE ZFP_DROPIN_BACKEND_NAME="${ZFP_DROPIN_BACKEND}"
)
target_link_libraries(zfp_dropin_compress PRIVATE ${ZFP_DROPIN_TARGET})

add_test(NAME zfp_dropin_compress COMMAND zfp_dropin_compress)
