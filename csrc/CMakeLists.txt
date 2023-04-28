add_library(colo_asyncio
        OBJECT
        aio.cpp
        uring.cpp
        gds.cu)
target_link_libraries(colo_asyncio
        PUBLIC uring
        PUBLIC aio
        PUBLIC cufile
        PUBLIC cuda)
target_include_directories(colo_asyncio PUBLIC ../include)

set_target_properties(colo_asyncio PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(space_mgr
        OBJECT
        space_mgr.cpp)
target_include_directories(space_mgr PUBLIC ../include)
