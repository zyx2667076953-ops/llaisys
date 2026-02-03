target("llaisys-device-cpu")
    set_kind("static")
    set_languages("cxx17")
    if is_plat("windows") then
        set_warnings("all")
        add_cxflags("/wd4819", "/wd4996", "/wd4267", "/wd4244")
    else
        set_warnings("all", "error")
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/cpu/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    if is_plat("windows") then
        set_warnings("all")
        add_cxflags("/wd4819", "/wd4996", "/wd4267", "/wd4244", "/openmp")
    else
        set_warnings("all", "error")
        add_cxflags("-fPIC", "-Wno-unknown-pragmas", "-fopenmp")
        add_ldflags("-fopenmp")
    end

    add_files("../src/ops/*/cpu/*.cpp")

    on_install(function (target) end)
target_end()

