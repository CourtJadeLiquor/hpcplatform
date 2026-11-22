# Thin shell (docs/architecture.md §5): core and backend suites are mounted
# here; operator suites live in their own folders under src/ops/.
add_subdirectory(core)
add_subdirectory(backends)
add_subdirectory(perf)

# The config-driven operator suite requires yaml-cpp; built only with the
# feature on. The default-build stub behavior is covered in tests/core.
if(MYCV_WITH_YAML)
  add_subdirectory(operator)
endif()
