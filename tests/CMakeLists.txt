# Thin shell (docs/architecture.md §5): core and backend suites are mounted
# here; operator suites live in their own folders under src/ops/.
add_subdirectory(core)
add_subdirectory(backends)
add_subdirectory(perf)
