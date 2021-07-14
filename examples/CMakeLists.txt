SET(${PROJECT_NAME}_EXAMPLES_PYTHON
  talos_walk
  )

INSTALL(FILES __init__.py DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples)

FOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})
  PYTHON_BUILD(. "${examples}.py")
  INSTALL(FILES ${examples}.py DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME}/examples)
  ADD_CUSTOM_TARGET("examples-${examples}"
    ${CMAKE_COMMAND} -E env PYTHONPATH=${PROJECT_BINARY_DIR}/python:${PROJECT_BINARY_DIR}/examples:$ENV{PYTHONPATH}
    ${PYTHON_EXECUTABLE} -c "import ${examples}" \${INPUT})

  # examples are too slow in Debug mode to be used as tests
  ADD_PYTHON_UNIT_TEST("example-python-${examples}" "examples/${examples}.py" "python/vhip")
ENDFOREACH(examples ${${PROJECT_NAME}_EXAMPLES_PYTHON})