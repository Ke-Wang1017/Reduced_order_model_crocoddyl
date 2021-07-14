SET(${PROJECT_NAME}_PYTHON_FILES
  __init__.py
  vhip_model.py
  vertex_control.py
  asymmetric_cone.py
  biped_vertex.py
  util.py
  )

FOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})
  PYTHON_BUILD(vhip ${python})
  INSTALL(FILES vhip/${python} DESTINATION ${PYTHON_SITELIB}/${PROJECT_NAME})
ENDFOREACH(python ${${PROJECT_NAME}_PYTHON_FILES})