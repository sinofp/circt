##===- CMakeLists.txt - Define a support library --------------*- cmake -*-===//
##
##===----------------------------------------------------------------------===//

#-------------------------------------------------------------------------------
# VCSRevision.h and Version.cpp, from llvm/include/llvm/Support/CMakeLists.txt
#-------------------------------------------------------------------------------
configure_file(Version.cpp.in Version.cpp)

find_first_existing_vc_file("${circt_SOURCE_DIR}" CIRCT_GIT_LOGS_HEAD)

set(VCSRevision_h "${circt_BINARY_DIR}/include/circt/Support/VCSRevision.h")

set(generate_vcs_version_script "${LLVM_CMAKE_DIR}/GenerateVersionFromVCS.cmake")

add_custom_command(OUTPUT "${VCSRevision_h}"
  DEPENDS "${CIRCT_GIT_LOGS_HEAD}" "${generate_vcs_version_script}"
  COMMAND ${CMAKE_COMMAND} "-DNAMES=CIRCT"
  "-DCIRCT_SOURCE_DIR=${circt_SOURCE_DIR}"
  "-DHEADER_FILE=${VCSRevision_h}"
  -P "${generate_vcs_version_script}")

set_source_files_properties("${VCSRevision_h}"
  PROPERTIES GENERATED TRUE
  HEADER_FILE_ONLY TRUE)

#-------------------------------------------------------------------------------

add_circt_library(CIRCTSupport
  BackedgeBuilder.cpp
  FieldRef.cpp
  LoweringOptions.cpp
  Path.cpp
  APInt.cpp
  "${VCSRevision_h}"
  "${CMAKE_CURRENT_BINARY_DIR}/Version.cpp"

  ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  MLIRIR
  )
