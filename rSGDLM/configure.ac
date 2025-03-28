AC_INIT([rSGDLM],[1.1.0]) dnl package name, version

AC_MSG_NOTICE([start of configure.ac for rSGDLM])

dnl Read user-supplied nvcc flags
AC_ARG_WITH([nvcc-flags],
            AS_HELP_STRING([--with-nvcc-flags=CUSTOM FLAGS],[custom flags supplied to nvcc]),
            [nvcc_custom_flags=$withval])
NVCC_CUSTOM_FLAGS=""
if test [ -n "${nvcc_custom_flags}" ] ; then
   AC_MSG_NOTICE([received nvcc_custom_flags ${nvcc_custom_flags}])
   NVCC_CUSTOM_FLAGS="${nvcc_custom_flags}"
fi

dnl Read user-supplied gencode flag
AC_ARG_WITH([nvcc-gencode],
            AS_HELP_STRING([--with-nvcc-gencode=GENCODE FLAG],[specify GPU arch and code version, for example arch=compute_35,code=sm_35]),
            [nvcc_gencode_flag=$withval])
NVCC_GENCODE_FLAGS=""
if test [ -n "${nvcc_gencode_flag}" ] ; then
   AC_MSG_NOTICE([received nvcc_gencode_flag ${nvcc_gencode_flag}])
   NVCC_GENCODE_FLAGS="-gencode ${nvcc_gencode_flag}"
fi

dnl Select cuda include path, from a configure option
dnl or from an environment variable.
AC_ARG_WITH([cuda-include],
            AS_HELP_STRING([--with-cuda-include=CUDA_INC_PATH],[the location of cuda header files]),
            [cuda_include_path=$withval])
CUDA_INC_PATH="."
if test [ -n "${cuda_include_path}" ] ; then
   AC_MSG_NOTICE([received cuda_include_path ${cuda_include_path}])
   CUDA_INC_PATH="${cuda_include_path}"
else
  if test [ -n "${CUDA_HOME}" ] ; then
     CUDA_INC_PATH="${CUDA_HOME}/include"
  fi
fi

dnl set cuda library path
AC_ARG_WITH([cuda-lib],
            AS_HELP_STRING([--with-cuda-lib=CUDA_LIB_PATH],[the location of cuda libraries]),
            [cuda_lib_path=$withval])
CUDA_LIB_PATH="."
if test [ -n "${cuda_lib_path}" ] ; then
   AC_MSG_NOTICE([received cuda_lib_path ${cuda_lib_path}])
   CUDA_LIB_PATH="${cuda_lib_path}"
else
  if test [ -n "${CUDA_HOME}" ] ; then
     CUDA_LIB_PATH="${CUDA_HOME}/lib64"
  fi
fi

dnl substitute CUDA_INC_PATH and CUDA_LIB_PATH
AC_SUBST(NVCC_GENCODE_FLAGS)
AC_SUBST(NVCC_CUSTOM_FLAGS)
AC_SUBST(CUDA_INC_PATH)
AC_SUBST(CUDA_LIB_PATH)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
