AC_INIT([rSGDLM], 1.0.2) dnl package name, version

dnl Select cuda include path, from a configure option
dnl or from an environment variable.
AC_ARG_WITH([cuda-include],
            AC_HELP_STRING([--with-cuda-include=CUDA_INC_PATH],
                           [the location of cuda header files]),
            [cuda_include_path=$withval])
CUDA_INC_PATH="."
if test [ -n "$cuda_include_path" ] ; then
   CUDA_INC_PATH="${cuda_include_path}"
else
  if test [ -n "${CUDA_HOME}" ] ; then
     CUDA_INC_PATH="${CUDA_HOME}/include"
  fi
fi

dnl set cuda library path
AC_ARG_WITH([cuda-lib],
            AC_HELP_STRING([--with-cuda-lib=CUDA_LIB_PATH],
                           [the location of cuda libraries]),
            [cuda_lib_path=$withval])
CUDA_LIB_PATH="."
if test [ -n "$cuda_lib_path" ] ; then
   CUDA_LIB_PATH="${cuda_lib_path}"
else
  if test [ -n "${CUDA_HOME}" ] ; then
     CUDA_LIB_PATH="${CUDA_HOME}/lib64"
  fi
fi

dnl substitute CUDA_INC_PATH and CUDA_LIB_PATH
AC_SUBST(CUDA_INC_PATH)
AC_SUBST(CUDA_LIB_PATH)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
