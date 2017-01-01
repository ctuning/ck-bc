#! /bin/bash

# Build program
pushd ..

make clean
if [ $? -ne 0 ]; then
   echo "Error: make failed!" 
    exit 1
  fi
  env;
  make CC=${CK_CC}  CUDA_HOME=${CK_ENV_COMPILER_CUDA} EXEC=${CK_PROG_TARGET_EXE}
  if [ $? -ne 0 ]; then
     echo "Error: make failed!" 
      exit 1
    fi

    popd

    mv -f ../${CK_PROG_TARGET_EXE} .
    if [ $? -ne 0 ]; then
       echo "Error: make failed!" 
        exit 1
      fi
