/* ************************************************************************* *\
               INTEL CORPORATION PROPRIETARY INFORMATION
     This software is supplied under the terms of a license agreement or 
     nondisclosure agreement with Intel Corporation and may not be copied 
     or disclosed except in accordance with the terms of that agreement. 
        Copyright (C) 2014-2019 Intel Corporation. All Rights Reserved.
\* ************************************************************************* */

#ifndef OPENCLUTILS_DOT_H
#define OPENCLUTILS_DOT_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <CL/cl.h>

// Util for error checking:
//#undef __OCL_NO_ERROR_CHECKING
#define __OCL_NO_ERROR_CHECKING

#ifdef __OCL_NO_ERROR_CHECKING
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)	\
	assert (CL_SUCCESS == __errNum__);
#else
#define CheckCLError(__errNum__, __failMsg__, __passMsg__)	\
if (CL_SUCCESS != __errNum__)								\
{															\
		char __msgBuf__[256];								\
		sprintf (__msgBuf__, "CL Error num %d: %s at line %d, file %s in function %s().\n", __errNum__, __failMsg__, __LINE__, __FILE__, __FUNCTION__);	\
		printf (__msgBuf__);								\
		getchar();											\
		printf("Failed on OpenCLError\n");					\
		assert (CL_SUCCESS != __errNum__);					\
		exit(0);											\
} else if (__passMsg__)										\
{															\
	printf("CL Success: %s\n", __passMsg__);				\
}				
#endif

// Util for OpenCL build log:
void BuildFailLog(cl_program program,
                  cl_device_id device_id )
{
    size_t paramValueSizeRet = 0;
    cl_int status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &paramValueSizeRet);
    printf("clGetProgramBuildInfo returned %d\n", status);
	if (paramValueSizeRet == 0) {
		printf("\nOpenCL C Program Build Log is empty!\n");
		return;
	} else {
		printf("\nOpenCL C Build log is %zu characters long.\n", paramValueSizeRet);
	}
    char* buildLogMsgBuf = (char *)malloc(sizeof(char) * paramValueSizeRet + 1);
	if( buildLogMsgBuf )
	{
		status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, paramValueSizeRet, buildLogMsgBuf, &paramValueSizeRet);
		printf("clGetProgramBuildInfo returned %d\n", status);
		buildLogMsgBuf[paramValueSizeRet] = '\0';	//mark end of message string

		printf("\nOpenCL C Program Build Log:\n");
		puts(buildLogMsgBuf);
		fflush(stdout);

		free(buildLogMsgBuf);
	}
}

static bool isNvidiaGpu = false;

#endif
