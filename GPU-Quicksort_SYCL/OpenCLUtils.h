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
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &paramValueSizeRet);

    char* buildLogMsgBuf = (char *)malloc(sizeof(char) * paramValueSizeRet + 1);
	if( buildLogMsgBuf )
	{
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, paramValueSizeRet, buildLogMsgBuf, &paramValueSizeRet);
		buildLogMsgBuf[paramValueSizeRet] = '\0';	//mark end of message string

		printf("\nOpenCL C Program Build Log:\n");
		puts(buildLogMsgBuf);
		fflush(stdout);

		free(buildLogMsgBuf);
	}
}

static bool isNvidiaGpu = false;

void CreateOCLProgramFromSourceFile(char const *pSrcFilePath, cl_context hClContext, cl_program *pCLProgram )
{
	    FILE* fp = fopen(pSrcFilePath, "rb");
        if (!fp) 
		{
			printf("Failed to find OpenCL source program: %s\n", pSrcFilePath);
			//Cleanup (-1, true, "Failed to open CL Source file.\n");
			exit(0);
		}

        fseek(fp, 0, SEEK_END);
        long size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        char* buf = (char*)malloc((size + 1)*sizeof(char));
		if (buf == 0)
		{
			printf("Failed to allocated buffer of sufficient size fo the file: %s\n", pSrcFilePath);
			exit(-1);
		}
        buf[size] = '\0';
        size_t records_read = fread(buf, size, 1, fp);
		if (records_read != 1) {
			printf("Failed to read the file: %s\n", pSrcFilePath);
			free(buf);
			exit(-1);
		}
        int err = fclose(fp);
		if (err != 0) {
			printf("Failed to close the file: %s\n", pSrcFilePath);
			free(buf);
			exit(-1);
		}

		size_t szKernelLength = size;
		cl_int ciErrNum;
		*pCLProgram = clCreateProgramWithSource(hClContext, 1, (const char **) &buf, &szKernelLength, &ciErrNum);
        CheckCLError (ciErrNum, "Failed to create program.", "Created program.");

        free(buf);
}

void CompileOpenCLProgram(bool bCPUDevice, cl_device_id oclDeviceID, cl_context oclContextHdl, const char* pSourceFileStr, cl_program* pOclProgramHdl)
{
	cl_int		ciErrNum;
	cl_program	oclProgramHdl;

	*pOclProgramHdl = NULL;

	CreateOCLProgramFromSourceFile(pSourceFileStr, oclContextHdl, &oclProgramHdl);

	if (bCPUDevice) {
		ciErrNum = clBuildProgram(oclProgramHdl, 0, NULL, "-cl-std=CL1.2 -cl-mad-enable -DCPU_DEVICE=1", NULL, NULL);
	} else {
    if (isNvidiaGpu) {
		  ciErrNum = clBuildProgram(oclProgramHdl, 0, NULL, "-cl-std=CL1.2 -cl-mad-enable -DNVIDIA_GPU=1", NULL, NULL);
    } else {
		  ciErrNum = clBuildProgram(oclProgramHdl, 0, NULL, "-cl-std=CL1.2 -cl-mad-enable", NULL, NULL);
    }
	}
	if (ciErrNum != CL_SUCCESS)
	{
		printf("ERROR: Failed to build program... ciErrNum = %d\n", ciErrNum);
		BuildFailLog(oclProgramHdl, oclDeviceID);
	}
	CheckCLError (ciErrNum, "Program building failed.", "Built Program");
	if (ciErrNum != CL_SUCCESS)
	{
		printf("Enter any key to exit.\n");
		getchar();
		exit(0);
	}

	// Output parameters:
	*pOclProgramHdl = oclProgramHdl;
}

#endif
