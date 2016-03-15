/***************************************************************************
 *
 * Copyright (c) wyr. All Rights Reserved
 *
 **************************************************************************/



/**
 * @file main.cpp
 * @author wangyanran100@gmail.com
 * @brief
 *
 **/

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define FLT_MAX 3.4028234663852886E+38F
#define dataType float
#define FEALEN 4000
#define FILENUM 47214
float* fea_Data;
float* feaLine;
float* kernel;
using namespace std;

vector<dataType> splitex(string sr) {
    vector<dataType> datavec;
    string strtemp;
    
    string::size_type pos1, pos2;
    pos2 = sr.find('\t');
    pos1 = 0;
	while (string::npos != pos2) {
        datavec.push_back((float)atof(sr.substr(pos1, pos2 - pos1).c_str()));
        pos1 = pos2 + 1;
        pos2 = sr.find('\t', pos1);
    }
    datavec.push_back((float)atof(sr.substr(pos1).c_str()));
    
    return datavec;
}

int main( int argc, char** argv ) {
    
    if(argc < 5) {
        printf("kernelMultiThreads <input Fea> <output file> <normalized option> <kernel type>\n");
        printf("normalized option: 0 -- none\n");
        printf("                   1 -- L2 normalization\n");
        printf("      kernel type: 1 -- chi square\n");
        printf("                   2 -- histogram intersection\n");
        return 1;
    }
    
	char* inputFea = argv[1];
    char* outputfile = argv[2];
    
    long normalizeOp = (long)atoi(argv[3]);
    long kernelType = (long)atoi(argv[4]);
    long Dim = (long)FEALEN;
	
    string s;
    vector<dataType> featuredata;
    ifstream ins;
	
    long fileNum = (long)FILENUM;
    fea_Data = (float*)malloc(sizeof(float) * (long)fileNum * Dim);
    feaLine = (float*)malloc(sizeof(float) * (long)Dim);
    printf("malloc finished...\n");
    printf("file numbers: %ld\n", fileNum);
    printf("dim: %ld\n", Dim);
    float normSum=0;
        
    ins.open(inputFea);
    if(!ins.is_open()) cout << "Error:can't open fea file: " << inputFea << endl;
    for(int i = 0; i < fileNum; i++) {
        getline(ins, s);
        featuredata = splitex(s);
        if (normalizeOp == 1) {
            normSum = 0;
            for (long j = 0; j < FEALEN; j++) {
                normSum += featuredata[j] * featuredata[j];
            }
            normSum = sqrtf(normSum);
            if (normSum != 0) {
                for (long j = 0; j < Dim; j++) {
                    feaLine[j] = featuredata[j] / normSum;
                }
            }
        } else if(normalizeOp == 0) {
            for (long j = 0; j < Dim; j++) {
                feaLine[j] = featuredata[j];
            }
        }
        
        for (long j = 0; j < Dim; j++) {
            fea_Data[(long)i * Dim + j] = feaLine[j];
        }
        //if ((i+1)%100==0) {
            printf("%d read...          \r",i + 1);
        //}
    }
    ins.close();
    printf("\n");
    printf("load finished...\n");
    
    kernel = (float*)malloc(sizeof(float) * fileNum * fileNum);
    
    clock_t timeS, timeE;
    double timeDur;
    timeS = clock();
    
    long num_threads = 1;
#if defined (_OPENMP)
    num_threads = omp_get_num_procs();
    printf("%d threads.\n", num_threads);
    omp_set_num_threads(num_threads);
#endif
    
    for (long i = 0; i < fileNum; i++) {
        float* a = fea_Data + (long)i * Dim;
        for (long j = i; j < fileNum; j++) {
            float* b = fea_Data + (long)j * Dim;
            float d2 = 0;
            char space[64];
            
#if defined(_OPENMP)
#pragma omp parallel for reduction(+:d2)
#endif
            for (long thread_n = 0; thread_n < num_threads; thread_n++) {
                long nStart = thread_n * Dim / num_threads;
                long nEnd = (thread_n + 1) * Dim / num_threads;//nStart + Dim/num_threads;
                for (long k = nStart; k < nEnd; k++) {
                	if(kernelType == 1) {
                    		float d0;
                    		float d1;
                    		d0 = a[k] - b[k];
                    		d0 = d0 * d0;
                    		d1 = a[k] + b[k];
                    		if (d1 != 0)
                        		d2 += d0 / d1;
                        } else if(kernelType == 2) {
                        	d2 += min(a[k], b[k]);
                        }
                }
            }
            
            kernel[i * fileNum + j] = d2;
            kernel[j * fileNum + i] = d2;
            if ((i + 1) % 10 == 0) {
                printf("%d lines finished...        \r", i + 1);
            }
        }
    }
    printf("\n");
    
    timeE = clock();
    timeDur = (double)(timeE - timeS) / CLOCKS_PER_SEC;
    printf("%f seconds.         \n", timeDur);
    
	ofstream outs(outputfile);
	for (long i = 0; i < fileNum; i++) {
        for (long j = 0; j < fileNum; j++) {
            outs << kernel[i * fileNum + j] << '\t';
        }
        outs << endl;
    }
	outs.close();
	
	free(fea_Data);
	free(feaLine);
	free(kernel);
	
	return 0;
}

