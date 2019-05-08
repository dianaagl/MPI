#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

const int TAG = 65;
const char* fileName = "output_1.txt";

int N_PROC, PROC_ID;

int MAX_ITER;
int M, N, M_TOTAL; // i, j, k
int FROM, TO;
float* bData, *mData;
int min(int a, int b) {
    return a < b ? a: b;
}

float borderFunc(int i, int j, int k) {
    return i + j + k;
}

int index(int i,int j,int k){
    return ((((k) - FROM) * M + (i)) * N + (j));
}

void calcCurrentBlock(void) {
    int m = (M_TOTAL - 2) / N_PROC;
    if (PROC_ID < (M_TOTAL - 2) % N_PROC) m++;

    // Process matrices from (M, N, FROM) to (M, N, TO)
    FROM = (M_TOTAL - 2) / N_PROC * PROC_ID + min((M_TOTAL - 2) % N_PROC, PROC_ID);
    TO = FROM + m + 1;
}


float* allocDataArray(int dataSize) {
    float *data = (float*)malloc(dataSize * sizeof(float));

    if (data == NULL) {
        printf("%d can't allocate %d bytes\n",
               PROC_ID + 1, dataSize * sizeof(float));
        MPI_Finalize(); 
        exit(1);
    }
    return data;
}


void _writeMatricesBinary(FILE *f, float* data, size_t from, size_t to) {
    fwrite(
        data + index(0, 0, from), sizeof(float),
        index(0, 0, to + 1) - index(0, 0, from), f);
}


void _writeMatricesText(FILE *f, float* data, size_t from, size_t to) {
   
    for (int w = from; w <= to; ++w) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                fprintf(f,"%.2f ", data[index(i, j, w)]);
            }
            fprintf(f,"\n");
        }
        fprintf(f, "\n\n");
    }
}


void writeMatrices(FILE *f, float *data, size_t from, size_t to) {
    _writeMatricesText(f, data, from, to);
}

void writeToFile(FILE *f, float* data, size_t from, size_t to) {
    _writeMatricesText(f, data, from, to);
}
void writeResult() {
    FILE *fs;
    float *d = bData;
    
    MPI_Status st;
    printf("%d", PROC_ID);
    if (PROC_ID == 0) {
        fs = fopen(fileName, "w");
        writeToFile(fs, d, 0, TO - 1);
        fclose(fs);
        printf("o %d", PROC_ID);
        MPI_Send(NULL, 0, MPI_DOUBLE, 1, TAG, MPI_COMM_WORLD);
        
    }
    else if (PROC_ID == N_PROC - 1) {    
        
        MPI_Recv(NULL, 0, MPI_DOUBLE, N_PROC - 2, TAG, MPI_COMM_WORLD, &st);
        printf("l %d", PROC_ID);
        fs = fopen(fileName, "ab");
        writeToFile(fs, d, FROM + 1, TO);
        fclose(fs);
    }
    else {
        MPI_Recv(NULL, 0, MPI_DOUBLE, PROC_ID - 1, TAG, MPI_COMM_WORLD, &st);
        printf("else %d %d %d", PROC_ID, FROM, TO);
        fs = fopen(fileName, "ab");
        writeToFile(fs, d, FROM + 1, TO - 1);
        fclose(fs);
        MPI_Send(NULL, 0, MPI_DOUBLE, PROC_ID + 1, TAG, MPI_COMM_WORLD); 
    } 
}


int main(int argc, char **argv) {

    MAX_ITER = 500;
    // stack M_TOTAL matrices of sizes (M, N)
    M = 200;
    N = 200;
    M_TOTAL = 200;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &N_PROC);
    MPI_Comm_rank(MPI_COMM_WORLD, &PROC_ID); 

    double start = MPI_Wtime();
    calcCurrentBlock();
    MPI_Barrier(MPI_COMM_WORLD);

    size_t dataSize = M * N * (TO - FROM + 1);
    bData = allocDataArray(dataSize);
    mData = allocDataArray(dataSize);

    // Initialize data array
    memset(bData, 0, dataSize * sizeof(float));
    memset(mData, 0, dataSize * sizeof(float));
    for (size_t w = FROM; w <= TO; ++w) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (w == 0 || i == 0 || j == 0 ||
                    w == M_TOTAL - 1 || i == M - 1 || j == N - 1)
                {
                    bData[index(i, j, w)] = borderFunc(i, j, w);
                    mData[index(i, j, w)] = bData[index(i, j, w)];
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Request sendReq;
    MPI_Status status;

    MPI_Request recvReqs[2];
    int waitRecv[2] = {};

    for (int iter = 1; iter <= MAX_ITER; ++iter) {
        // Asynchronously receive data
        if (PROC_ID > 0) {
            waitRecv[0] = 1;
            MPI_Irecv(mData, M * N, MPI_FLOAT, PROC_ID - 1,
                      0, MPI_COMM_WORLD, &recvReqs[0]);
        } else {
            waitRecv[0] = 0;
        }
        if (PROC_ID < N_PROC - 1) {
            waitRecv[1] = 1;
            MPI_Irecv(mData + index(0, 0, TO), M * N, MPI_FLOAT, PROC_ID + 1,
                      1, MPI_COMM_WORLD, &recvReqs[1]);
        } else {
            waitRecv[1] = 0;
        }

        // Calc next iteration
        float *d = bData;
        for (size_t w = FROM + 1; w <= TO - 1; ++w) {
            for (size_t i = 1; i < M - 1; ++i) {
                for (size_t j = 1; j < N - 1; ++j) {
                    mData[index(i, j, w)] = (
                        d[index(i + 1, j, w)] + d[index(i - 1, j, w)] +
                        d[index(i, j + 1, w)] + d[index(i, j - 1, w)] +
                        d[index(i, j, w + 1)] + d[index(i, j, w - 1)]
                    ) / 6.0;
                }
            }
        }

        // Send data
        if (PROC_ID > 0) {
            MPI_Isend(mData + index(0, 0, FROM + 1), M * N, MPI_FLOAT, PROC_ID - 1,
                      1, MPI_COMM_WORLD, &sendReq);
            MPI_Wait(&sendReq, &status);
        }
        if (PROC_ID < N_PROC - 1) {
            MPI_Isend(mData + index(0, 0, TO - 1), M * N, MPI_FLOAT, PROC_ID + 1,
                      0, MPI_COMM_WORLD, &sendReq);
            MPI_Wait(&sendReq, &status); 
        }
        // Wait for received data
        if (waitRecv[0]) {
            MPI_Wait(&recvReqs[0], &status);
        }
        if (waitRecv[1]) {
            MPI_Wait(&recvReqs[1], &status);
        }

        bData = mData;
        mData = d;
    }

    FILE *fs;
    
    fs = fopen("fout.txt", "ab");
    fprintf(fs,"%lf\n", MPI_Wtime() - start);
    fclose(fs);
    MPI_Barrier(MPI_COMM_WORLD);
    writeResult();
    free(bData);
    free(mData);
    MPI_Finalize(); 
    return 0;
}


