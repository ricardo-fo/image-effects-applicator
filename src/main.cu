#include <stdio.h>

// read a ppm image from a file
unsigned char *read_ppm(const char *filename, int *width, int *height, int *maxval)
{
    FILE *fp;
    unsigned char *data_cpu;
    int w, h, m;

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        printf("Error reading file\n");
        exit(-1);
    }

    // read the header
    fscanf(fp, "P6\n");
    fscanf(fp, "%d %d\n", &w, &h);
    fscanf(fp, "%d\n", &m);

    // allocate space for the image data_cpu
    data_cpu = (unsigned char *)malloc(3*w*h*sizeof(unsigned char));
    if (data_cpu == NULL)
    {
        printf("Error reading file\n");
        exit(-1);
    }

    // read the image data_cpu
    fread(data_cpu, sizeof(unsigned char), 3*w*h, fp);

    // close the file and return the image data_cpu
    fclose(fp);
    *width = w;
    *height = h;
    *maxval = m;
    return data_cpu;
}

// write a ppm image to a file
void write_ppm(const char *filename, int width, int height, int maxval, unsigned char *data_cpu)
{
    FILE *fp;

    fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("Error writing file\n");
        exit(-1);
    }

    // write the header
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "%d\n", maxval);

    // write the image data_cpu
    fwrite(data_cpu, sizeof(unsigned char), 3*width*height, fp);

    // close the file
    fclose(fp);
}

// copy a ppm image to gpu memory using cudaMemcpy
void copy_ppm_to_gpu(int width, int height, unsigned char *data_cpu, unsigned char **data_gpu)
{
    // allocate space for the image data_cpu on the device
    cudaMalloc(data_gpu, 3*width*height*sizeof(unsigned char));

    // copy the image data_cpu to the device
    cudaMemcpy(*data_gpu, data_cpu, 3*width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
}

// copy a ppm image from gpu memory using cudaMemcpy
void copy_ppm_from_gpu(int width, int height, unsigned char *data_cpu, unsigned char *data_gpu)
{
    // copy the image data_cpu from the device
    cudaMemcpy(data_cpu, data_gpu, 3*width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

// transform a ppm image to grayscale using CUDA
__global__ void transform_ppm_to_grayscale(int width, int height, unsigned char *data_gpu)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int length = width*height;
    int section_size = (length-1) / (blockDim.x*gridDim.x) + 1;
    int start = x*section_size;

    for (int k = 0; k < section_size; k++) {
        if (start+k < length) {
            int index = 3*(start+k);
            int gray = (data_gpu[index] + data_gpu[index+1] + data_gpu[index+2]) / 3;
            data_gpu[index] = gray;
            data_gpu[index+1] = gray;
            data_gpu[index+2] = gray;
        }
    }
}

// transform a ppm image to greenscale using CUDA
__global__ void transform_ppm_to_greenscale(int width, int height, unsigned char *data_gpu)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int length = width*height;
    int section_size = (length-1) / (blockDim.x*gridDim.x) + 1;
    int start = x*section_size;

    for (int k = 0; k < section_size; k++) {
        if (start+k < length) {
            int index = 3*(start+k);
            data_gpu[index] = 0;
            data_gpu[index+2] = 0;
        }
    }
}

int main() {
    int width, height, maxval;
    unsigned char *data_cpu;
    unsigned char *data_gpu;
    cudaDeviceProp pdev;

    // get gpu properties
    cudaGetDeviceProperties(&pdev, 0);

    // read the image data
    data_cpu = read_ppm("img/sample.ppm", &width, &height, &maxval);

    // copy the image data to gpu memory
    copy_ppm_to_gpu(width, height, data_cpu, &data_gpu);

    // transform the image data on the device
    transform_ppm_to_grayscale<<<(width * height + pdev.maxThreadsDim[0]) / pdev.maxThreadsDim[0], pdev.maxThreadsDim[0]>>>(width, height, data_gpu);

    // copy the image data from gpu memory
    copy_ppm_from_gpu(width, height, data_cpu, data_gpu);

    // write the image data to a new file
    write_ppm("img/sample_copy.ppm", width, height, maxval, data_cpu);

    // free the image data
    free(data_cpu);

    // free the image data on the device
    cudaFree(data_gpu);

    return 0;
}
