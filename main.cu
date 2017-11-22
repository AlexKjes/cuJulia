#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>


typedef unsigned int uint;
typedef unsigned char uchar;

__device__
void cmap(uint value, uchar *pixel){

    pixel[1] = (255-value)*.4;
    pixel[2] = (value%10)*255/10;
    pixel[0] = 255*(1-(255/(255+value)))*.65;

}

__global__
void julia(uchar *img, uint width, double r_from, double i_from, double step, cuDoubleComplex c, uint n_iter, uint MAX){

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < MAX) {
        uint x = index % width;
        uint y = index / width;
        cuDoubleComplex z = make_cuDoubleComplex(r_from + x * step, i_from + y * step);
        uint iter = 0;
        for (iter = 0; iter < n_iter && cuCabs(cuCmul(z, z)) < 4; iter++) {
            z = cuCadd(cuCmul(z, z), c);
        }
        cmap(iter, &img[index*3]);
    }
}

void savebmp(char *name,uchar *buffer,int x,int y) {
    FILE *f=fopen(name,"wb");
    if(!f) {
        printf("Error writing image to disk.\n");
        return;
    }
    unsigned int size=x*y*3+54;
    uchar header[54]={'B','M',size&255,(size>>8)&255,(size>>16)&255,size>>24,0,
                      0,0,0,54,0,0,0,40,0,0,0,x&255,x>>8,0,0,y&255,y>>8,0,0,1,0,24,0,0,0,0,0,0,
                      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    fwrite(header,1,54,f);
    fwrite(buffer,1,x*y*3,f);
    fclose(f);
}


int main() {

    // image params
    uint X = 1920;  // resulting image width
    uint Y = 1080;  // resulting image height
    uint N = X*Y;   // total pixels

    // julia params
    cuDoubleComplex c = make_cuDoubleComplex(-.788, .145);
    double r_min = -.18; // real lower bound
    double r_max = -.08; // real upper bound
    double step = (r_max-r_min)/X; // step length
    double i_min = 0.02;  // imaginary lower bound
    uint max_iter = 255;  // max number of iterations per pixel

    // memalloc
    uchar* img = (uchar*)calloc(N*3, sizeof(uchar));
    uchar* d_img;
    cudaMalloc(&d_img, N*3*sizeof(uchar));


    julia <<< (N / 1024)+(N%1024==0?0:1), 1024 >>> (d_img, X, r_min, i_min, step, c, max_iter, N);
    cudaMemcpy(img, d_img, N * 3 * sizeof(uchar), cudaMemcpyDeviceToHost);
    savebmp("img.bmp", img, X, Y);


    return 0;
}

