/*

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 64, 64, 1)]  0
__________________________________________________________________________________________________
rescaling (Rescaling)           (None, 64, 64, 1)    0           input_1[0][0]
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 32, 32, 144)  720         rescaling[0][0]
__________________________________________________________________________________________________
activation (Activation)         (None, 32, 32, 144)  0           conv2d[0][0]
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 32, 32, 144)  576         activation[0][0]
__________________________________________________________________________________________________
dropout (Dropout)               (None, 32, 32, 144)  0           batch_normalization[0][0]
__________________________________________________________________________________________________
depthwise_conv2d (DepthwiseConv (None, 32, 32, 144)  3744        dropout[0][0]
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 144)  0           depthwise_conv2d[0][0]
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 32, 32, 144)  576         activation_1[0][0]
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_1[0][0]
__________________________________________________________________________________________________
add (Add)                       (None, 32, 32, 144)  0           dropout_1[0][0]
                                                                 dropout[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 144)  20880       add[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 144)  0           conv2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 32, 32, 144)  576         activation_2[0][0]
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_2[0][0]
__________________________________________________________________________________________________
depthwise_conv2d_1 (DepthwiseCo (None, 32, 32, 144)  3744        dropout_2[0][0]
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 144)  0           depthwise_conv2d_1[0][0]
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 32, 32, 144)  576         activation_3[0][0]
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_3[0][0]
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 144)  0           dropout_3[0][0]
                                                                 dropout_2[0][0]
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 144)  20880       add_1[0][0]
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 144)  0           conv2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 32, 32, 144)  576         activation_4[0][0]
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_4[0][0]
__________________________________________________________________________________________________
depthwise_conv2d_2 (DepthwiseCo (None, 32, 32, 144)  3744        dropout_4[0][0]
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 144)  0           depthwise_conv2d_2[0][0]
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 32, 32, 144)  576         activation_5[0][0]
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_5[0][0]
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 144)  0           dropout_5[0][0]
                                                                 dropout_4[0][0]
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 144)  20880       add_2[0][0]
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 144)  0           conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 32, 32, 144)  576         activation_6[0][0]
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_6[0][0]
__________________________________________________________________________________________________
depthwise_conv2d_3 (DepthwiseCo (None, 32, 32, 144)  3744        dropout_6[0][0]
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 144)  0           depthwise_conv2d_3[0][0]
__________________________________________________________________________________________________
batch_normalization_7 (BatchNor (None, 32, 32, 144)  576         activation_7[0][0]
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_7[0][0]
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 144)  0           dropout_7[0][0]
                                                                 dropout_6[0][0]
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 144)  20880       add_3[0][0]
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 144)  0           conv2d_4[0][0]
__________________________________________________________________________________________________
batch_normalization_8 (BatchNor (None, 32, 32, 144)  576         activation_8[0][0]
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, 32, 32, 144)  0           batch_normalization_8[0][0]
__________________________________________________________________________________________________
global_average_pooling2d (Globa (None, 144)          0           dropout_8[0][0]
__________________________________________________________________________________________________
dense (Dense)                   (None, 3225)         467625      global_average_pooling2d[0][0]
==================================================================================================
Total params: 572,025
Trainable params: 569,433
Non-trainable params: 2,592
__________________________________________________________________________________________________

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <cmath>
#include <string>
#include <codecvt>
#include <Windows.h> 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

class jp_classifier
{
private:

    const float eps = 0.001f;

    // jp character map
    vector<wstring> jpMap;

    float** input;

    // weights
    float**** const0, **** const6, **** const12, **** const18, **** const24, **** const30,
        **** const36, **** const42, **** const48;

    float** const54;

    // bias/normalization
    float* const1, * const2, * const3, * const4, * const5, * const7, * const8, * const9,
        * const10, * const11, * const13, * const14, * const15, * const16, * const17,
        * const19, * const20, * const21, * const22, * const23, * const25, * const26,
        * const27, * const28, * const29, * const31, * const32, * const33, * const34,
        * const35, * const37, * const38, * const39, * const40, * const41, * const43,
        * const44, * const45, * const46, * const47, * const49, * const50, * const51,
        * const52, * const53, * const55;

    float*** l0, *** l1, *** l2, *** l3, *** l4, *** l5, *** l6, *** l7, *** l8;

    float* l9, *l10;

    float**** getArray(ifstream* fin, int mi, int mj, int mk, int ml)
    {
        float**** buff = (float****)createArray(mi, mj, mk, ml, sizeof(float));
        for (int i = 0; i < mi; i++) {
            for (int j = 0; j < mj; j++) {
                for (int k = 0; k < mk; k++) {
                    fin->read(reinterpret_cast<char*>(buff[i][j][k]), sizeof(float) * ml);
                }
            }
        }
        return buff;
    }

    float*** getArray(ifstream* fin, int mi, int mj, int mk)
    {
        float*** buff = (float***)createArray(mi, mj, mk, sizeof(float));
        for (int i = 0; i < mi; i++) {
            for (int j = 0; j < mj; j++) {
                fin->read(reinterpret_cast<char*>(buff[i][j]), sizeof(float) * mk);
            }
        }
        return buff;
    }

    float** getArray(ifstream* fin, int mi, int mj)
    {
        float** buff = (float**)createArray(mi, mj, sizeof(float));
        for (int i = 0; i < mi; i++) {
            fin->read(reinterpret_cast<char*>(buff[i]), sizeof(float) * mj);
        }
        return buff;
    }

    float* getArray(ifstream* fin, int mi)
    {
        float* buff = (float*)malloc(mi * sizeof(float));
        fin->read(reinterpret_cast<char*>(buff), sizeof(float) * mi);
        return buff;
    }

    // https://www.johndcook.com/blog/cpp_erf/
    float erf(float x)
    {
        // constants
        float a1 = 0.254829592;
        float a2 = -0.284496736;
        float a3 = 1.421413741;
        float a4 = -1.453152027;
        float a5 = 1.061405429;
        float p = 0.3275911;

        // Save the sign of x
        int sign = 1;
        if (x < 0)
            sign = -1;
        x = fabs(x);

        // A&S formula 7.1.26
        float t = 1.0 / (1.0 + p * x);
        float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

        return sign * y;
    }

    void testErf()
    {
        // Select a few input values
        float x[] =
        {
            -3,
            -1,
            0.0,
            0.5,
            2.1
        };

        // Output computed by Mathematica
        // y = Erf[x]
        float y[] =
        {
            -0.999977909503,
            -0.842700792950,
            0.0,
            0.520499877813,
            0.997020533344
        };

        int numTests = sizeof(x) / sizeof(float);

        float maxError = 0.0;
        for (int i = 0; i < numTests; ++i) {
            float error = fabs(y[i] - erf(x[i]));
            if (error > maxError)
                maxError = error;
        }

        std::cout << "Maximum error: " << maxError << "\n";
    }

    inline float GELU(float x)
    {
        //// APROX
        //x = x + 0.044715f * powf(x, 3);
        //// 1 + tanh(sqrt(2/pi) * x)
        //x = (1.0f + tanhf(0.79788456f * x)) * 0.5f * x;
        //return x;
        // poly approx
        return 0.5f * x * (1.0f + erf(x / 1.4142135624f));
    }

    inline float batchNorm(float x, float gamma, float beta, float mean, float var)
    {
        //z1_hat = (x - pop_mean) / sqrt(pop_var + epsilon)
        //  BN1 = gamma * z1_hat + beta
        return ((x - mean) / sqrtf(var + eps)) * gamma + beta;
    }

    inline float padLayerEven(float*** layer, int x, int y, int z, int xm, int ym)
    {
        if (x < 2 || y < 2 || x > xm + 1 || y > ym + 1) return 0.0f;
        return layer[x - 2][y - 2][z];
    }

public:
    // Annoying mallocs
    static float** createArray(int i, int j, size_t size)
    {
        float** r = new float* [i * sizeof(float*)];
        for (int x = 0; x < i; x++) {
            r[x] = new float[j * size];
        }
        return r;
    }

    static float*** createArray(int i, int j, int k, size_t size)
    {
        float*** r = new float** [i * sizeof(float*)];
        for (int x = 0; x < i; x++) {
            r[x] = new float* [j * sizeof(float*)];
            for (int y = 0; y < j; y++) {
                r[x][y] = new float[k * size];
            }
        }
        return r;
    }

    static float**** createArray(int i, int j, int k, int l, size_t size)
    {
        float**** r = new float*** [i * sizeof(float*)];
        for (int x = 0; x < i; x++) {
            r[x] = new float** [j * sizeof(float*)];
            for (int y = 0; y < j; y++) {
                r[x][y] = new float* [k * sizeof(float*)];
                for (int z = 0; z < k; z++) {
                    r[x][y][z] = new float[l * size];
                }
            }
        }
        return r;
    }

    // Annoying malloc frees
    static void freeArray(int i, float* a)
    {
        delete[] a;
    }

    static void freeArray(int i, int j, float** a)
    {
        for (int x = 0; x < i; x++) {
            delete[] a[x];
        }
        delete[] a;
    }

    static void freeArray(int i, int j, int k, float*** a)
    {
        for (int x = 0; x < i; x++) {
            for (int y = 0; y < j; y++) {
                delete[] a[x][y];
            }
            delete[] a[x];
        }
        delete[] a;
    }

    static void freeArray(int i, int j, int k, int l, float**** a)
    {
        for (int x = 0; x < i; x++) {
            for (int y = 0; y < j; y++) {
                for (int z = 0; z < k; z++) {
                    delete[] a[x][y][z];
                }
                delete[] a[x][y];
            }
            delete[] a[x];
        }
        delete[] a;
    }

    float** loadAsArray(string path, int* width, int* height, int* channels)
    {
        // Not HDR, just need raw float values
        stbi_ldr_to_hdr_gamma(1.0f);
        float* img = stbi_loadf(path.c_str(), width, height, channels, 0);
        if (img == NULL) {
            cout << "Error loading " << path << endl;
            exit(1);
        }
        printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", *width, *height, *channels);

        float** img_array = createArray(*width, *height, sizeof(float));
        for (int h = 0; h < *height; h++)
            for (int w = 0; w < *width; w++)
                img_array[h][w] = img[h * *height + w];

        stbi_image_free(img);
        return img_array;
    }

    jp_classifier(string path, string pathSeq2Text)
    {
        wifstream fin3(pathSeq2Text);
        if (!fin3) {
            cout << "error opening jp file" << endl;
            exit(-1);
        }

        for (int i = 0; i < 3; i++) jpMap.push_back(L"");

        wstring wline;
        wstring wtoken;
        while (std::getline(fin3, wline))
        {
            wstringstream ss(wline);
            getline(ss, wtoken, L'\t');
            jpMap.push_back(wtoken);
        }

        fin3.close();

        ifstream fin(path, ios::binary);
        if (!fin) {
            cout << "error opening stream" << endl;
            exit(-1);
        }

        // read weights
        const0 = getArray(&fin, 2, 2, 1, 144);
        const1 = getArray(&fin, 144);
        const2 = getArray(&fin, 144);
        const3 = getArray(&fin, 144);
        const4 = getArray(&fin, 144);
        const5 = getArray(&fin, 144);
        const6 = getArray(&fin, 5, 5, 144, 1);
        const7 = getArray(&fin, 144);
        const8 = getArray(&fin, 144);
        const9 = getArray(&fin, 144);
        const10 = getArray(&fin, 144);
        const11 = getArray(&fin, 144);
        const12 = getArray(&fin, 1, 1, 144, 144);
        const13 = getArray(&fin, 144);
        const14 = getArray(&fin, 144);
        const15 = getArray(&fin, 144);
        const16 = getArray(&fin, 144);
        const17 = getArray(&fin, 144);
        const18 = getArray(&fin, 5, 5, 144, 1);
        const19 = getArray(&fin, 144);
        const20 = getArray(&fin, 144);
        const21 = getArray(&fin, 144);
        const22 = getArray(&fin, 144);
        const23 = getArray(&fin, 144);
        const24 = getArray(&fin, 1, 1, 144, 144);
        const25 = getArray(&fin, 144);
        const26 = getArray(&fin, 144);
        const27 = getArray(&fin, 144);
        const28 = getArray(&fin, 144);
        const29 = getArray(&fin, 144);
        const30 = getArray(&fin, 5, 5, 144, 1);
        const31 = getArray(&fin, 144);
        const32 = getArray(&fin, 144);
        const33 = getArray(&fin, 144);
        const34 = getArray(&fin, 144);
        const35 = getArray(&fin, 144);
        const36 = getArray(&fin, 1, 1, 144, 144);
        const37 = getArray(&fin, 144);
        const38 = getArray(&fin, 144);
        const39 = getArray(&fin, 144);
        const40 = getArray(&fin, 144);
        const41 = getArray(&fin, 144);
        const42 = getArray(&fin, 5, 5, 144, 1);
        const43 = getArray(&fin, 144);
        const44 = getArray(&fin, 144);
        const45 = getArray(&fin, 144);
        const46 = getArray(&fin, 144);
        const47 = getArray(&fin, 144);
        const48 = getArray(&fin, 1, 1, 144, 144);
        const49 = getArray(&fin, 144);
        const50 = getArray(&fin, 144);
        const51 = getArray(&fin, 144);
        const52 = getArray(&fin, 144);
        const53 = getArray(&fin, 144);
        const54 = getArray(&fin, 144, 3225);
        const55 = getArray(&fin, 3225);

        fin.close();

        // allocate outputs
        l0 = (float***)createArray(32, 32, 144, sizeof(float));
        l1 = (float***)createArray(32, 32, 144, sizeof(float));
        l2 = (float***)createArray(32, 32, 144, sizeof(float));
        l3 = (float***)createArray(32, 32, 144, sizeof(float));
        l4 = (float***)createArray(32, 32, 144, sizeof(float));
        l5 = (float***)createArray(32, 32, 144, sizeof(float));
        l6 = (float***)createArray(32, 32, 144, sizeof(float));
        l7 = (float***)createArray(32, 32, 144, sizeof(float));
        l8 = (float***)createArray(32, 32, 144, sizeof(float));
        l9 = new float[144];
        l10 = new float[3225];
    }

    ~jp_classifier()
    {
        freeArray(2, 2, 1, 144, const0);
        freeArray(144, const1);
        freeArray(144, const2);
        freeArray(144, const3);
        freeArray(144, const4);
        freeArray(144, const5);
        freeArray(5, 5, 144, 1, const6);
        freeArray(144, const7);
        freeArray(144, const8);
        freeArray(144, const9);
        freeArray(144, const10);
        freeArray(144, const11);
        freeArray(1, 1, 144, 144, const12);
        freeArray(144, const13);
        freeArray(144, const14);
        freeArray(144, const15);
        freeArray(144, const16);
        freeArray(144, const17);
        freeArray(5, 5, 144, 1, const18);
        freeArray(144, const19);
        freeArray(144, const20);
        freeArray(144, const21);
        freeArray(144, const22);
        freeArray(144, const23);
        freeArray(1, 1, 144, 144, const24);
        freeArray(144, const25);
        freeArray(144, const26);
        freeArray(144, const27);
        freeArray(144, const28);
        freeArray(144, const29);
        freeArray(5, 5, 144, 1, const30);
        freeArray(144, const31);
        freeArray(144, const32);
        freeArray(144, const33);
        freeArray(144, const34);
        freeArray(144, const35);
        freeArray(1, 1, 144, 144, const36);
        freeArray(144, const37);
        freeArray(144, const38);
        freeArray(144, const39);
        freeArray(144, const40);
        freeArray(144, const41);
        freeArray(5, 5, 144, 1, const42);
        freeArray(144, const43);
        freeArray(144, const44);
        freeArray(144, const45);
        freeArray(144, const46);
        freeArray(144, const47);
        freeArray(1, 1, 144, 144, const48);
        freeArray(144, const49);
        freeArray(144, const50);
        freeArray(144, const51);
        freeArray(144, const52);
        freeArray(144, const53);
        freeArray(144, 3225, const54);
        freeArray(3225, const55);
        freeArray(32, 32, 144, l0);
        freeArray(32, 32, 144, l1);
        freeArray(32, 32, 144, l2);
        freeArray(32, 32, 144, l3);
        freeArray(32, 32, 144, l4);
        freeArray(32, 32, 144, l5);
        freeArray(32, 32, 144, l6);
        freeArray(32, 32, 144, l7);
        freeArray(32, 32, 144, l8);
        delete[] l9;
        delete[] l10;
    }

    // convnet "patches"
    void kernelPaddedEvenL0(float*** cl, float**** cw, float* bias,
        float* gamma, float* beta, float* mm, float* mv, float** pl, int im, int jm, int k)
    {
        for (int i = 0; i < im; i++) {
            for (int j = 0; j < jm; j++) {
                cl[i][j][k] = 0.0f;
                int i0 = i * 2, i1 = i0 + 1;
                int j0 = j * 2, j1 = j0 + 1;
                // kernel
                cl[i][j][k] +=
                    pl[i0][j0] * cw[0][0][0][k] +
                    pl[i0][j1] * cw[0][1][0][k] +
                    pl[i1][j0] * cw[1][0][0][k] +
                    pl[i1][j1] * cw[1][1][0][k];
                // bias
                cl[i][j][k] = cl[i][j][k] + bias[k];
                // activation
                cl[i][j][k] = GELU(cl[i][j][k]);
                // batch norm
                cl[i][j][k] = batchNorm(cl[i][j][k], gamma[k], beta[k], mm[k], mv[k]);
            }
        }
    }

    // depth wise convolution
    void depthConv5x5(float*** cl, float**** cw, float* bias,
        float* gamma, float* beta, float* mm, float* mv, float*** pl, int im, int jm, int k)
    {
        for (int i = 0; i < im; i++) {
            for (int j = 0; j < jm; j++) {
                int i0 = i, i1 = i0 + 1, i2 = i0 + 2, i3 = i0 + 3, i4 = i0 + 4;
                int j0 = j, j1 = j0 + 1, j2 = j0 + 2, j3 = j0 + 3, j4 = j0 + 4;
                // 5x5 depth wise kernel
                cl[i][j][k] =
                    padLayerEven(pl, i0, j0, k, im, jm) * cw[0][0][k][0] +
                    padLayerEven(pl, i0, j1, k, im, jm) * cw[0][1][k][0] +
                    padLayerEven(pl, i0, j2, k, im, jm) * cw[0][2][k][0] +
                    padLayerEven(pl, i0, j3, k, im, jm) * cw[0][3][k][0] +
                    padLayerEven(pl, i0, j4, k, im, jm) * cw[0][4][k][0] +
                    padLayerEven(pl, i1, j0, k, im, jm) * cw[1][0][k][0] +
                    padLayerEven(pl, i1, j1, k, im, jm) * cw[1][1][k][0] +
                    padLayerEven(pl, i1, j2, k, im, jm) * cw[1][2][k][0] +
                    padLayerEven(pl, i1, j3, k, im, jm) * cw[1][3][k][0] +
                    padLayerEven(pl, i1, j4, k, im, jm) * cw[1][4][k][0] +
                    padLayerEven(pl, i2, j0, k, im, jm) * cw[2][0][k][0] +
                    padLayerEven(pl, i2, j1, k, im, jm) * cw[2][1][k][0] +
                    padLayerEven(pl, i2, j2, k, im, jm) * cw[2][2][k][0] +
                    padLayerEven(pl, i2, j3, k, im, jm) * cw[2][3][k][0] +
                    padLayerEven(pl, i2, j4, k, im, jm) * cw[2][4][k][0] +
                    padLayerEven(pl, i3, j0, k, im, jm) * cw[3][0][k][0] +
                    padLayerEven(pl, i3, j1, k, im, jm) * cw[3][1][k][0] +
                    padLayerEven(pl, i3, j2, k, im, jm) * cw[3][2][k][0] +
                    padLayerEven(pl, i3, j3, k, im, jm) * cw[3][3][k][0] +
                    padLayerEven(pl, i3, j4, k, im, jm) * cw[3][4][k][0] +
                    padLayerEven(pl, i4, j0, k, im, jm) * cw[4][0][k][0] +
                    padLayerEven(pl, i4, j1, k, im, jm) * cw[4][1][k][0] +
                    padLayerEven(pl, i4, j2, k, im, jm) * cw[4][2][k][0] +
                    padLayerEven(pl, i4, j3, k, im, jm) * cw[4][3][k][0] +
                    padLayerEven(pl, i4, j4, k, im, jm) * cw[4][4][k][0];

                // bias
                cl[i][j][k] = cl[i][j][k] + bias[k];
                // activation
                cl[i][j][k] = GELU(cl[i][j][k]);
                // batch norm
                cl[i][j][k] = batchNorm(cl[i][j][k], gamma[k], beta[k], mm[k], mv[k]);
            }
        }
    }

    // point wise convolution
    void pointConv1x1(float*** cl, float**** cw, float* bias,
        float* gamma, float* beta, float* mm, float* mv, float*** pl1, float*** pl2,
        int im, int jm, int k, int lm)
    {
        for (int i = 0; i < im; i++) {
            for (int j = 0; j < jm; j++) {
                cl[i][j][k] = 0.0f;
                // 1x1 kernel
                for (int l = 0; l < lm; l++) {
                    cl[i][j][k] += (pl1[i][j][l] + pl2[i][j][l]) * cw[0][0][l][k];
                }
                // bias
                cl[i][j][k] = cl[i][j][k] + bias[k];
                // activation
                cl[i][j][k] = GELU(cl[i][j][k]);
                // batch norm
                cl[i][j][k] = batchNorm(cl[i][j][k], gamma[k], beta[k], mm[k], mv[k]);
            }
        }
    }

    // average layers
    void globalPool(float* cl, float*** pl, int im, int jm, int k)
    {
        cl[k] = 0.0f;

        for (int i = 0; i < im; i++) {
            for (int j = 0; j < jm; j++) {
                cl[k] += pl[i][j][k];
            }
        }

        cl[k] /= 1024.0f;
    }

    // dense layer with softmax
    void denseLayer(float* cl, float** cw, float* bias, float* pl, int k, int lm)
    {
        cl[k] = 0.0f;

        for (int l = 0; l < lm; l++) {
            cl[k] += pl[l] * cw[l][k];
        }
        cl[k] = cl[k] + bias[k];

        // softmax at the end to get the % but it can be ommited for prediction
    }

    void forwardProp(float** imgIn)
    {
        input = imgIn;

        vector<thread> threads;

        // L0, kernel=2x2, stride=2, padding=even, batch norm
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::kernelPaddedEvenL0, this, l0, const0, const1,
                const2, const3, const4, const5, input, 32, 32, k);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        //// Same depthwise/pointwise convolutions looped but I kept the functions seperate

        // L1, depthwise conv=5x5x144, stride=1, padding=even, batch norm
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::depthConv5x5, this, l1, const6, const7,
                const8, const9, const10, const11, l0, 32, 32, k);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L2, kernel=1x1, stride=1, padding=even, add in L0
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::pointConv1x1, this, l2, const12, const13,
                const14, const15, const16, const17, l1, l0, 32, 32, k, 144);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L3, depthwise conv=5x5x144, stride=1, padding=even, batch norm
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::depthConv5x5, this, l3, const18, const19,
                const20, const21, const22, const23, l2, 32, 32, k);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L4, kernel=1x1, stride=1, padding=even, add in L2
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::pointConv1x1, this, l4, const24, const25,
                const26, const27, const28, const29, l3, l2, 32, 32, k, 144);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L5, depthwise conv=5x5x144, stride=1, padding=even, batch norm
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::depthConv5x5, this, l5, const30, const31,
                const32, const33, const34, const35, l4, 32, 32, k);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L6, kernel=1x1, stride=1, padding=even, add in L4
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::pointConv1x1, this, l6, const36, const37,
                const38, const39, const40, const41, l5, l4, 32, 32, k, 144);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L7, depthwise conv=5x5x144, stride=1, padding=even, batch norm
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::depthConv5x5, this, l7, const42, const43,
                const44, const45, const46, const47, l6, 32, 32, k);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L8, kernel=1x1, stride=1, padding=even, add in L6
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::pointConv1x1, this, l8, const48, const49,
                const50, const51, const52, const53, l7, l6, 32, 32, k, 144);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L9, global pooling
        for (int k = 0; k < 144; k++) {
            thread t(&jp_classifier::globalPool, this, l9, l8, 32, 32, k);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        // L10, classify
        for (int k = 0; k < 3225; k++) {
            thread t(&jp_classifier::denseLayer, this, l10, const54, const55, l9, k, 144);
            threads.push_back(move(t));
        }
        for (auto& th : threads) th.join();
        threads.clear();

        float maxNum = FLT_MIN;
        int maxInd = 0;
        for (int i = 0; i < 3225; i++)
        {
            float score = l10[i];
            maxInd = score > maxNum ? i : maxInd;
            maxNum = score > maxNum ? score : maxNum;
        }

        wcout << L"Predicted: " << jpMap[maxInd + 3];
    }
};

int main()
{
    SetConsoleOutputCP(CP_UTF8);
    
    // yes it's hardcoded cause i'm too lazy
    string BAKED_PATH = "D:\\Storage\\Python\\convmixer\\model\\jp_convmixer.bytes";
    string IMG_PATH = "D:\\Storage\\Python\\convmixer\\input\\input1.jpg";
    string PATHSEQ2TEXT = ".\\jp_seq2text.tsv";

    jp_classifier classifier = jp_classifier(BAKED_PATH, PATHSEQ2TEXT);

    int width, height, channels;
    float** imgin = classifier.loadAsArray(IMG_PATH, &width, &height, &channels);

	//for (int i = 0; i < 64; i++) {
	//	for (int j = 0; j < 64; j++) {
	//		imgin[i][j] = (i / 63.0f) * (j / (63.0f * 0.5));
	//	}
	//}

    if (width != 64 || height != 64 || channels != 1) exit(1);
    classifier.forwardProp(imgin);

    // wait for input
    getc(stdin);

    classifier.freeArray(64, 64, imgin);
    return 0;
}