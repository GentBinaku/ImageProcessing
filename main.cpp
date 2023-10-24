#include <iostream>
#include <vector>
#include <cmath>
#include <dbg.h>

#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


#define M_PI 3.142857

typedef struct {
    int nb_row, nb_col, zoom;
    float light;
    float *r, *g, *b;
} t_image_rgb;

vector<vector<int>> quantizationMatrix = {
        {16, 11, 10, 16, 24,  40,  51,  61},
        {12, 12, 14, 19, 26,  58,  60,  55},
        {14, 13, 16, 24, 40,  57,  69,  56},
        {14, 17, 22, 29, 51,  87,  80,  62},
        {18, 22, 37, 56, 68,  109, 103, 77},
        {24, 35, 55, 64, 81,  104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
};

auto ToFloat(const cv::Mat &img) {
    Mat channels[3];
    split(img, channels);
    auto b = (float *) malloc(img.rows * img.cols * sizeof(float));
    memcpy(b, channels[0].data, img.rows * img.cols * sizeof(float));
    return b;
}

auto reduceIntensity = [](cv::Mat &image, int div) {
    int nl = image.rows;
    int nc = image.cols * image.channels();

    for (int j = 0; j < nl; j++) {
        uchar *data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++) {
            data[i] = floor(data[i] / (256 / div) * (256 / div));
        }
    }
};

void colorReduce(cv::Mat &image, int div = 64) {
    int level = 256;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            cv::Mat out = image.clone();
            reduceIntensity(out, level);
            plt::subplot2grid(2, 4, i, j);
            plt::imshow(out);
            level /= 2;
        }
    }
    plt::show();
}

void blur(cv::Mat &image, int ksize = 3) {
    cv::Mat out;


    cv::blur(image, out, cv::Size(3, 3));
    plt::subplot2grid(1, 3, 0, 0);
    plt::imshow(out);


    cv::blur(image, out, cv::Size(3, 3));
    plt::subplot2grid(1, 3, 0, 1);
    plt::imshow(out);


    cv::blur(image, out, cv::Size(10, 10));
    plt::subplot2grid(1, 3, 0, 2);
    plt::imshow(out);

    plt::show();
}

void rotate(cv::Mat &image) {
    cv::Mat out;

    cv::rotate(image, out, cv::ROTATE_90_CLOCKWISE);
    plt::subplot2grid(1, 2, 0, 0);
    plt::imshow(out);

    cv::rotate(image, out, cv::ROTATE_90_COUNTERCLOCKWISE);
    plt::subplot2grid(1, 2, 0, 1);
    plt::imshow(out);

    plt::show();
}

void averageBlock(cv::Mat &image) {
    vector<int> stride = {3, 5, 7};

    vector<cv::Mat> plot;

    for (int i = 0; i < stride.size(); i++) {
        cv::Mat out = image.clone();
        for (int j = 0; j < out.rows / stride[i] * stride[i]; j += stride[i]) {
            for (int k = 0; k < out.cols / stride[i] * stride[i]; k += stride[i]) {
                cv::Rect roi(k, j, stride[i], stride[i]);
                cv::Mat block = out(roi);
                cv::Scalar mean = cv::mean(block);
                cv::rectangle(out, roi, mean, -1);
            }
        }
        plot.push_back(out);
    }

    for (int i = 0; i < plot.size(); i++) {
        plt::subplot2grid(1, 3, 0, i);
        plt::imshow(plot[i]);
    }

    plt::show();
}

auto subdivide = [](const cv::Mat &img, const int rowDivisor, const int colDivisor, vector<cv::Mat> &blocks) {

    /* Checking if the image was passed correctly */
    if (!img.data || img.empty())
        std::cerr << "Problem Loading Image" << std::endl;

    /* Cloning the image to another for visualization later, if you do not want to visualize the result just comment every line related to visualization */
    cv::Mat maskImg = img.clone();
    /* Checking if the clone image was cloned correctly */
    if (!maskImg.data || maskImg.empty())
        std::cerr << "Problem Loading Image" << std::endl;

    for (int y = 0; y < img.rows; y += rowDivisor) {
        for (int x = 0; x < img.cols; x += colDivisor) {

            if (x + colDivisor > img.cols || y + rowDivisor > img.rows)
                continue;

            cv::Rect rect(x, y, colDivisor, rowDivisor);
            cv::Mat block = maskImg(rect);
            blocks.push_back(block);
        }
    }
};

auto divideBlocks = [](const cv::Mat &image, int blockSize) -> vector<cv::Mat> {

    std::vector<cv::Mat> blocks;

    int cols = image.cols;
    int rows = image.rows;

    for (int y = 0; y < image.rows; y += blockSize) {
        for (int x = 0; x < image.cols; x += blockSize) {

            int roi_width = std::min(blockSize, cols - x);
            int roi_height = std::min(blockSize, rows - y);

            cv::Rect rect(x, y, roi_width, roi_height);
            cv::Mat block = image(rect);
            blocks.push_back(block);
        }
    }

    return blocks;
};

auto mapValue = [](float value, float min, float max, float newMin, float newMax) -> float {
    return (value - min) * (newMax - newMin) / (max - min) + newMin;
};

auto toMat = [](vector<vector<float>> matrix, int N, int M) -> cv::Mat {

    cv::Mat mat(N, M, CV_32FC1);

    for (int i = 0; i < N; ++i) {
        float *row_ptr = mat.ptr<float>(i);
        for (int j = 0; j < M; ++j) {
            row_ptr[j] = matrix[i][j];
        }
    }

    return mat;
};

auto vecToMat = [](const std::vector<cv::Mat> &blocks, int N, int M, int K, int T) -> cv::Mat {
    cv::Mat mat(N, M, blocks[0].type(), cv::Scalar(0));

    int blockIndex = 0;

    for (int i = 0; i < N; i += K) {
        for (int j = 0; j < M; j += T) {
            if (i + K > N || j + T > M) {
                continue;
            }

            int index_i = mapValue(i, 0, N, 0, N / K - 1); // Adjusted index mapping
            int index_j = mapValue(j, 0, M, 0, M / T - 1); // Adjusted index mapping

            if (index_i >= 0 && index_i < N / K && index_j >= 0 && index_j < M / T) {
                cv::Rect roi(j, i, T, K);
                cv::Mat block = mat(roi);
                blocks[blockIndex].copyTo(block);
                blockIndex++;
            }
        }
    }

    return mat;
};

auto toFloat = [](cv::Mat &img) -> vector<vector<float>> {

    if (img.type() == CV_32FC1) {
        std::cerr << "Image is not float" << std::endl;
        return {};
    }

    int rows = img.rows;
    int cols = img.cols;


    vector<vector<float>> data(rows, vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        const float *row_ptr = img.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            data[i][j] = row_ptr[j];
        }
    }

    return data;
};

auto inverseQuantization = [](cv::Mat &img, int offset) -> cv::Mat {
    if (img.type() != CV_32FC1) {
        std::cerr << "Image is not float" << std::endl;
    }

    int rows = img.rows;
    int cols = img.cols;

    cv::Mat quantized(rows, cols, CV_32FC1);

    for (int i = 0; i < rows; ++i) {
        const float *row_ptr = img.ptr<float>(i);
        float *row_quantized_ptr = quantized.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            row_quantized_ptr[j] = row_ptr[j] * quantizationMatrix[i][j] + offset;
        }
    }

    return quantized;
};

auto quantization = [](cv::Mat &img) -> cv::Mat {
    if (img.type() != CV_32FC1) {
        std::cerr << "Image is not float" << std::endl;
    }

    int rows = img.rows;
    int cols = img.cols;

    cv::Mat quantized(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        const float *row_ptr = img.ptr<float>(i);
        float *row_quantized_ptr = quantized.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            row_quantized_ptr[j] = floor(row_ptr[j] / quantizationMatrix[i][j])* quantizationMatrix[i][j];
        }
    }

    return quantized;
};

void dct(vector<vector<float>> &DCTMatrix, vector<vector<float>> Matrix, int N, int M) {

    int i, j, u, v;
    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
            DCTMatrix[u][v] = 0;
            for (i = 0; i < N; i++) {
                for (j = 0; j < M; j++) {
                    DCTMatrix[u][v] += Matrix[i][j] * cos(M_PI / ((float) N) * (i + 1. / 2.) * u) *
                                       cos(M_PI / ((float) M) * (j + 1. / 2.) * v);
                }
            }
        }
    }
}

void idct(vector<vector<float>> &Matrix, vector<vector<float>> DCTMatrix, int N, int M) {

    int i, j, u, v;
    for (u = 0; u < N; ++u) {
        for (v = 0; v < M; ++v) {
            Matrix[u][v] = 1 / 4. * DCTMatrix[0][0];
            for (i = 1; i < N; i++) {
                Matrix[u][v] += 1 / 2. * DCTMatrix[i][0];
            }
            for (j = 1; j < M; j++) {
                Matrix[u][v] += 1 / 2. * DCTMatrix[0][j];
            }

            for (i = 1; i < N; i++) {
                for (j = 1; j < M; j++) {
                    Matrix[u][v] += DCTMatrix[i][j] * cos(M_PI / ((float) N) * (u + 1. / 2.) * i) *
                                    cos(M_PI / ((float) M) * (v + 1. / 2.) * j);
                }
            }
            Matrix[u][v] *= 2. / ((float) N) * 2. / ((float) M);
        }
    }
}

cv::Mat dft(cv::Mat &image) {

    cv::Mat planes[] = {cv::Mat_<float>(image), cv::Mat::zeros(image.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);

    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];

    magI += cv::Scalar::all(1);
    cv::log(magI, magI);

    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // Rearrange the quadrants of the magnitude image
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

    return magI;
}

void idft(cv::Mat &image) {
    cv::dft(image, image, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
}

auto quantize = [](cv::Mat &block, int quality) {
    for (int i = 0; i < block.rows; i++) {
        for (int j = 0; j < block.cols; j++) {
            float quantization_value = quantizationMatrix[i][j] * quality;
            block.at<float>(i, j) = std::round(block.at<float>(i, j) / quantization_value) * quantization_value;
        }
    }
};

void jpeg(cv::Mat &image) {

    int quality = 50; // Quality factor

    // Divide the image into non-overlapping 8x8 blocks
    int block_size = 8;
    int rows = image.rows;
    int cols = image.cols;

    // Create an empty image to store the reconstructed image
    cv::Mat reconstructed_image = cv::Mat::zeros(rows, cols, CV_8UC1);

    for (int x = 0; x < cols; x += block_size) {
        for (int y = 0; y < rows; y += block_size) {

            int roi_width = std::min(block_size, cols - x);
            int roi_height = std::min(block_size, rows - y);

            // Extract the 8x8 block
            cv::Mat block = image(cv::Rect(x, y, roi_width, roi_height));

            // Convert block to floating point type for DCT
            cv::Mat block_float;
            block.convertTo(block_float, CV_32FC2);

            cv::dft(block_float, block_float);

            // Quantize the coefficients
            quantize(block_float, quality);

            cv::idft(block_float, block_float, cv::DFT_SCALE);

            // Optionally, preserve the 8 largest coefficients
            for (int i = 0; i < block_float.rows; i++) {
                for (int j = 0; j < block_float.cols; j++) {
                    if (i + j >= 8) {
                        block_float.at<float>(i, j) = 0;
                    }
                }
            }

            // Convert the block back to 8-bit for visualization
            cv::Mat block_8bit;
            block_float.convertTo(block_8bit, CV_8U);

            // Place the block back into the reconstructed image
            cv::Rect roi(x, y, roi_width, roi_height);
            block_8bit.copyTo(reconstructed_image(roi));

        }
    }


    // Show the original and reconstructed images
    cv::imshow("Original Image", image);
    cv::imshow("Reconstructed Image", reconstructed_image);
    cv::waitKey(0);

}

// Function to get quantization value for a given position in the quantization matrix
int getQuantizationValue(int i, int j) {
    const int standardQuantizationMatrix[8][8] = {
            {16, 11, 10, 16, 24,  40,  51,  61},
            {12, 12, 14, 19, 26,  58,  60,  55},
            {14, 13, 16, 24, 40,  57,  69,  56},
            {14, 17, 22, 29, 51,  87,  80,  62},
            {18, 22, 37, 56, 68,  109, 103, 77},
            {24, 35, 55, 64, 81,  104, 113, 92},
            {49, 64, 78, 87, 103, 121, 120, 101},
            {72, 92, 95, 98, 112, 100, 103, 99}
    };

    return standardQuantizationMatrix[i % 8][j % 8];
}

// Function to get quantization matrix based on quality factor
cv::Mat getQuantizationMatrix(int quality, const cv::Size &size) {
    cv::Mat quantizationMatrix(size, CV_32F);

    for (int i = 0; i < size.height; i++) {
        for (int j = 0; j < size.width; j++) {
            quantizationMatrix.at<float>(i, j) = static_cast<float>(getQuantizationValue(i, j));
        }
    }

    return quantizationMatrix;
}

cv::Mat convertTo32FC1(const cv::Mat &channel) {
    cv::Mat converted;
    channel.convertTo(converted, CV_32FC1);
    return converted;
}

// Function to combine blocks into a single image
cv::Mat combineBlocks(const std::vector<cv::Mat> &blocks, int image_width, int image_height) {
    cv::Mat image(image_height, image_width, CV_32FC1, cv::Scalar(0));

    int block_size = 8;
    int block_index = 0;

    for (int y = 0; y < image_height; y += block_size) {
        for (int x = 0; x < image_width; x += block_size) {
            cv::Rect roi(x, y, block_size, block_size);
            blocks[block_index].copyTo(image(roi));
            block_index++;
        }
    }

    return image;
}

cv::Mat jpegCompression(const cv::Mat &channel, int quality) {

    cv::Mat compressed, dequantized;

    cv::dft(channel, compressed);

    // Quantize the DCT coefficients
    auto quantiztedMatirx = quantization(compressed);

    dequantized = inverseQuantization(quantiztedMatirx, 128);

    cv::idft(compressed, dequantized, cv::DFT_SCALE);

    return dequantized;
}

void jpegRBG(const cv::Mat &image) {

    int qualityY = 10;
    int qualityCb = 8;
    int qualityCr = 8;

    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    cv::Mat ycbcrImage;
    cv::cvtColor(floatImage, ycbcrImage, cv::COLOR_BGR2YCrCb);

    cv::imshow("YCbCr Image", ycbcrImage);

    vector<cv::Mat> channels;
    cv::split(ycbcrImage, channels);

    auto yBlocks = divideBlocks(channels[0], 8);
    auto cbBlocks = divideBlocks(channels[1], 8);
    auto crBlocks = divideBlocks(channels[2], 8);

    // Perform JPEG compression on each channel
    for (cv::Mat &yBlock: yBlocks) {
        yBlock = jpegCompression(yBlock, qualityY);
    }
    for (cv::Mat &cbBlock: cbBlocks) {
        cbBlock = jpegCompression(cbBlock, qualityCb);
    }
    for (cv::Mat &crBlock: crBlocks) {
        crBlock = jpegCompression(crBlock, qualityCr);
    }

    // Combine the compressed blocks
    cv::Mat compressedY = combineBlocks(yBlocks, channels[0].cols, channels[0].rows);
    cv::Mat compressedCb = combineBlocks(cbBlocks, channels[1].cols, channels[1].rows);
    cv::Mat compressedCr = combineBlocks(crBlocks, channels[2].cols, channels[2].rows);

    imshow("Compressed Y", compressedY);
    imshow("Compressed Cb", compressedCb);
    imshow("Compressed Cr", compressedCr);

    // Merge the compressed YCbCr channels
    channels[0] = compressedY;
    channels[1] = compressedCb;
    channels[2] = compressedCr;

    cv::merge(channels, ycbcrImage);


    cv::Mat rgbImage;

    cv::cvtColor(ycbcrImage, rgbImage, cv::COLOR_YCrCb2BGR);

    // Show the original and reconstructed images
    cv::imshow("Original RGB Image", image);
    cv::imshow("Reconstructed RGB Image", rgbImage);
    cv::waitKey(0);

}

void hist(cv::Mat &image){
    cv::Mat hist_original;
    int histSize = 256;
    float range[] = {0, 256};
    const float *histRange = {range};
    cv::calcHist(&image, 1, 0, cv::Mat(), hist_original, 1, &histSize, &histRange);


}

int main(int argc, char *argv[]) {
    // Load input image (colored, 3-channel, BGR)
    cv::Mat input = cv::imread(argv[1]);


    std::cout << "Rows" << input.rows << std::endl;
    std::cout << "Cols" << input.cols << std::endl;

    if (input.empty()) {
        std::cout << "!!! Failed imread()" << std::endl;
        return -1;
    }

    //colorReduce(input);
    //blur(input);
    //rotate(input);
    //averageBlock(input);
    //jpeg(input);
    //jpegRBG(input);


    return 0;
}
