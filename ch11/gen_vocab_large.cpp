#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    // string dataset_dir = argv[1];
    // ifstream fin ( dataset_dir+"/associate.txt" );
    // if ( !fin )
    // {
    //     cout<<"please generate the associate file called associate.txt!"<<endl;
    //     return 1;
    // }

    String directoryPath = "/home/jiaoao/Downloads/TUM数据集/rgb";//图像路径
    vector<String> imagesPath;
    glob(directoryPath, imagesPath);


    cout<<"generating features ... "<<endl;//输出generating features (正在检测ORB特征)...
    vector<Mat> descriptors;//描述子
    Ptr< Feature2D > detector = ORB::create();
    int index = 1;
    for ( String path : imagesPath )
    {
        Mat image = imread(path);
        vector<KeyPoint> keypoints; //关键点
        Mat descriptor;//描述子
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
        cout<<"extracting features from image " << index++ <<endl;//输出extracting features from image(从图像中提取特征)
    }
    cout<<"extract total "<<descriptors.size()*500<<" features."<<endl;
    
    // create vocabulary 
    cout<<"creating vocabulary, please wait ... "<<endl;//输出creating vocabulary, please wait (创建词典，请稍等)...
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "vocab_larger.yml.gz" );//保存词典
    cout<<"done"<<endl;
    
    return 0;
}