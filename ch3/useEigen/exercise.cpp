#include <iostream>

using namespace std;

#include <ctime>
// Eigen 核心部分
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc, char **argv) {

    Matrix3d  mr3;

    mr3 << -0.381326,0.882821,-0.274256,
    0.159702,-0.229302,-0.960165,
    -0.910541,-0.409935,-0.0535494;
     // 输出矩阵的行列式(验证旋转矩阵)
    cout << "determinant: " << mr3.determinant() << endl;
    // 逐元素输出
    // for (int i = 0; i < 3; i++){
    //     for (int j = 0; j < 3; j++){
    //         cout << mr3(i,j) << ' ';
    //     }
    //     cout << endl;
    // }
    cout << mr3 << endl;
    // 输出矩阵的迹
    cout << "trace: " << mr3.trace() << endl;
    // 输出矩阵的转置
    cout << "transpose: " << mr3.transpose() << endl;
    // 输出矩阵的逆
    cout << "inverse: " << mr3.inverse() << endl;
    // 转换成旋转向量的形式

   
    
    AngleAxisd rv(mr3);
    // 输出旋转向量的轴和角度
    std::cout << "Axis: " << rv.axis().transpose() << std::endl;
    std::cout << "Angle (in radians): " << rv.angle() << std::endl;

    // 转换成欧拉角
    Vector3d ea = mr3.eulerAngles(2,1,0);
    cout << "yaw pitch roll:" << ea.transpose() << endl;

    // 转换成四元数
    Quaterniond qv = Quaterniond(rv);
    cout << "quaternion from rv: " << qv.coeffs().transpose() << endl;

    Quaterniond qv2 = Quaterniond(mr3);
    cout << "quaternion from rm: " << qv2.coeffs().transpose() << endl;


    return 0;
}

