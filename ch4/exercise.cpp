#include <iostream>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

int main() {
    // 定义四元数和平移向量
    Eigen::Quaterniond q(0.35, 0.2, 0.3, 0.1);
    Eigen::Vector3d t(0.3, 0.1, 0.1);

    // 将四元数转换为SO3（李群）
    Sophus::SO3d so3(q);

    // 打印旋转矩阵
    std::cout << "旋转矩阵 R:" << std::endl << so3.matrix() << std::endl;

    // 将SO3（李群）转换为李代数
    Sophus::Vector3d so3_vec = so3.log();

    // 打印李代数
    std::cout << "李代数 so3:" << std::endl << so3_vec.transpose() << std::endl;

    // 构造SE3（李群）
    Sophus::SE3d se3(so3, t);

    // 打印变换矩阵
    std::cout << "变换矩阵 T:" << std::endl << se3.matrix() << std::endl;

    // 将SE3（李群）转换为李代数
    Sophus::Vector6d se3_vec = se3.log();

    // 打印李代数
    std::cout << "李代数 se3:" << std::endl << se3_vec.transpose() << std::endl;

    return 0;
}
