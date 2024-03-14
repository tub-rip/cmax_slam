#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace dvs {

class EquirectangularCamera {
public:
    EquirectangularCamera() {}
    EquirectangularCamera(const cv::Size& imageSize, double hfov, double vfov) :
        imageSize(imageSize), center(Eigen::Vector2d((double)imageSize.width / 2.0, (double)imageSize.height / 2.0)) {
        std::pair<double, double> focals = focalFromFOV(imageSize, hfov, vfov);
        fx = focals.first;
        fy = focals.second;
    }

    virtual Eigen::Vector2d projectToImage(const Eigen::Vector3d& P, cv::Matx23f* jacobian) const {
        double x, y, z;
        x = P[0];
        y = P[1];
        z = P[2];

        // From https://github.com/uzh-rpg/rpg_image_reconstruction_from_events/blob/master/matlab/project_EquirectangularProjection.m
        const double phi = std::atan2(x,z);
        const double theta = std::asin(y/std::sqrt(x*x+y*y+z*z));

        const double rho = P.norm();
        const double Ydivrho = y / rho;

        if (jacobian != nullptr)
        {
            const double XdivZ = x / z;
            const double tmp1 = fx /((1 + XdivZ*XdivZ) * z);
            const double tmp2 = -fy/std::sqrt(1 - Ydivrho*Ydivrho);
            const double tmp3 = Ydivrho / (rho*rho);
            (*jacobian)(0,0) = tmp1;
            (*jacobian)(0,1) = 0;
            (*jacobian)(0,2) = -tmp1 * XdivZ;
            (*jacobian)(1,0) = tmp2 * tmp3 * x;
            (*jacobian)(1,1) = tmp2 * (tmp3*y - 1/rho);
            (*jacobian)(1,2) = tmp2 * tmp3 * z;
        }
        return center + Eigen::Vector2d(phi * fx, theta * fy);
    }

    virtual Eigen::Vector3d liftToUnitSphere(const Eigen::Vector2d& p) const {
        // @TODO
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }

    cv::Size getImageSize() const
    {
        return imageSize;
    }

    void getScalingFactors(double &f_x, double &f_y) const
    {
        f_x = fx;
        f_y = fy;
    }

private:
    static std::pair<double, double> focalFromFOV(const cv::Size& imageSize, double hfov, double vfov) {
        return std::pair<double, double>(double((imageSize.width / hfov) * 180.0 / CV_PI),
                                         double((imageSize.height / vfov) * 180.0 / CV_PI));
    }

    static double normalizeAngle(double angle)
    {
        double newAngle = angle;
        while (newAngle <= -CV_PI) newAngle += 2*CV_PI;
        while (newAngle > CV_PI) newAngle -= 2*CV_PI;
        return newAngle;
    }

    cv::Size imageSize;
    Eigen::Vector2d center;
    double fx;
    double fy;
};

}
