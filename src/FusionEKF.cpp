#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  /**
   * TODO: Finish initializing the FusionEKF.
   * TODO: Set the process and measurement noises
   */
  
  // create a 4D state vector
  ekf_.x_ = VectorXd(4);

  // the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
  
  // measurement matrix for Laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  // create the state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  
  // create the noise covariance matrix
  ekf_.Q_ = MatrixXd(4, 4);

  // Noise
  noise_ax = 9;
  noise_ay = 9;


}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    /**
     * TODO: Initialize the state ekf_.x_ with the first measurement.
     * TODO: Create the covariance matrix.
     * You'll need to convert radar from polar to cartesian coordinates.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // TODO: Convert radar from polar to cartesian coordinates 
      //         and initialize state.
      
      float ro = measurement_pack.raw_measurements_[0]; 
      float theta = measurement_pack.raw_measurements_[1];  
      float ro_dot = measurement_pack.raw_measurements_[2]; 

      ekf_.x_ << ro * cos(theta), 
                 ro * sin(theta) *(-1),
                 ro_dot * cos(theta), 
                 ro_dot * sin(theta) * (-1);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // TODO: Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0], 
                 measurement_pack.raw_measurements_[1], 
                 0, 
                 0;
    }
    
    // initiate state covariance matrix P
    // ref: Lesson 24, Section 13
    ekf_.P_ <<  1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1000, 0,
                0, 0, 0, 1000;
    
    // update timestamp
    previous_timestamp_ = measurement_pack.timestamp_;
    
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /**
   * Prediction
   */

  /**
   * TODO: Update the state transition matrix F according to the new elapsed time.
   * Time is measured in seconds.
   * TODO: Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  
  // dt - expressed in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  // Update the state transition matrix F
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the process noise covariance matrix.
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
              0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
              dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
              0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /**
   * Update
   */

  /**
   * TODO:
   * - Use the sensor type to perform the update step.
   * - Update the state and covariance matrices.
   */
  
  VectorXd z = measurement_pack.raw_measurements_;
  
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
  
    VectorXd z = measurement_pack.raw_measurements_;
  
  	Hj_ =  tools.CalculateJacobian(ekf_.x_); //shape(3, 4)

    // VectorXd z_pred = Hj_ * ekf_.x_;   //shape(3, 1)
    VectorXd z_pred = VectorXd(3);
    float px = ekf_.x_(0);
    float py = ekf_.x_(1);
    float vx = ekf_.x_(2);
    float vy = ekf_.x_(3);

    // pre-compute a set of terms to avoid repeated calculation
    float c2 = sqrt(px*px+py*py);
    if (fabs(c2) < 0.01) {
      // rho_dot = sqrt(vx*vx+vy*vy);
      c2 = 0.01;
    }
    float rho_dot = (px*vx+py*vy)/c2;
    float theta = atan2(py, px);
    
    z_pred << c2, theta, rho_dot;              
    VectorXd y = z - z_pred;          //shape(3, 1)
    
    //Normalization
    float pi = 3.14159265358979323846;
    if (y(1) > pi){
      //cout << "y1 = " << y(1) << endl;
      y(1) = y(1) - (2*pi);
    }
    if (y(1) < (-1)*pi){
      //cout << "y1 = " << y(1)<< endl;
      y(1) = y(1) + (2*pi);
    }
    
    MatrixXd Ht = Hj_.transpose();     //shape(4, 3)
    MatrixXd S = Hj_ * ekf_.P_ * Ht + R_radar_;   //shape (3, 4) x (4, 4) x (4, 3) + (3, 3) = (3, 3)
    MatrixXd Si = S.inverse();       //shape (3, 3)
    MatrixXd PHt = ekf_.P_ * Ht;     //shape (4, 3)
    MatrixXd K = PHt * Si;           //shape (4, 3)

    //new estimate
    ekf_.x_ = ekf_.x_ + (K * y);     //shape (4, 1)
    long x_size = ekf_.x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    ekf_.P_ = (I - K * Hj_) * ekf_.P_; //shape (4, 4)

  } 
  else {
    // TODO: Laser updates
    
    VectorXd z_pred = H_laser_* ekf_.x_;  //shape(2, 1)
    VectorXd y = z - z_pred;              //shape(2, 1)
    MatrixXd Ht = H_laser_.transpose();    //shape(4, 2)
    MatrixXd S = H_laser_ * ekf_.P_ * Ht + R_laser_;   //shape (2, 4) x (4, 4) x (4, 2) + (2, 2) = (2, 2)
    MatrixXd Si = S.inverse();       //shape (2, 2)
    MatrixXd PHt = ekf_.P_ * Ht;     //shape (4, 2)
    MatrixXd K = PHt * Si;           //shape (4, 2)

    //new estimate
    ekf_.x_ = ekf_.x_ + (K * y);     //shape (4, 1)
    long x_size = ekf_.x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    ekf_.P_ = (I - K * H_laser_) * ekf_.P_; //shape (4, 4)
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
