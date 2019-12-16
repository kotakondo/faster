#include "solverGurobi.hpp"
#include "solverGurobi_utils.hpp"  //This must go here, and not in solverGurobi.hpp
#include <chrono>
#include <unistd.h>
#include <ros/package.h>

#define WHOLE_TRAJ 0
#define RESCUE_PATH 1

mycallback::mycallback()
{
  should_terminate_ = false;
}

void mycallback::callback()
{  // This function is called periodically along the optimization process.
  // It is called several times more after terminating the program
  if (should_terminate_ == true)
  {
    // std::cout << "Aborting the execution of the current optimization" << std::endl;
    GRBCallback::abort();  // It seems that this function only does effect when inside the function callback() of this
                           // class
    // terminated_ = true;
  }
}

/*void mycallback::abortar()
{
  GRBCallback::abort();
}*/

void SolverGurobi::StopExecution()
{
  // cb_.abortar();
  // cb_.terminated_ = false;
  cb_.should_terminate_ = true;
  std::cout << "Activated flag to stop execution" << std::endl;
}

void SolverGurobi::ResetToNormalState()
{
  cb_.should_terminate_ = false;
}

SolverGurobi::SolverGurobi()
{
  std::cout << "In the Gurobi Constructor\n";

  v_max_ = 5;
  a_max_ = 3;
  j_max_ = 5;
  // N_ = 10;  // Segments: 0,1,...,N_-1

  // Model
  /*  env = new GRBEnv();
    m = GRBModel(*env);*/
  m.set(GRB_StringAttr_ModelName, "planning");

  cb_ = mycallback();

  m.setCallback(&cb_);  // The callback will be called periodically along the optimization

  /*  setDC(0.01);  LO HAGO EN CVX
  set_max(max_values);  LO HAGO EN CVX

    setX0(x0);   LO HAGO EN CVX
    setXf(xf);   LO HAGO EN CVX


    setMaxConstraints();  // Only needed to set once///////////////////////


 // setPolytopes();  LO HAGO EN CVX

    genNewTraj();



  // Solve*/

  /*catch (GRBException e)
  {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  }*/
  /*catch (...)
  {
    cout << "Exception during optimization" << endl;
  }*/

  // delete env;
}

void SolverGurobi::setN(int N)
{
  N_ = N;
}

void SolverGurobi::setMode(int mode)
{
  mode_ = mode;
}

void SolverGurobi::createVars()
{
  std::vector<std::string> coeff = { "ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz", "dx", "dy", "dz" };

  // Variables: Coefficients of the polynomials
  for (int t = 0; t < N_; t++)
  {
    std::vector<GRBVar> row_t;
    for (int i = 0; i < 12; i++)
    {
      row_t.push_back(m.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_CONTINUOUS, coeff[i] + std::to_string(t)));
    }
    x.push_back(row_t);
  }
}

void SolverGurobi::setObjective()  // I need to set it every time, because the objective depends on the xFinal
{
  GRBQuadExpr control_cost = 0;
  /*  GRBQuadExpr final_state_cost = 0;
    // GRBQuadExpr distance_to_JPS_cost = 0;

    std::vector<GRBLinExpr> xFinal = {
      getPos(N_ - 1, 0, 0),   getPos(N_ - 1, 0, 1),   getPos(N_ - 1, 0, 2),   //////////////////////////////////
      getVel(N_ - 1, 0, 0),   getVel(N_ - 1, 0, 1),   getVel(N_ - 1, 0, 2),   /////////////////////////////
      getAccel(N_ - 1, 0, 0), getAccel(N_ - 1, 0, 1), getAccel(N_ - 1, 0, 2)  /////////////////////////////
    };
    std::vector<double> xf(std::begin(xf_), std::end(xf_));  // Transform xf_ into a std vector
    final_state_cost = q_ * GetNorm2(xFinal - xf);*/

  /*  for (int t = 0; t < samples_penalize_.size(); t++)  // This loop is not executed when computing the rescue path
    {
      std::vector<GRBLinExpr> sample_i = { samples_penalize_[t][0], samples_penalize_[t][1], samples_penalize_[t][2] };
      // printf("in loop %d\n", t);

      double tau = (t == samples_penalize_.size() - 1) ? dt_ : 0;  // If the last interval -->  at the end of it
      double interval = (t == samples_penalize_.size() - 1) ? t - 1 : t;

      std::vector<GRBLinExpr> pos_i = { getPos(interval, tau, 0), getPos(interval, tau, 1), getPos(interval, tau, 2) };

      distance_to_JPS_cost = distance_to_JPS_cost + GetNorm2(sample_i - pos_i);
    }*/

  for (int t = 0; t < N_; t++)
  {
    std::vector<GRBLinExpr> ut = { getJerk(t, 0, 0), getJerk(t, 0, 1), getJerk(t, 0, 2) };
    control_cost = control_cost + GetNorm2(ut);
  }
  // m.setObjective(control_cost + final_state_cost + distance_to_JPS_cost, GRB_MINIMIZE);
  m.setObjective(control_cost, GRB_MINIMIZE);
}

void SolverGurobi::fillXandU()
{
  // std::cout << "in fillXandU" << std::endl;
  double t = 0;
  int interval = 0;
  //#pragma omp parallel for
  //  {
  // std::cout << "voy a fill" << std::endl;
  // std::cout << "U_temp_.rows()=" << U_temp_.rows() << std::endl;
  for (int i = 0; i < X_temp_.size(); i++)
  {
    // std::cout << "row=" << i << std::endl;
    t = t + DC;
    // std::cout << "t=" << t << std::endl;
    if (t > dt_ * (interval + 1))
    {
      interval = std::min(interval + 1, N_ - 1);

      // std::cout << "*****Interval=" << interval << std::endl;
    }

    // std::cout << "t_rel=" << t - interval * dt_ << std::endl;
    double jerkx = getJerk(interval, t - interval * dt_, 0).getValue();
    double jerky = getJerk(interval, t - interval * dt_, 1).getValue();
    double jerkz = getJerk(interval, t - interval * dt_, 2).getValue();

    /*    Eigen::Matrix<double, 1, 3> input;
        input << jerkx, jerky, jerkz;

        U_temp_.row(i) = input;*/

    double posx = getPos(interval, t - interval * dt_, 0).getValue();
    double posy = getPos(interval, t - interval * dt_, 1).getValue();
    double posz = getPos(interval, t - interval * dt_, 2).getValue();

    double velx = getVel(interval, t - interval * dt_, 0).getValue();
    double vely = getVel(interval, t - interval * dt_, 1).getValue();
    double velz = getVel(interval, t - interval * dt_, 2).getValue();

    double accelx = getAccel(interval, t - interval * dt_, 0).getValue();
    double accely = getAccel(interval, t - interval * dt_, 1).getValue();
    double accelz = getAccel(interval, t - interval * dt_, 2).getValue();

    state state_i;
    state_i.setPos(posx, posy, posz);
    state_i.setVel(velx, vely, velz);
    state_i.setAccel(accelx, accely, accelz);
    state_i.setJerk(jerkx, jerky, jerkz);
    X_temp_[i] = (state_i);
  }
  // std::cout << "After the loop" << std::endl;
  // }  // End pragma parallel

  /*  int last = X_temp_.rows() - 1;

    // Force the final velocity and acceleration to be exactly the final conditions
    Eigen::Matrix<double, 1, 9> final_cond;
    final_cond << X_temp_(last, 0), X_temp_(last, 1), X_temp_(last, 2), xf_[3], xf_[4], xf_[5], xf_[6], xf_[7], xf_[8];
    X_temp_.row(X_temp_.rows() - 1) = final_cond;
  */

  // Force the final input to be 0 (I'll keep applying this input if when I arrive to the final state I still
  // haven't planned again).
  // U_temp_.row(U_temp_.rows() - 1) = Eigen::Vector3d::Zero().transpose();

  X_temp_[X_temp_.size() - 1].vel = Eigen::Vector3d::Zero().transpose();
  X_temp_[X_temp_.size() - 1].accel = Eigen::Vector3d::Zero().transpose();
  X_temp_[X_temp_.size() - 1].jerk = Eigen::Vector3d::Zero().transpose();

  // std::cout << "end of fillXandU" << std::endl;
  /*  std::cout << "***********The final states are***********" << std::endl;
    std::cout << X_temp_.row(X_temp_.rows() - 1).transpose() << std::endl;

    std::cout << "***********The final conditions were***********" << std::endl;
    std::cout << xf_[0] << std::endl;
    std::cout << xf_[1] << std::endl;
    std::cout << xf_[2] << std::endl;
    std::cout << xf_[3] << std::endl;
    std::cout << xf_[4] << std::endl;
    std::cout << xf_[5] << std::endl;
    std::cout << xf_[6] << std::endl;
    std::cout << xf_[7] << std::endl;
    std::cout << xf_[8] << std::endl;*/

  /*  std::cout << "***********The states are***********" << std::endl;
    std::cout << X_temp_ << std::endl;
    std::cout << "***********The input is***********" << std::endl;
    std::cout << U_temp_ << std::endl;*/
}

/*void SolverGurobi::setSamplesPenalize(vec_Vecf<3>& samples_penalize)
{
  samples_penalize_.clear();
  samples_penalize_ = samples_penalize;
}*/

void SolverGurobi::setDistances(vec_Vecf<3>& samples,
                                std::vector<double> dist_near_obs)  // Distance values (used for the
                                                                    // rescue path, used in the distance constraint4s)
{
  samples_ = samples;
  dist_near_obs_ = dist_near_obs;
}

void SolverGurobi::setForceFinalConstraint(bool forceFinalConstraint)
{
  forceFinalConstraint_ = forceFinalConstraint;
}

void SolverGurobi::setDistanceConstraints()  // Set the distance constraints
{
  // Remove previous distance constraints
  for (int i = 0; i < distances_cons.size(); i++)
  {
    m.remove(distances_cons[i]);
  }
  distances_cons.clear();

  for (int t = 0; t < samples_.size(); t++)
  {
    std::vector<GRBLinExpr> p = { samples_[t][0], samples_[t][1], samples_[t][2] };
    /*    double tau = 0;
        double interval = t;
        if (t == samples_.size() - 1)  // If the last interval --> constraint at the end of that interval
        {
          tau = dt_;
          interval = t - 1;
        }*/
    /*    GRBLinExpr posx = getPos(interval, tau, 0);
        GRBLinExpr posy = getPos(interval, tau, 1);
        GRBLinExpr posz = getPos(interval, tau, 2);*/
    // std::vector<GRBLinExpr> var = { posx, posy, posz };

    // std::cout << "Going to set Distance Constraints, interval=" << t << std::endl;

    std::vector<GRBLinExpr> cp0 = getCP0(t);  // Control Point 0
    std::vector<GRBLinExpr> cp1 = getCP1(t);  // Control Point 1
    std::vector<GRBLinExpr> cp2 = getCP2(t);  // Control Point 2
    std::vector<GRBLinExpr> cp3 = getCP3(t);  // Control Point 3

    double epsilon = dist_near_obs_[t] * dist_near_obs_[t];
    // printf("Cons with distance=%f ", sqrt(epsilon));
    // std::cout << "For the sample=" << samples_[t].transpose() << std::endl;

    distances_cons.push_back(m.addQConstr(GetNorm2(cp0 - p) <= epsilon));
    /*    distances_cons.push_back(m.addQConstr(GetNorm2(cp1 - p) <= epsilon));
        distances_cons.push_back(m.addQConstr(GetNorm2(cp2 - p) <= epsilon));*/
    distances_cons.push_back(m.addQConstr(GetNorm2(cp3 - p) <= epsilon));

    // std::cout << "Setting Epsilon=" << sqrt(epsilon) << std::endl;
  }
}

void SolverGurobi::setPolytopes(std::vector<LinearConstraint3D> polytopes)
{
  polytopes_ = polytopes;
}

void SolverGurobi::setPolytopesConstraints()
{
  // std::cout << "Setting POLYTOPES=" << polytopes_cons.size() << std::endl;

  // Remove previous polytopes constraints
  for (int i = 0; i < polytopes_cons.size(); i++)
  {
    m.remove(polytopes_cons[i]);
  }
  polytopes_cons.clear();

  // Remove previous polytopes constraints
  for (int i = 0; i < polytope_cons.size(); i++)
  {
    m.remove(polytope_cons[i]);
  }

  polytope_cons.clear();

  // Remove previous at_least_1_pol_cons constraints
  for (int i = 0; i < at_least_1_pol_cons.size(); i++)
  {
    m.remove(at_least_1_pol_cons[i]);
  }

  at_least_1_pol_cons.clear();

  // Remove previous binary variables  (They depend on the number of polytopes--> I can't reuse them)
  for (int i = 0; i < b.size(); i++)
  {
    for (int j = 0; j < b[i].size(); j++)
    {
      m.remove(b[i][j]);
    }
  }
  b.clear();

  if (polytopes_.size() > 0)  // If there are polytope constraints
  {
    if (mode_ == WHOLE_TRAJ)
    {
      // Declare binary variables
      for (int t = 0; t < N_ + 1; t++)
      {
        std::vector<GRBVar> row;
        for (int i = 0; i < polytopes_.size(); i++)  // For all the polytopes
        {
          GRBVar variable =
              m.addVar(-GRB_INFINITY, GRB_INFINITY, 0, GRB_BINARY, "s" + std::to_string(i) + "_" + std::to_string(t));
          row.push_back(variable);
        }
        b.push_back(row);
      }
    }

    // std::cout << "NUMBER OF POLYTOPES=" << polytopes_.size() << std::endl;
    // std::cout << "NUMBER OF FACES of the first polytope=" << polytopes_[0].A().rows() << std::endl;

    // Polytope constraints (if binary_varible==1 --> In that polytope) and at_least_1_pol_cons (at least one polytope)
    // constraints
    for (int t = 0; t < N_; t++)  // Start in t=1 (because t=0 is already fixed with the initial condition)
    {
      // std::cout << "*********************t= " << t << std::endl;

      if (mode_ == WHOLE_TRAJ)
      {
        GRBLinExpr sum = 0;
        for (int col = 0; col < b[0].size(); col++)
        {
          sum = sum + b[t][col];
        }
        at_least_1_pol_cons.push_back(m.addConstr(sum == 1, "At_least_1_pol_t_" + std::to_string(t)));  // at least in
                                                                                                        // one polytope
      }
      std::vector<GRBLinExpr> cp0 = getCP0(t);  // Control Point 0
      std::vector<GRBLinExpr> cp1 = getCP1(t);  // Control Point 1
      std::vector<GRBLinExpr> cp2 = getCP2(t);  // Control Point 2
      std::vector<GRBLinExpr> cp3 = getCP3(t);  // Control Point 3

      for (int n_poly = 0; n_poly < polytopes_.size(); n_poly++)  // Loop over the number of polytopes
      {
        // Constraint A1x<=b1
        Eigen::MatrixXd A1 = polytopes_[n_poly].A();
        auto bb = polytopes_[n_poly].b();

        // std::cout << "Using A1=" << A1 << std::endl;

        std::vector<std::vector<double>> A1std = eigenMatrix2std(A1);

        // std::cout << "Before multip for n_poly=" << n_poly << std::endl;
        std::vector<GRBLinExpr> Acp0 = MatrixMultiply(A1std, cp0);  // A times control point 0
        std::vector<GRBLinExpr> Acp1 = MatrixMultiply(A1std, cp1);  // A times control point 1
        std::vector<GRBLinExpr> Acp2 = MatrixMultiply(A1std, cp2);  // A times control point 2
        std::vector<GRBLinExpr> Acp3 = MatrixMultiply(A1std, cp3);  // A times control point 3

        for (int i = 0; i < bb.rows(); i++)
        {
          std::string name0 =
              "Poly" + std::to_string(n_poly) + "_face" + std::to_string(i) + "_t" + std::to_string(t) + "_cp0";
          std::string name1 =
              "Poly" + std::to_string(n_poly) + "_face" + std::to_string(i) + "_t" + std::to_string(t) + "_cp1";
          std::string name2 =
              "Poly" + std::to_string(n_poly) + "_face" + std::to_string(i) + "_t" + std::to_string(t) + "_cp2";
          std::string name3 =
              "Poly" + std::to_string(n_poly) + "_face" + std::to_string(i) + "_t" + std::to_string(t) + "_cp3";
          // std::cout << "Plane=" << i << "out of" << bb.rows() - 1 << std::endl;
          if (mode_ == WHOLE_TRAJ)
          {  // If b[t,0]==1, all the control points are in that polytope

            polytopes_cons.push_back(m.addGenConstrIndicator(b[t][n_poly], 1, Acp0[i], GRB_LESS_EQUAL, bb[i], name0));
            polytopes_cons.push_back(m.addGenConstrIndicator(b[t][n_poly], 1, Acp1[i], GRB_LESS_EQUAL, bb[i], name1));
            polytopes_cons.push_back(m.addGenConstrIndicator(b[t][n_poly], 1, Acp2[i], GRB_LESS_EQUAL, bb[i], name2));
            polytopes_cons.push_back(m.addGenConstrIndicator(b[t][n_poly], 1, Acp3[i], GRB_LESS_EQUAL, bb[i], name3));
          }
          if (mode_ == RESCUE_PATH)  // There will be only one polytope --> all the control points in that polytope
          {
            // std::cout << "Setting POLYTOPES=3" << std::endl;
            polytope_cons.push_back(m.addConstr(Acp0[i] <= bb[i], name0));
            // std::cout << "Setting POLYTOPES=3.5" << std::endl;
            polytope_cons.push_back(m.addConstr(Acp1[i] <= bb[i], name1));
            // std::cout << "Setting POLYTOPES=3.6" << std::endl;
            polytope_cons.push_back(m.addConstr(Acp2[i] <= bb[i], name2));
            polytope_cons.push_back(m.addConstr(Acp3[i] <= bb[i], name3));
            // std::cout << "Setting POLYTOPES=4" << std::endl;
          }
        }
      }
    }
  }
  // std::cout << "Done POLYTOPES=" << std::endl;
}

int SolverGurobi::getN()
{
  return N_;
}

void SolverGurobi::setDC(double dc)
{
  DC = dc;
}

/*Eigen::MatrixXd SolverGurobi::getX()
{
  return X_temp_;
}*/

/*Eigen::MatrixXd SolverGurobi::getU()
{
  return U_temp_;
}*/

void SolverGurobi::setX0(state& data)
{
  Eigen::Vector3d pos = data.pos;
  Eigen::Vector3d vel = data.vel;
  Eigen::Vector3d accel = data.accel;

  x0_[0] = data.pos.x();
  x0_[1] = data.pos.y();
  x0_[2] = data.pos.z();
  x0_[3] = data.vel.x();
  x0_[4] = data.vel.y();
  x0_[5] = data.vel.z();
  x0_[6] = data.accel.x();
  x0_[7] = data.accel.y();
  x0_[8] = data.accel.z();
}

void SolverGurobi::setXf(state& data)
{
  Eigen::Vector3d pos = data.pos;
  Eigen::Vector3d vel = data.vel;
  Eigen::Vector3d accel = data.accel;

  xf_[0] = data.pos.x();
  xf_[1] = data.pos.y();
  xf_[2] = data.pos.z();
  xf_[3] = data.vel.x();
  xf_[4] = data.vel.y();
  xf_[5] = data.vel.z();
  xf_[6] = data.accel.x();
  xf_[7] = data.accel.y();
  xf_[8] = data.accel.z();
}

void SolverGurobi::setConstraintsXf()
{
  // Remove previous final constraints
  for (int i = 0; i < final_cons.size(); i++)
  {
    m.remove(final_cons[i]);
  }

  final_cons.clear();

  // std::cout << "Setting Final Constraints!!" << std::endl;
  // Constraint xT==x_final
  for (int i = 0; i < 3; i++)
  {
    /*    final_cons.push_back(m.addConstr(getPos(N_ - 1, dt_, i) - xf_[i] <= 0.1));   // Final position
        final_cons.push_back(m.addConstr(getPos(N_ - 1, dt_, i) - xf_[i] >= -0.1));  // Final position

        final_cons.push_back(m.addConstr(getVel(N_ - 1, dt_, i) - xf_[i + 3] <= 0.05));   // Final velocity
        final_cons.push_back(m.addConstr(getVel(N_ - 1, dt_, i) - xf_[i + 3] >= -0.05));  // Final velocity

        final_cons.push_back(m.addConstr(getAccel(N_ - 1, dt_, i) - xf_[i + 6] <= 0.05));   // Final acceleration
        final_cons.push_back(m.addConstr(getAccel(N_ - 1, dt_, i) - xf_[i + 6] >= -0.05));  // Final acceleration*/
    if (forceFinalConstraint_ == true)
    {
      // std::cout << "*********FORCING FINAL CONSTRAINT******" << std::endl;
      // std::cout << xf_[i] << std::endl;
      final_cons.push_back(m.addConstr(getPos(N_ - 1, dt_, i) - xf_[i] == 0,
                                       "FinalPosAxis_" + std::to_string(i)));  // Final position
    }
    final_cons.push_back(m.addConstr(getVel(N_ - 1, dt_, i) - xf_[i + 3] == 0,
                                     "FinalVelAxis_" + std::to_string(i)));  // Final velocity
    final_cons.push_back(m.addConstr(getAccel(N_ - 1, dt_, i) - xf_[i + 6] == 0,
                                     "FinalAccel_" + std::to_string(i)));  // Final acceleration
  }
}

void SolverGurobi::setConstraintsX0()
{
  // Remove previous initial constraints
  for (int i = 0; i < init_cons.size(); i++)
  {
    m.remove(init_cons[i]);
  }

  init_cons.clear();
  // Constraint x0==x_initial
  for (int i = 0; i < 3; i++)
  {
    // std::cout << "x0_[i]= " << x0_[i] << std::endl;
    // std::cout << "Position" << std::endl;
    init_cons.push_back(m.addConstr(getPos(0, 0, i) == x0_[i],
                                    "InitialPosAxis_" + std::to_string(i)));  // Initial position
                                                                              // std::cout << "Velocity" << std::endl;
    init_cons.push_back(m.addConstr(getVel(0, 0, i) == x0_[i + 3],
                                    "InitialVelAxis_" + std::to_string(i)));  // Initial velocity
    // std::cout << "Accel" << std::endl;
    init_cons.push_back(m.addConstr(getAccel(0, 0, i) == x0_[i + 6],
                                    "InitialAccelAxis_" + std::to_string(i)));  // Initial acceleration}
  }
}

void SolverGurobi::resetXandU()
{
  int size = (int)(N_)*dt_ / DC;
  size = (size < 2) ? 2 : size;  // force size to be at least 2
  std::vector<state> tmp(size);
  X_temp_ = tmp;  // Eigen::MatrixXd::Zero(size, 12);
}

/*double SolverGurobi::getCost()
{
  double term_cost = 0;

  for (int i = 0; i < 3 * INPUT_ORDER; i++)
  {
    term_cost = term_cost + q_ * pow(x_[N_][i] - xf_[i], 2);
  }
        printf("real total cost=%f\n", jerk_get_cost());
        printf("real jerk cost=%f\n", jerk_get_cost() - term_cost);
  cost_ = (jerk_get_cost() - term_cost) * N_ * dt_ / (j_max_ * j_max_);

  return cost_;
}*/

void SolverGurobi::setMaxConstraints()
{
  // Constraint v<=vmax, a<=amax, u<=umax
  for (int t = 0; t < N_; t++)
  {
    for (int i = 0; i < 3; i++)
    {
      m.addConstr(getVel(t, 0, i) <= v_max_, "MaxVel_t" + std::to_string(t) + "_axis_" + std::to_string(i));
      m.addConstr(getVel(t, 0, i) >= -v_max_, "MinVel_t" + std::to_string(t) + "_axis_" + std::to_string(i));

      m.addConstr(getAccel(t, 0, i) <= a_max_, "MaxAccel_t" + std::to_string(t) + "_axis_" + std::to_string(i));
      m.addConstr(getAccel(t, 0, i) >= -a_max_, "MinAccel_t" + std::to_string(t) + "_axis_" + std::to_string(i));

      m.addConstr(getJerk(t, 0, i) <= j_max_, "MaxJerk_t" + std::to_string(t) + "_axis_" + std::to_string(i));
      m.addConstr(getJerk(t, 0, i) >= -j_max_, "MinJerk_t" + std::to_string(t) + "_axis_" + std::to_string(i));
    }
  }
}

/*void SolverGurobi::setQ(double q)
{
  q_ = q;
}*/

void SolverGurobi::set_max(double max_values[3])
{
  v_max_ = max_values[0];
  a_max_ = max_values[1];
  j_max_ = max_values[2];
  std::cout << "MAX Values:" << v_max_ << std::endl;
  std::cout << "MAX Values:" << a_max_ << std::endl;
  std::cout << "MAX Values:" << j_max_ << std::endl;

  setMaxConstraints();
}

void SolverGurobi::setFactorInitialAndFinalAndIncrement(double factor_initial, double factor_final,
                                                        double factor_increment)
{
  factor_initial_ = factor_initial;
  factor_final_ = factor_final;
  factor_increment_ = factor_increment;
}

bool SolverGurobi::genNewTraj()
{
  bool solved = false;

  // std::cout << "Factor_initial= " << factor_initial_ << std::endl;
  // std::cout << "Factor_final= " << factor_final_ << std::endl;
  trials_ = 0;

  /*  std::cout << "A es esto:\n";
    // std::cout << "Number of rows= " << bb.rows() << std::endl;
    std::cout << polytopes_[0].A() << std::endl;
    std::cout << "B es esto:\n";
    std::cout << polytopes_[0].b().transpose() << std::endl;*/

  if (factor_initial_ < 1)
  {
    std::cout << "factor_initial_ is less than one, it doesn't make sense" << std::endl;
  }

  // if (mode_ == WHOLE_TRAJ)
  //{
  runtime_ms_ = 0;

  for (double i = factor_initial_; i <= factor_final_ && solved == false && cb_.should_terminate_ == false;
       i = i + factor_increment_)
  {
    trials_ = trials_ + 1;
    findDT(i);
    // std::cout << "Going to try with dt_= " << dt_ << ", should_terminate_=" << cb_.should_terminate_ << std::endl;
    setPolytopesConstraints();
    // std::cout << "Setting x0" << std::endl;
    setConstraintsX0();
    setConstraintsXf();
    // std::cout << "Setting dynamic constraints" << std::endl;
    setDynamicConstraints();
    // setDistanceConstraints();
    setObjective();
    resetXandU();
    solved = callOptimizer();
    /*    if (solved == true)
        {
          solved = isWmaxSatisfied();
        }*/
    if (solved == true)  // solved and Wmax is satisfied
    {
      factor_that_worked_ = i;
      // std::cout << "Factor= " << i << "(dt_= " << dt_ << ")---> worked" << std::endl;
    }
    else
    {
      // std::cout << "Factor= " << i << "(dt_= " << dt_ << ")---> didn't worked" << std::endl;
    }
  }

  cb_.should_terminate_ = false;  // Should be at the end of genNewTaj, not at the beginning

  //}

  return solved;
}

void SolverGurobi::setThreads(int threads)
{
  m.set("Threads", std::to_string(threads));
}

void SolverGurobi::setVerbose(int verbose)
{
  m.set("OutputFlag", std::to_string(verbose));  // 1 if you want verbose, 0 if not
}

void SolverGurobi::setWMax(double w_max)
{
  w_max_ = w_max;
}

void SolverGurobi::findDT(double factor)
{
  // double dt = 2 * getDTInitial();
  dt_ = factor * std::max(getDTInitial(), 2 * DC);
  // std::cout << "Trying dt=" << dt_ << std::endl;
  // dt_ = 1;
}

void SolverGurobi::setDynamicConstraints()
{
  // Remove the previous dynamic constraints
  for (int i = 0; i < dyn_cons.size(); i++)
  {
    m.remove(dyn_cons[i]);
  }
  dyn_cons.clear();

  // Dynamic Constraints
  for (int t = 0; t < N_ - 1; t++)  // From 0....N_-2
  {
    for (int i = 0; i < 3; i++)
    {
      dyn_cons.push_back(m.addConstr(getPos(t, dt_, i) == getPos(t + 1, 0, i),
                                     "ContPos_t" + std::to_string(t) + "_axis" + std::to_string(i)));  // Continuity in
                                                                                                       // position
      dyn_cons.push_back(m.addConstr(getVel(t, dt_, i) == getVel(t + 1, 0, i),
                                     "ContVel_t" + std::to_string(t) + "_axis" + std::to_string(i)));  // Continuity in
                                                                                                       // velocity
      dyn_cons.push_back(
          m.addConstr(getAccel(t, dt_, i) == getAccel(t + 1, 0, i),
                      "ContAccel_t" + std::to_string(t) + "_axis" + std::to_string(i)));  // Continuity in acceleration
    }
  }
}

// For the Jackal:
bool SolverGurobi::isWmaxSatisfied()
{
  for (int n = 0; n < N_; n++)
  {
    double xd = getVel(n, 0, 0).getValue();
    double yd = getVel(n, 0, 1).getValue();
    double xd2 = getAccel(n, 0, 0).getValue();
    double yd2 = getAccel(n, 0, 1).getValue();

    double numerator = xd * yd2 - yd * xd2;
    double denominator = xd * xd + yd * yd;
    double w_desired = (denominator > 0.001) ? fabs(numerator / denominator) : 0.5 * w_max_;

    if (w_desired > w_max_)
    {
      std::cout << "w_desired > than w_max: " << w_desired << " > " << w_max_ << "  , solving again" << std::endl;
      return false;
    }
  }
  return true;
}

bool SolverGurobi::callOptimizer()
{
  // int threads = m.get(GRB_IntParam_Threads);

  // std::cout << "Running Gurobi with " << threads << " cthreads" << std::endl;

  bool solved = true;
  // std::cout << "CALLING OPTIMIZER OF GUROBI" << std::endl;

  // Select these parameteres with the tuning Tool of Gurobi

  /*  if (mode_ == RESCUE_PATH)
    {
      m.set("NumericFocus", "3");
      m.set("Presolve", "0");
    }

    if (mode_ == WHOLE_TRAJ)
    {
      m.set("NumericFocus", "3");
      m.set("Presolve", "0");
    }*/

  m.update();
  temporal_ = temporal_ + 1;
  // printf("Writing into model.lp number=%d\n", temporal_);
  // m.write(ros::package::getPath("faster") + "/models/model2_" + std::to_string(temporal_) + ".lp");

  // std::cout << "*************************Starting Optimization" << std::endl;

  // Solve model and capture solution information
  m.optimize();

  auto start = std::chrono::steady_clock::now();
  m.optimize();
  auto end = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  // std::cout << "*************************Finished Optimization: " << elapsed << " ms" << std::endl;
  // std::cout << "*************************Gurobi RUNTIME: " << m.get(GRB_DoubleAttr_Runtime) * 1000 << " ms"
  //          << std::endl;

  runtime_ms_ = runtime_ms_ + m.get(GRB_DoubleAttr_Runtime) * 1000;

  /*  times_log.open("/home/jtorde/Desktop/ws/src/acl-planning/cvx/models/times_log.txt", std::ios_base::app);
    times_log << elapsed << "\n";
    times_log.close();*/

  // printf("Going to check status");
  int optimstatus = m.get(GRB_IntAttr_Status);
  if (optimstatus == GRB_OPTIMAL)
  {
    if (mode_ == WHOLE_TRAJ)
    {
      // m.write(ros::package::getPath("cvx") + "/models/model_wt" + std::to_string(temporal_) + ".lp");
    }
    if (mode_ == RESCUE_PATH)
    {
      // m.write(ros::package::getPath("cvx") + "/models/model_rp" + std::to_string(temporal_) + ".lp");
    }
    // printf("GUROBI SOLUTION: Optimal");

    /*    if (polytopes_cons.size() > 0)  // Print the binary matrix only if I've included the polytope constraints
        {
          std::cout << "Solution: Binary Matrix:" << std::endl;

          for (int poly = 0; poly < b[0].size(); poly++)
          {
            for (int t = 0; t < N_; t++)
            {
              std::cout << b[t][poly].get(GRB_DoubleAttr_X) << " ";
            }
            std::cout << std::endl;
          }
        }
        std::cout << "Solution: Positions:" << std::endl;
        for (int t = 0; t < N_; t++)
        {
          std::cout << getPos(t, 0, 0) << "   ";
          std::cout << getPos(t, 0, 1) << "   ";
          std::cout << getPos(t, 0, 2) << std::endl;
        }

        std::cout << getPos(N_ - 1, dt_, 0) << "   ";
        std::cout << getPos(N_ - 1, dt_, 1) << "   ";
        std::cout << getPos(N_ - 1, dt_, 2) << std::endl;

        std::cout << "Solution: Coefficients d:" << std::endl;
        for (int t = 0; t < N_; t++)
        {
          std::cout << x[t][9].get(GRB_DoubleAttr_X) << "   ";
          std::cout << x[t][10].get(GRB_DoubleAttr_X) << "   ";
          std::cout << x[t][11].get(GRB_DoubleAttr_X) << std::endl;
        }*/
  }

  else
  {
    // total_not_solved = total_not_solved + 1;
    // std::cout << "TOTAL NOT SOLVED" << total_not_solved << std::endl;
    // No solution
    if (mode_ == RESCUE_PATH)
    {
      // m.write("/home/jtorde/Desktop/ws/src/acl-planning/cvx/models/model_rp" + std::to_string(temporal_) +
      //         "dt=" + std::to_string(dt_) + ".lp");
    }
    solved = false;
    if (optimstatus == GRB_INF_OR_UNBD)
    {
      // printf("GUROBI SOLUTION: Unbounded or Infeasible. Maybe too small dt?\n");

      /*      m.computeIIS();  // Compute the Irreducible Inconsistent Subsystem and write it on a file
            m.write("/home/jtorde/Desktop/ws/src/acl-planning/cvx/models/model_rp" + std::to_string(temporal_) +
                    "dt=" + std::to_string(dt_) + ".ilp");*/
    }

    if (optimstatus == GRB_NUMERIC)
    {
      // printf("GUROBI Status: Numerical issues\n");
      // printf("Model may be infeasible or unbounded\n");  // Taken from the Gurobi documentation
    }

    if (optimstatus == GRB_INTERRUPTED)
    {
      printf("GUROBI Status: Interrumped by the user\n");
    }
  }
  return solved;

  // printf("Going out from callOptimizer, optimstatus= %d\n", optimstatus);
  // std::cout << "*************************Finished Optimization" << std::endl;

  /*  std::cout << "\nOBJECTIVE: " << m.get(GRB_DoubleAttr_ObjVal) << std::endl;


*/

  /*  // printf("In callOptimizer\n");
    bool converged = false;

    // ROS_INFO("dt I found= %0.2f", dt_found);
    double dt = getDTInitial();  // 0.025
                                 // ROS_INFO("empezando con, dt = %0.2f", dt);

    double** x;
    int i = 0;
    int r = 0;
    while (1)
    {
      dt = dt + 4 * 0.025;  // To make sure that it will converge in very few iterations (hopefully only 1)
      i = i + 1;
      // printf("Loading default data!\n");

      jerk_load_default_data(dt, v_max_, a_max_, j_max_, x0_, xf_, q_);
      r = jerk_optimize();
      if (r == 1)
      {
        x = jerk_get_state();
        converged = checkConvergence(x[N_]);
      }

      if (converged == 1)
      {
        break;
      }
      // dt += 0.025;
    }

    if (i > 1)
    {
      printf("Iterations = %d\n", i);
      printf("Iterations>1, if you increase dt at the beginning, it would be faster\n");
    }
    // ROS_INFO("Iterations = %d\n", i);
    // ROS_INFO("converged, dt = %f", dt);*/
}

double SolverGurobi::getDTInitial()
{  // TODO: Not sure if this is right or not. Implement also for Jerk? See page 4, (up-right part) of Search-based
   // Motion Planning for Quadrotors using Linear Quadratic Minimum Time Control
  double dt_initial = 0;
  float t_vx = 0;
  float t_vy = 0;
  float t_vz = 0;
  float t_ax = 0;
  float t_ay = 0;
  float t_az = 0;
  float t_jx = 0;
  float t_jy = 0;
  float t_jz = 0;

  /*  std::cout << "get DT, x0_=" << x0_[0] << " " << x0_[1] << " " << x0_[2] << "--> xf_=" << xf_[0] << " " << xf_[1]
              << " " << xf_[2] << std::endl;*/

  t_vx = fabs(xf_[0] - x0_[0]) / v_max_;
  t_vy = fabs(xf_[1] - x0_[1]) / v_max_;
  t_vz = fabs(xf_[2] - x0_[2]) / v_max_;
  // printf("%f\n", t_vx);
  // printf("%f\n", t_vy);
  // printf("%f\n", t_vz);

  /*  printf("times vel: t_ax, t_ay, t_az:\n");
    std::cout << t_vx << "  " << t_vy << "  " << t_vz << std::endl;*/

  float jerkx = copysign(1, xf_[0] - x0_[0]) * j_max_;
  float jerky = copysign(1, xf_[1] - x0_[1]) * j_max_;
  float jerkz = copysign(1, xf_[2] - x0_[2]) * j_max_;
  float a0x = x0_[6];
  float a0y = x0_[7];
  float a0z = x0_[8];
  float v0x = x0_[3];
  float v0y = x0_[4];
  float v0z = x0_[5];

  // Solve For JERK
  // polynomial ax3+bx2+cx+d=0 --> coeff=[d c b a]
  Eigen::Vector4d coeff_jx(x0_[0] - xf_[0], v0x, a0x / 2.0, jerkx / 6.0);
  Eigen::Vector4d coeff_jy(x0_[1] - xf_[1], v0y, a0y / 2.0, jerky / 6.0);
  Eigen::Vector4d coeff_jz(x0_[2] - xf_[2], v0z, a0z / 2.0, jerkz / 6.0);

  /*  std::cout << "Coefficients for jerk" << std::endl;
    std::cout << "Coeffx=" << coeff_jx.transpose() << std::endl;
    std::cout << "Coeffy=" << coeff_jy.transpose() << std::endl;
    std::cout << "Coeffz=" << coeff_jz.transpose() << std::endl;*/

  Eigen::PolynomialSolver<double, Eigen::Dynamic> psolve_jx(coeff_jx);
  Eigen::PolynomialSolver<double, Eigen::Dynamic> psolve_jy(coeff_jy);
  Eigen::PolynomialSolver<double, Eigen::Dynamic> psolve_jz(coeff_jz);

  std::vector<double> realRoots_jx;
  std::vector<double> realRoots_jy;
  std::vector<double> realRoots_jz;
  psolve_jx.realRoots(realRoots_jx);
  psolve_jy.realRoots(realRoots_jy);
  psolve_jz.realRoots(realRoots_jz);

  t_jx = MinPositiveElement(realRoots_jx);
  t_jy = MinPositiveElement(realRoots_jy);
  t_jz = MinPositiveElement(realRoots_jz);

  /*  printf("times jerk: t_jx, t_jy, t_jz:\n");
    std::cout << t_jx << "  " << t_jy << "  " << t_jz << std::endl;*/

  float accelx = copysign(1, xf_[0] - x0_[0]) * a_max_;
  float accely = copysign(1, xf_[1] - x0_[1]) * a_max_;
  float accelz = copysign(1, xf_[2] - x0_[2]) * a_max_;

  // Solve For ACCELERATION
  // polynomial ax2+bx+c=0 --> coeff=[c b a]
  Eigen::Vector3d coeff_ax(x0_[0] - xf_[0], v0x, 0.5 * accelx);
  Eigen::Vector3d coeff_ay(x0_[1] - xf_[1], v0y, 0.5 * accely);
  Eigen::Vector3d coeff_az(x0_[2] - xf_[2], v0z, 0.5 * accelz);

  /*  std::cout << "Coefficients for accel" << std::endl;
    std::cout << "coeff_ax=" << coeff_ax.transpose() << std::endl;
    std::cout << "coeff_ay=" << coeff_ay.transpose() << std::endl;
    std::cout << "coeff_az=" << coeff_az.transpose() << std::endl;*/

  Eigen::PolynomialSolver<double, Eigen::Dynamic> psolve_ax(coeff_ax);
  Eigen::PolynomialSolver<double, Eigen::Dynamic> psolve_ay(coeff_ay);
  Eigen::PolynomialSolver<double, Eigen::Dynamic> psolve_az(coeff_az);

  std::vector<double> realRoots_ax;
  std::vector<double> realRoots_ay;
  std::vector<double> realRoots_az;
  psolve_ax.realRoots(realRoots_ax);
  psolve_ay.realRoots(realRoots_ay);
  psolve_az.realRoots(realRoots_az);

  t_ax = MinPositiveElement(realRoots_ax);
  t_ay = MinPositiveElement(realRoots_ay);
  t_az = MinPositiveElement(realRoots_az);

  /*  // Solve equation xf=x0+v0t+0.5*a*t^2
    Eigen::Vector3f coeff3x(0.5 * accelx, v0x, x0_[0] - xf_[0]);
    Eigen::Vector3f coeff3y(0.5 * accely, v0y, x0_[1] - xf_[1]);
    Eigen::Vector3f coeff3z(0.5 * accelz, v0z, x0_[2] - xf_[2]);



    std::cout << "Coefficients for accel" << std::endl;
    std::cout << "Coeff3x=" << coeff3x.transpose() << std::endl;
    std::cout << "Coeff3y=" << coeff3y.transpose() << std::endl;
    std::cout << "Coeff3z=" << coeff3z.transpose() << std::endl;

    t_ax = solvePolynomialOrder2(coeff3x);
    t_ay = solvePolynomialOrder2(coeff3y);
    t_az = solvePolynomialOrder2(coeff3z);*/

  /*  printf("times accel: t_ax, t_ay, t_az:\n");
    std::cout << t_ax << "  " << t_ay << "  " << t_az << std::endl;
  */
  dt_initial = std::max({ t_vx, t_vy, t_vz, t_ax, t_ay, t_az, t_jx, t_jy, t_jz }) / N_;
  if (dt_initial > 10000)  // happens when there is no solution to the previous eq.
  {
    printf("there is not a solution to the previous equations\n");
    dt_initial = 0;
  }
  // printf("returning dt_initial=%f\n", dt_initial);
  return dt_initial;
}

GRBLinExpr SolverGurobi::getPos(int t, double tau, int ii)
{
  GRBLinExpr pos = x[t][0 + ii] * tau * tau * tau + x[t][3 + ii] * tau * tau + x[t][6 + ii] * tau + x[t][9 + ii];
  // std::cout << "x tiene size" << x.size() << std::endl;
  // std::cout << "getPos devuelve=" << pos << std::endl;
  return pos;
}

GRBLinExpr SolverGurobi::getVel(int t, double tau, int ii)
{  // t is the segment, tau is the time inside a specific segment (\in[0,dt], i is the axis)

  GRBLinExpr vel = 3 * x[t][0 + ii] * tau * tau + 2 * x[t][3 + ii] * tau + x[t][6 + ii];
  return vel;
}

GRBLinExpr SolverGurobi::getAccel(int t, double tau, int ii)
{  // t is the segment, tau is the time inside a specific segment(\in[0, dt], i is the axis)

  GRBLinExpr accel = 6 * x[t][0 + ii] * tau + 2 * x[t][3 + ii];
  return accel;
}

GRBLinExpr SolverGurobi::getJerk(int t, double tau, int ii)
{  // t is the segment, tau is the time inside a specific segment (\in[0,dt], i is the axis)

  GRBLinExpr jerk = 6 * x[t][0 + ii];  // Note that here tau doesn't appear (makes sense)
  return jerk;
}

// Coefficient getters: At^3 + Bt^2 + Ct + D  , t \in [0, dt_]
GRBLinExpr SolverGurobi::getA(int t, int ii)  // interval, axis
{
  return x[t][0 + ii];
}

GRBLinExpr SolverGurobi::getB(int t, int ii)  // interval, axis
{
  return x[t][3 + ii];
}

GRBLinExpr SolverGurobi::getC(int t, int ii)  // interval, axis
{
  return x[t][6 + ii];
}

GRBLinExpr SolverGurobi::getD(int t, int ii)  // interval, axis
{
  return x[t][9 + ii];
}

// Coefficients Normalized: At^3 + Bt^2 + Ct + D  , t \in [0, 1]
GRBLinExpr SolverGurobi::getAn(int t, int ii)  // interval, axis
{
  return x[t][0 + ii] * dt_ * dt_ * dt_;
}

GRBLinExpr SolverGurobi::getBn(int t, int ii)  // interval, axis
{
  return x[t][3 + ii] * dt_ * dt_;
}

GRBLinExpr SolverGurobi::getCn(int t, int ii)  // interval, axis
{
  return x[t][6 + ii] * dt_;
}

GRBLinExpr SolverGurobi::getDn(int t, int ii)  // interval, axis
{
  return x[t][9 + ii];
}

// Control Points (of the splines) getters
std::vector<GRBLinExpr> SolverGurobi::getCP0(int t)  // Control Point 0 of interval t
{                                                    // Control Point 0 is initial position
                                                     // std::cout << "Getting CP0" << std::endl;
  std::vector<GRBLinExpr> cp = { getPos(t, 0, 0), getPos(t, 0, 1), getPos(t, 0, 2) };
  return cp;
}

std::vector<GRBLinExpr> SolverGurobi::getCP1(int t)  // Control Point 1 of interval t
{
  GRBLinExpr cpx = (getCn(t, 0) + 3 * getDn(t, 0)) / 3;
  GRBLinExpr cpy = (getCn(t, 1) + 3 * getDn(t, 1)) / 3;
  GRBLinExpr cpz = (getCn(t, 2) + 3 * getDn(t, 2)) / 3;
  std::vector<GRBLinExpr> cp = { cpx, cpy, cpz };
  return cp;
}

std::vector<GRBLinExpr> SolverGurobi::getCP2(int t)  // Control Point 2 of interval t
{
  GRBLinExpr cpx = (getBn(t, 0) + 2 * getCn(t, 0) + 3 * getDn(t, 0)) / 3;
  GRBLinExpr cpy = (getBn(t, 1) + 2 * getCn(t, 1) + 3 * getDn(t, 1)) / 3;
  GRBLinExpr cpz = (getBn(t, 2) + 2 * getCn(t, 2) + 3 * getDn(t, 2)) / 3;
  std::vector<GRBLinExpr> cp = { cpx, cpy, cpz };
  return cp;
}

std::vector<GRBLinExpr> SolverGurobi::getCP3(int t)  // Control Point 3 of interval t
{                                                    // Control Point 3 is end position
  std::vector<GRBLinExpr> cp = { getPos(t, dt_, 0), getPos(t, dt_, 1), getPos(t, dt_, 2) };
  return cp;
}