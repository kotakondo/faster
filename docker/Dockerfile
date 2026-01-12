FROM ros:melodic

LABEL Maintainer="Kota Kondo <kkondo@mit.edu>" \
      Description="Ubuntu 18.04, ROS Melodic for FASTER"

ENV ROS_DISTRO=melodic
ENV DEBIAN_FRONTEND=noninteractive

RUN sudo rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
      apt-get upgrade -y && \
      apt-get install -q -y --no-install-recommends git apt-utils wget tmux vim tmuxp make && \
      apt-get autoremove -y && \
      apt-get clean

RUN sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ros-${ROS_DISTRO}-gazebo-ros-pkgs \
     ros-${ROS_DISTRO}-mavros-msgs \
     ros-${ROS_DISTRO}-tf2-sensor-msgs \
     ros-${ROS_DISTRO}-control-toolbox  \
     ros-${ROS_DISTRO}-ros-control  \
     ros-${ROS_DISTRO}-robot-localization  \
     ros-${ROS_DISTRO}-lms1xx  \
     ros-${ROS_DISTRO}-interactive-marker-twist-server  \
     ros-${ROS_DISTRO}-hector-gazebo-plugins  \
     ros-${ROS_DISTRO}-move-base  \
     ros-${ROS_DISTRO}-ros-controllers  \
     ros-${ROS_DISTRO}-pointgrey-camera-description  \
     ros-${ROS_DISTRO}-hardware-interface  \
     ros-${ROS_DISTRO}-message-to-tf  \
     ros-${ROS_DISTRO}-gazebo-ros-control \
     python-pip \
     ros-${ROS_DISTRO}-pcl-ros \
     ros-${ROS_DISTRO}-rqt-gui \
     ros-${ROS_DISTRO}-rqt-gui-py \
     ros-${ROS_DISTRO}-rviz \
     python3-catkin-tools \
     rqt*

RUN sudo apt install python3-pycryptodome -y
RUN python -m pip install pyquaternion pycryptodome 

RUN pip install numpy rospkg matplotlib scipy pandas netifaces

# Livox-SDK2
WORKDIR /home/kota/
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git
WORKDIR /home/kota/Livox-SDK2
RUN mkdir build
WORKDIR /home/kota/Livox-SDK2/build
RUN cmake .. && make -j
RUN sudo make install

# Install Faster
RUN mkdir -p /home/kota/ws/src
WORKDIR /home/kota/ws/src
ARG DUMMY=unknown
RUN DUMMY=${DUMMY} git clone https://github.com/kotakondo/faster.git
RUN wstool init
RUN wstool merge ./faster/faster/install/faster.rosinstall
RUN wstool merge ./faster/faster/install/faster_ground_robot.rosinstall
RUN wstool update -j8
WORKDIR /home/kota/ws

## Install gurobi
RUN wget https://packages.gurobi.com/9.5/gurobi9.5.2_linux64.tar.gz -P . && \
      tar -xzf gurobi9.5.2_linux64.tar.gz && \
      rm gurobi9.5.2_linux64.tar.gz && \
      mv gurobi952/ /opt && \
      . /opt/ros/melodic/setup.sh
WORKDIR /opt/gurobi952/linux64/src/build
RUN apt-get update && apt-get install make && \
      apt install build-essential -y --no-install-recommends && \
      make && \
      cp libgurobi_c++.a ../../lib/

# Update acl-gazebo
WORKDIR /home/kota/ws/src/acl-gazebo
RUN git checkout benchmark_for_dynus

# Build Faster
WORKDIR /home/kota/ws
RUN catkin config -DCMAKE_BUILD_TYPE=Release
RUN export GUROBI_HOME="/opt/gurobi952/linux64" && \
      export PATH="${PATH}:${GUROBI_HOME}/bin" && \
      export LD_LIBRARY_PATH="${GUROBI_HOME}/lib" && \
      . /opt/ros/melodic/setup.sh && catkin build

# livox_ros_driver2
RUN mkdir -p /home/kota/livox_ros_ws/src
WORKDIR /home/kota/livox_ros_ws/src
RUN git clone https://github.com/Livox-SDK/livox_ros_driver2.git
WORKDIR /home/kota/livox_ros_ws/src/livox_ros_driver2
RUN ./build.sh ROS1

# Mid360_simulation_plugin
RUN mkdir -p /home/kota/mid360_ws/src
WORKDIR /home/kota/mid360_ws/src
RUN git clone https://github.com/kotakondo/Mid360_simulation_plugin.git
WORKDIR /home/kota/mid360_ws
RUN bash -c "source /opt/ros/melodic/setup.bash && source /home/kota/ws/devel/setup.bash && catkin_make -DBUILD_TYPE=Release"

WORKDIR /home/kota/ws

RUN echo "export GUROBI_HOME=\"/opt/gurobi952/linux64\"" >> ~/.bashrc
RUN echo "export PATH=\"\${PATH}:\${GUROBI_HOME}/bin\"" >> ~/.bashrc
RUN echo "export LD_LIBRARY_PATH=\"\${GUROBI_HOME}/lib\"" >> ~/.bashrc

# Set up bashrc
RUN echo "source /home/kota/ws/devel/setup.bash" >> ~/.bashrc

# Set up ROS
# RUN echo "export ROS_MASTER_URI=http://192.168.0.1:11311" >> ~/.bashrc
# RUN echo "export ROS_MASTER_URI=http://172.17.0.2:11311" >> ~/.bashrc
# RUN echo "export ROS_MASTER_URI=http://localhost:11311" >> ~/.bashrc
RUN echo "export ROS_MASTER_URI=http://127.0.0.1:11311" >> ~/.bashrc
RUN echo "export ROS_IP=localhost" >> ~/.bashrc
RUN echo "export ROS_HOSTNAME=localhost" >> ~/.bashrc

# shortcut 
RUN echo "alias tks='tmux kill-server'" >> ~/.bashrc
RUN echo "alias sb='source ~/.bashrc'" >> ~/.bashrc
RUN echo "alias eb='code ~/.bashrc'" >> ~/.bashrc
RUN echo "alias gs='git status'" >> ~/.bashrc
RUN echo "alias gp='git push'" >> ~/.bashrc
RUN echo "alias roscd='cd /home/kota/ws'" >> ~/.bashrc
RUN echo "alias cb='roscd && catkin build && sb'" >> ~/.bashrc
RUN echo "alias ss='roscd && source devel/setup.bash'" >> ~/.bashrc
RUN echo "alias cbf='clear && roscd && catkin build && ss'" >> ~/.bashrc
RUN echo "alias run_faster='tmuxp load /home/kota/ws/src/faster/docker/faster_sim.yml'" >> ~/.bashrc

# # Set up Entry Point
# COPY ./faster_ground_robot.yml /
# RUN chmod +x /faster_ground_robot.yml
# ENTRYPOINT ["tmuxp", "load", "/faster_ground_robot.yml"]

# COPY ./faster_sim.yml /
# RUN chmod +x /faster_sim.yml
# ENTRYPOINT ["tmuxp", "load", "/faster_sim.yml"]
