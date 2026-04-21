export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm

# lerobot-calibrate \
#     --teleop.type=so101_leader \
#     --teleop.port=/dev/ttyACM1 \
#     --teleop.id=leader_arm