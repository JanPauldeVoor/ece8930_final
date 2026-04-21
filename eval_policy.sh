
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -m lerobot.async_inference.robot_client \
    --server_address=10.125.2.6:8080 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_so101 \
    --robot.cameras="{ wrist: {type: intelrealsense, serial_number_or_name: 335122272701, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 346522073763, width: 640, height: 480, fps: 30}}" \
    --task="Put the cube in the green bin" \
    --policy_type=pi0 \
    --pretrained_name_or_path=jdevoor/pi0_b16_policy \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True