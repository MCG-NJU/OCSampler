# Demo 示例

## 目录

- [预测视频的动作标签](#预测视频的动作标签)
- [预测视频的时空检测结果](#预测视频的时空检测结果)
- [可视化输入视频的 GradCAM](#可视化输入视频的-GradCAM)
- [使用网络摄像头的实时动作识别](#使用网络摄像头的实时动作识别)
- [滑动窗口预测长视频中不同动作类别](#滑动窗口预测长视频中不同动作类别)

## 预测视频的动作标签

MMAction2 提供如下脚本以预测视频的动作标签。为得到 [0, 1] 间的动作分值，请确保在配置文件中设定 `model['test_cfg'] = dict(average_clips='prob')`。

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} {LABEL_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--fps {FPS}] [--font-size {FONT_SIZE}] [--font-color {FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

可选参数：

- `--use-frames`: 如指定，代表使用帧目录作为输入；否则代表使用视频作为输入。
- `DEVICE_TYPE`: 指定脚本运行设备，支持 cuda 设备（如 `cuda:0`）或 cpu（`cpu`）。默认为 `cuda:0`。
- `FPS`: 使用帧目录作为输入时，代表输入的帧率。默认为 30。
- `FONT_SIZE`: 输出视频上的字体大小。默认为 20。
- `FONT_COLOR`: 输出视频上的字体颜色，默认为白色（ `white`）。
- `TARGET_RESOLUTION`: 输出视频的分辨率，如未指定，使用输入视频的分辨率。
- `RESIZE_ALGORITHM`: 缩放视频时使用的插值方法，默认为 `bicubic`。
- `OUT_FILE`: 输出视频的路径，如未指定，则不会生成输出视频。

示例：

以下示例假设用户的当前目录为 `$MMACTION2`，并已经将所需的模型权重文件下载至目录 `checkpoints/` 下，用户也可以使用所提供的 URL 来直接加载模型权重，文件将会被默认下载至 `$HOME/.cache/torch/checkpoints`。

1. 在 cuda 设备上，使用 TSN 模型进行视频识别：

    ```shell
    # demo.mp4 及 label_map_k400.txt 均来自 Kinetics-400 数据集
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt
    ```

2. 在 cuda 设备上，使用 TSN 模型进行视频识别，并利用 URL 加载模型权重文件：

    ```shell
    # demo.mp4 及 label_map_k400.txt 均来自 Kinetics-400 数据集
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt
    ```

3. 在 CPU 上，使用 TSN 模型进行视频识别，输入为视频抽好的帧：

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --device cpu
    ```

4. 使用 TSN 模型进行视频识别，输出 MP4 格式的识别结果：

    ```shell
    # demo.mp4 及 label_map_k400.txt 均来自 Kinetics-400 数据集
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt --out-filename demo/demo_out.mp4
    ```

5. 使用 TSN 模型进行视频识别，输入为视频抽好的帧，将识别结果存为 GIF 格式：

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --out-filename demo/demo_out.gif
    ```

6. 使用 TSN 模型进行视频识别，输出 MP4 格式的识别结果，并指定输出视频分辨率及缩放视频时使用的插值方法：

    ```shell
    # demo.mp4 及 label_map_k400.txt 均来自 Kinetics-400 数据集
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt --target-resolution 340 256 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

    ```shell
    # demo.mp4 及 label_map_k400.txt 均来自 Kinetics-400 数据集
    # 若 TARGET_RESOLUTION 的任一维度被设置为 -1，视频帧缩放时将保持长宽比
    # 如设定 --target-resolution 为 170 -1，原先长宽为 (340, 256) 的视频帧将被缩放至 (170, 128)
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt --target-resolution 170 -1 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

7. 使用 TSN 模型进行视频识别，输出 MP4 格式的识别结果，指定输出视频中使用红色文字，字体大小为 10 像素：

    ```shell
    # demo.mp4 及 label_map_k400.txt 均来自 Kinetics-400 数据集
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt --font-size 10 --font-color red \
        --out-filename demo/demo_out.mp4
    ```

8. 使用 TSN 模型进行视频识别，输入为视频抽好的帧，将识别结果存为 MP4 格式，帧率设置为 24fps：

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --fps 24 --out-filename demo/demo_out.gif
    ```

## 预测视频的时空检测结果

MMAction2 提供如下脚本以预测视频的时空检测结果。

```shell
python demo/demo_spatiotemporal_det.py --video ${VIDEO_FILE} \
    [--config ${SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE}] \
    [--checkpoint ${SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--action-score-thr ${ACTION_DETECTION_SCORE_THRESHOLD}] \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--out-filename ${OUTPUT_FILENAME}] \
    [--predict-stepsize ${PREDICT_STEPSIZE}] \
    [--output-stepsize ${OUTPUT_STEPSIZE}] \
    [--output-fps ${OUTPUT_FPS}]
```

可选参数：

- `SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE`: 时空检测配置文件路径。
- `SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT`: 时空检测模型权重文件路径。
- `HUMAN_DETECTION_CONFIG_FILE`: 人体检测配置文件路径。
- `HUMAN_DETECTION_CHECKPOINT`: 人体检测模型权重文件路径。
- `HUMAN_DETECTION_SCORE_THRE`: 人体检测分数阈值：默认为 0.9。
- `ACTION_DETECTION_SCORE_THRESHOLD`: 动作检测分数阈值：默认为 0.5。
- `LABEL_MAP`: 所使用的标签映射文件，默认为 `demo/label_map_ava.txt`。
- `DEVICE`:  指定脚本运行设备，支持 cuda 设备（如 `cuda:0`）或 cpu（`cpu`）。默认为 `cuda:0`。
- `OUTPUT_FILENAME`: 输出视频的路径，默认为 `demo/stdet_demo.mp4`。
- `PREDICT_STEPSIZE`: 每 N 帧进行一次预测（以节约计算资源），默认值为 8。
- `OUTPUT_STEPSIZE`: 对于输入视频的每 N 帧，输出 1 帧至输出视频中， 默认值为 4，注意需满足 `PREDICT_STEPSIZE % OUTPUT_STEPSIZE == 0`。
- `OUTPUT_FPS`: 输出视频的帧率，默认值为 6。

示例：

以下示例假设用户的当前目录为 `$MMACTION2`，并已经将所需的模型权重文件下载至目录 `checkpoints/` 下，用户也可以使用所提供的 URL 来直接加载模型权重，文件将会被默认下载至 `$HOME/.cache/torch/checkpoints`。

1. 使用 Faster RCNN 作为人体检测器，SlowOnly-8x8-R101 作为动作检测器。每 8 帧进行一次预测，原视频中每 4 帧输出 1 帧至输出视频中，设置输出视频的帧率为 6。

```shell
python demo/demo_spatiotemporal_det.py --video demo/demo.mp4 \
    --config configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py \
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map demo/label_map_ava.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
```

## 可视化输入视频的 GradCAM

MMAction2 提供如下脚本以可视化输入视频的 GradCAM。

```shell
python demo/demo_gradcam.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--target-layer-name ${TARGET_LAYER_NAME}] [--fps {FPS}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

可选参数：

- `--use-frames`: 如指定，代表使用帧目录作为输入；否则代表使用视频作为输入。
- `DEVICE_TYPE`: 指定脚本运行设备，支持 cuda 设备（如 `cuda:0`）或 cpu（`cpu`）。默认为 `cuda:0`。
- `TARGET_LAYER_NAME`: 需要生成 GradCAM 可视化的网络层名称。
- `FPS`: 使用帧目录作为输入时，代表输入的帧率。默认为 30。
- `TARGET_RESOLUTION`: 输出视频的分辨率，如未指定，使用输入视频的分辨率。
- `RESIZE_ALGORITHM`: 缩放视频时使用的插值方法，默认为 `bilinear`。
- `OUT_FILE`: 输出视频的路径，如未指定，则不会生成输出视频。

示例：

以下示例假设用户的当前目录为 `$MMACTION2`，并已经将所需的模型权重文件下载至目录 `checkpoints/` 下，用户也可以使用所提供的 URL 来直接加载模型权重，文件将会被默认下载至 `$HOME/.cache/torch/checkpoints`。

1. 对于 I3D 模型进行 GradCAM 的可视化，使用视频作为输入，并输出一帧率为 10 的 GIF 文件：

    ```shell
    python demo/demo_gradcam.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
        checkpoints/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth demo/demo.mp4 \
        --target-layer-name backbone/layer4/1/relu --fps 10 \
        --out-filename demo/demo_gradcam.gif
    ```

2. 对于 I3D 模型进行 GradCAM 的可视化，使用视频作为输入，并输出一 GIF 文件，此示例利用 URL 加载模型权重文件：

    ```shell
    python demo/demo_gradcam.py configs/recognition/tsm/tsm_r50_video_inference_1x1x8_100e_kinetics400_rgb.py \
        https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth \
        demo/demo.mp4 --target-layer-name backbone/layer4/1/relu --out-filename demo/demo_gradcam_tsm.gif
    ```

## 使用网络摄像头的实时动作识别

MMAction2 提供如下脚本来进行使用网络摄像头的实时动作识别。为得到 [0, 1] 间的动作分值，请确保在配置文件中设定 `model['test_cfg'] = dict(average_clips='prob')` 。

```shell
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--camera-id ${CAMERA_ID}] [--threshold ${THRESHOLD}] \
    [--average-size ${AVERAGE_SIZE}] [--drawing-fps ${DRAWING_FPS}] [--inference-fps ${INFERENCE_FPS}]
```

可选参数：

- `DEVICE_TYPE`: 指定脚本运行设备，支持 cuda 设备（如 `cuda:0`）或 cpu（`cpu`）。默认为 `cuda:0`。
- `CAMERA_ID`: 摄像头设备的 ID，默认为 0。
- `THRESHOLD`: 动作识别的分数阈值，只有分数大于阈值的动作类型会被显示，默认为 0。
- `AVERAGE_SIZE`: 使用最近 N 个片段的平均结果作为预测，默认为 1。
- `DRAWING_FPS`: 可视化结果时的最高帧率，默认为 20。
- `INFERENCE_FPS`: 进行推理时的最高帧率，默认为 4。

**注**： 若用户的硬件配置足够，可增大可视化帧率和推理帧率以带来更好体验。

示例：

以下示例假设用户的当前目录为 `$MMACTION2`，并已经将所需的模型权重文件下载至目录 `checkpoints/` 下，用户也可以使用所提供的 URL 来直接加载模型权重，文件将会被默认下载至 `$HOME/.cache/torch/checkpoints`。

1. 使用 TSN 模型进行利用网络摄像头的实时动作识别，平均最近 5 个片段结果作为预测，输出大于阈值 0.2 的动作类别：

```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth demo/label_map_k400.txt --average-size 5 \
      --threshold 0.2 --device cpu
```

2. 使用 TSN 模型在 CPU 上进行利用网络摄像头的实时动作识别，平均最近 5 个片段结果作为预测，输出大于阈值 0.2 的动作类别，此示例利用 URL 加载模型权重文件：

```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      demo/label_map_k400.txt --average-size 5 --threshold 0.2 --device cpu
```

3. 使用 I3D 模型在 GPU 上进行利用网络摄像头的实时动作识别，平均最近 5 个片段结果作为预测，输出大于阈值 0.2 的动作类别：

```shell
    python demo/webcam_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth demo/label_map_k400.txt \
      --average-size 5 --threshold 0.2
```

**注:** 考虑到用户所使用的推理设备具有性能差异，可进行如下改动在用户设备上取得更好效果：

1). 更改配置文件中的 `test_pipeline` 下 `SampleFrames` 步骤 （特别是 `clip_len` 与 `num_clips`）。
2). 更改配置文件中的 `test_pipeline` 下的裁剪方式类型（可选项含：`TenCrop`, `ThreeCrop`, `CenterCrop`）。
3). 调低 `AVERAGE_SIZE` 以加快推理。

## 滑动窗口预测长视频中不同动作类别

MMAction2 提供如下脚本来预测长视频中的不同动作类别。为得到 [0, 1] 间的动作分值，请确保在配置文件中设定 `model['test_cfg'] = dict(average_clips='prob')` 。

```shell
python demo/long_video_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    ${OUT_FILE} [--input-step ${INPUT_STEP}] [--device ${DEVICE_TYPE}] [--threshold ${THRESHOLD}]
```

可选参数：

- `OUT_FILE`: 输出视频的路径。
- `INPUT_STEP`: 在视频中的每 N 帧中选取一帧作为输入，默认为 1。
- `DEVICE_TYPE`: 指定脚本运行设备，支持 cuda 设备（如 `cuda:0`）或 cpu（`cpu`）。默认为 `cuda:0`。
- `THRESHOLD`: 动作识别的分数阈值，只有分数大于阈值的动作类型会被显示，默认为 0.01。
- `STRIDE`: 默认情况下，脚本为每帧给出单独预测，较为耗时。可以设定 `STRIDE` 参数进行加速，此时脚本将会为每 `STRIDE x sample_length` 帧做一次预测（`sample_length` 指模型采帧时的时间窗大小，等于 `clip_len x frame_interval`）。例如，若 sample_length 为 64 帧且 `STRIDE` 设定为 0.5，模型将每 32 帧做一次预测。若 `STRIDE` 设为 0，模型将为每帧做一次预测。`STRIDE` 的理想取值为 (0, 1] 间，若大于 1，脚本亦可正常执行。`STRIDE` 默认值为 0。

示例：

以下示例假设用户的当前目录为 `$MMACTION2`，并已经将所需的模型权重文件下载至目录 `checkpoints/` 下，用户也可以使用所提供的 URL 来直接加载模型权重，文件将会被默认下载至 `$HOME/.cache/torch/checkpoints`。

1. 利用 TSN 模型在 CPU 上预测长视频中的不同动作类别，设置 `INPUT_STEP` 为 3（即每 3 帧随机选取 1 帧作为输入），输出分值大于 0.2 的动作类别：

 ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth PATH_TO_LONG_VIDEO demo/label_map_k400.txt PATH_TO_SAVED_VIDEO \
      --input-step 3 --device cpu --threshold 0.2
 ```

2. 利用 TSN 模型在 CPU 上预测长视频中的不同动作类别，设置 `INPUT_STEP` 为 3，输出分值大于 0.2 的动作类别，此示例利用 URL 加载模型权重文件：

 ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      PATH_TO_LONG_VIDEO demo/label_map_k400.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
 ```

3. 利用 TSN 模型在 CPU 上预测网络长视频（利用 URL 读取）中的不同动作类别，设置 `INPUT_STEP` 为 3，输出分值大于 0.2 的动作类别，此示例利用 URL 加载模型权重文件：

 ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4 \
      demo/label_map_k400.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
 ```

4. 利用 I3D 模型在 GPU 上预测长视频中的不同动作类别，设置 `INPUT_STEP` 为 3，动作识别的分数阈值为 0.01：

    ```shell
    python demo/long_video_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth PATH_TO_LONG_VIDEO demo/label_map_k400.txt PATH_TO_SAVED_VIDEO \
    ```
