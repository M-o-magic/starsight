# RS_StarSight 代码库说明

本代码库实现了遥感多模态数据集 **Valid / VRSBench / MME** 三个赛题的评测流程。  
整体代码支持不同子任务的模型推理、指标计算，并可在 Docker 容器中"一键式"运行（Valid 数据集），或按照功能拆分运行（VRSBench、MME）。

---

## 环境要求

- Python ≥ 3.8
- PyTorch ≥ 2.0，支持 GPU（CUDA 11.8/12.1，需配合 NVIDIA Container Toolkit）
- 推荐显存 ≥ 24 GB（部分大模型任务需要）
- 依赖库请参考 `docker/requirements.txt`

---

## 数据集说明与评测方式

### 1. Valid 数据集

- 提供了完整的 **一键式评测脚本**：  
  [`src/datasets/valid/run_valid_eval.sh`](src/datasets/valid/run_valid_eval.sh)

- 脚本功能覆盖：
    - JSON 切分与子任务生成
    - 变更检测（SkySense + 建筑变化检测）
    - 异常检测与推理
    - 复杂计数推理
    - 环境条件推理
    - 对象分类、颜色识别、空间关系
    - 总体计数 / 路径规划
    - 整体与区域地类分类
    - 区域计数
    - 运动状态识别（FM9G + OBB 检测器）
    - 精度计算与汇总

- 使用方式：
    ```bash
    # 确保权重与数据已挂载到容器对应路径
    bash src/datasets/valid/run_valid_eval.sh
    ```

- 输出：所有任务的 `eval_*.json` 结果与最终精度统计，保存于：
    ```
    src/datasets/valid/output/
    ```

### 2. VRSBench 数据集

#### VQA 任务

**流程：**

1. 切分 VQA JSON
    ```bash
    python src/datasets/vrsbench/vqa/split_vqa.py --input <json> --outdir <sub_json_dir>
    ```

2. 按子功能脚本运行，每个脚本需在指定行配置参数路径：
    - **Object Color**：`src/datasets/vrsbench/vqa/raw/val_color_withobb_crop.py` （505 行）
    - **Image**：`src/datasets/vrsbench/vqa/raw/val_image.py` （1314 行）
    - **Object Direction**：`src/datasets/vrsbench/vqa/raw/val_object_direction.py` （325 行）
    - **Object Exist**：`src/datasets/vrsbench/vqa/raw/val_object_exist.py` （311 行）
    - **Object Shape**：`src/datasets/vrsbench/vqa/raw/val_object_shape.py` （316 行）
    - **Object Position**：`src/datasets/vrsbench/vqa/raw/test_json_position_output_bad_case_chw_2025813.py` （446 行）
    - **Object Size**：`src/datasets/vrsbench/vqa/raw/val_object_size.py` （317 行）
    - **Rural or Urban**：`src/datasets/vrsbench/vqa/raw/val_rural_or_urban.py` （315 行）
    - **Scene Type**：`src/datasets/vrsbench/vqa/raw/val_sence_type.py` （317 行）
    - **Object Quantity**：`src/datasets/vrsbench/vqa/raw/val_yolo_count_1.py` （219 行）
    - **Categories**：`src/datasets/vrsbench/vqa/raw/val_vrs_category.py` （239 行配置检测后图像路径）
    - **Reasoning**：`src/datasets/vrsbench/vqa/raw/val_reasoning.py`

3. 辅助：需先运行 YOLO 检测
    - `src/datasets/vrsbench/vqa/raw/yolo_predict.py` （配置权重路径、输入图像文件夹）

#### Ref 任务

- **单 GPU**：`src/datasets/vrsbench/ref/test_json_position_ref.py`
    - 229 行：YOLO 权重
    - 232 行：大模型路径
    - 241、269 行：输入文件路径
    - 350、362、381 行：输出地址

- **多 GPU**：`src/datasets/vrsbench/ref/test_json_position_ref_multi_gpu.py`
    - 484 行：路径配置

### 3. MME 数据集

三个功能模块：

- **VQA - Color**：`src/datasets/mme/MME_color_818_positional_test_seting_56_67.py` （610 行）
- **Count**：`src/datasets/mme/MME_count_819_positional_299.py` （605 行）
- **Position**：`src/datasets/mme/val_model_position_819.py` （11 行）

> 以上脚本均需手动在指定行修改权重、输入、输出路径。

---

## Docker 使用方式

### 构建镜像

```bash
docker build -f docker/Dockerfile -t rs-eval:1.0 .

```
### 导入镜像

```bash
unzip docker_image.zip
docker load -i docker_image.tar
```
### 导入镜像

```bash
docker run --rm -it --gpus all \
  -v /ABS/path/valid_images:/data/cj/valid_contest/valid_images \
  -v /ABS/path/tool:/data/cj/RS_StarSight/tool \
  -v /ABS/path/output:/data/cj/RS_StarSight/src/datasets/valid/output \
  rs-eval:1.0
```
## 注意事项

- VRSBench 与 MME 的评测脚本未完全封装为一键式，需要在代码中修改路径后单独运行
- 输出 JSON 均默认写入各自 output/ 目录
- **Position**：`src/datasets/mme/val_model_position_819.py` （11 行）

## 总结
- **Valid 数据集**：提供`run_valid_eval.sh` 支持"一键式"全流程评测
- **Valid 数据集**：提供 VQA/Ref 功能性脚本，需手动配置参数
- **MME 数据集**：提交时将环境、依赖和代码通过 Docker 封装，确保评测方可复现
