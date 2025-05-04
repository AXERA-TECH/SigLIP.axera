# 模型转换

## 创建虚拟环境

```
conda create -n siglip python=3.9
conda activate siglip
```

## 安装依赖

```
git clone https://github.com/ml-inory/siglip.axera.git
cd model_convert
pip install -r requirements.txt
```

## 导出模型（PyTorch -> ONNX）

此处导出 **siglip-so400m-patch14-384** 模型，其他规格请根据需要选择,导出成功后会在model_convert/onnx下生成两个encoder的onnx文件

```python
# 同时导出 vision encoder 和 text encoder:
python model_convert/onnx/export_onnx.py
```

## 转换模型（ONNX -> Axera）

使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 准备适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

### 准备数据集
- 下载vision encoder数据集

- 生成text encoder数据集
    ```
    # 生成数据
    python model_convert/cali_datda/imagenet_dataset.py
    
    # 打包
    cd model_convert/cali_data
    zip -r text_cali.zip text_cali
    ````

### 模型转换

#### 修改配置文件

修改以 `config/vision` 和 `config/text` 路径下的 json 文件中 calibration_dataset 字段为 **准备数据集** 步骤中的 `zip` 文件路径

#### Pulsar2 build

参考命令如下：

**vision encoder**

```
pulsar2 build --config ./siglip-so400m-patch14-384_vision.json --input ./siglip-so400m-patch14-384_vision.onnx --output_dir /output/vision/u16u8
```

**text encoder**

```
pulsar2 build --config ./siglip-so400m-patch14-384_text.json --input ./siglip-so400m-patch14-384_text.onnx --output_dir /output/text/u16
```
