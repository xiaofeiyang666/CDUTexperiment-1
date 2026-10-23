from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "数字识别实验报告.docx"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, color="000000"):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = "Microsoft YaHei"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_table_borders(table):
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "D9D9D9")


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="Body Text")
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        r.bold = True
        p.add_run(text[len(bold_prefix):])
    else:
        p.add_run(text)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(89, 89, 89)


doc = Document()
section = doc.sections[0]
section.top_margin = Cm(2.2)
section.bottom_margin = Cm(2.0)
section.left_margin = Cm(2.4)
section.right_margin = Cm(2.4)

styles = doc.styles
styles["Normal"].font.name = "Microsoft YaHei"
styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
styles["Normal"].font.size = Pt(10.5)
for name in ["Heading 1", "Heading 2", "Title"]:
    styles[name].font.name = "Microsoft YaHei"
    styles[name]._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    styles[name].font.color.rgb = RGBColor(0, 0, 0)
styles["Title"].font.size = Pt(22)
styles["Title"].font.bold = True
styles["Heading 1"].font.size = Pt(14)
styles["Heading 1"].font.bold = True
styles["Heading 2"].font.size = Pt(11.5)
styles["Heading 2"].font.bold = True

title = doc.add_paragraph(style="Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.add_run("实验一 数字识别实验报告")
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_after = Pt(16)
run = sub.add_run("基于 MNIST 数据集的 ResNet18 图像分类实验")
run.font.size = Pt(11)
run.font.color.rgb = RGBColor(89, 89, 89)

info = doc.add_table(rows=2, cols=4)
info.autofit = True
set_table_borders(info)
for row in info.rows:
    for cell in row.cells:
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
labels = [("姓名", "夏富源"), ("学号", "202421020120"), ("班级", "2024210201"), ("实验日期", "2026年10月17日")]
for i, (label, value) in enumerate(labels):
    set_cell_text(info.cell(0, i), label, bold=True, color="FFFFFF")
    set_cell_shading(info.cell(0, i), "44546A")
    set_cell_text(info.cell(1, i), value)

add_heading(doc, "一 实验目的", 1)
add_body(doc, "本次实验使用 MNIST 手写数字数据集，训练一个自定义的 ResNet18 网络，对 0 到 9 的数字图片进行分类。训练时记录每轮的损失和验证准确率，训练结束后再用测试集检查识别效果。")

add_heading(doc, "二 实验环境与数据", 1)
env = doc.add_table(rows=1, cols=2)
set_table_borders(env)
for i, h in enumerate(["项目", "配置"]):
    set_cell_text(env.cell(0, i), h, bold=True, color="FFFFFF")
    set_cell_shading(env.cell(0, i), "44546A")
for k, v in [
    ("操作环境", "Windows + Conda 环境 env"),
    ("Python 与深度学习框架", "Python（Conda env）+ PyTorch 2.4.1 + torchvision 0.20.0"),
    ("硬件", "NVIDIA GeForce RTX 4060 Laptop GPU，使用 CUDA"),
    ("数据集", "MNIST；训练集 60000 张，测试集 10000 张，灰度图像"),
    ("预处理", "Resize 到 224×224，ToTensor；训练集按 80%/20% 划分训练集和验证集"),
]:
    row = env.add_row().cells
    set_cell_text(row[0], k)
    set_cell_text(row[1], v)

add_heading(doc, "三 实验步骤", 1)
steps = [
    "准备 Conda 环境并安装 PyTorch、torchvision、numpy、pandas、matplotlib 等依赖。",
    "下载并读取 MNIST 数据集，将图像缩放到 224×224，并转换为张量。",
    "构建 ResNet18 模型，使用交叉熵损失函数和 Adam 优化器，学习率为 1e-3，batch size 为 32。",
    "训练 5 个 epoch；每个 epoch 完成训练集前向传播、反向传播和验证集评估，并保存验证准确率最高的模型权重。",
    "使用最佳权重在测试集上进行评估，输出最终测试准确率，并生成正确和错误预测样本图。",
]
for i, step in enumerate(steps, 1):
    p = doc.add_paragraph(style="Body Text")
    p.paragraph_format.left_indent = Cm(0.4)
    p.paragraph_format.space_after = Pt(3)
    p.add_run(f"{i}. ").bold = True
    p.add_run(step)

add_heading(doc, "四 模型架构", 1)
add_body(doc, "输入图片是单通道的 224×224 灰度图。网络前面使用 7×7 卷积、ReLU、BatchNorm 和最大池化，后面接四组残差块，通道数依次为 64、128、256 和 512。最后经过自适应平均池化和全连接层，输出 10 个数字类别。")
arch = doc.add_table(rows=1, cols=3)
set_table_borders(arch)
for i, h in enumerate(["模块", "主要结构", "输出特征"]):
    set_cell_text(arch.cell(0, i), h, bold=True, color="FFFFFF")
    set_cell_shading(arch.cell(0, i), "44546A")
for vals in [
    ("输入与 b1", "Conv7×7 / ReLU / BatchNorm / MaxPool", "64 通道"),
    ("b2", "2 个 Residual(64, 64)", "64 通道"),
    ("b3", "Residual(64,128,stride=2) + Residual(128,128)", "128 通道"),
    ("b4", "Residual(128,256,stride=2) + Residual(256,256)", "256 通道"),
    ("b5", "Residual(256,512,stride=2) + Residual(512,512)", "512 通道"),
    ("b6", "AdaptiveAvgPool2d + Flatten + Linear(512,10)", "10 类输出"),
]:
    cells = arch.add_row().cells
    for i, value in enumerate(vals):
        set_cell_text(cells[i], value)

add_heading(doc, "五 训练过程与结果", 1)
add_body(doc, "从训练记录看，训练损失由 0.1307 降到 0.0263，验证损失由 0.0485 降到 0.0317。验证准确率从 98.52% 提高到 99.10%，第 5 轮的结果最好，所以保存了第 5 轮的模型参数。")
metrics = doc.add_table(rows=1, cols=5)
set_table_borders(metrics)
for i, h in enumerate(["Epoch", "Train loss", "Val loss", "Train acc", "Val acc"]):
    set_cell_text(metrics.cell(0, i), h, bold=True, color="FFFFFF")
    set_cell_shading(metrics.cell(0, i), "44546A")
rows = [
    (1, "0.1307", "0.0485", "0.9605", "0.9852"),
    (2, "0.0522", "0.0524", "0.9835", "0.9828"),
    (3, "0.0396", "0.0386", "0.9879", "0.9884"),
    (4, "0.0334", "0.0350", "0.9896", "0.9894"),
    (5, "0.0263", "0.0317", "0.9919", "0.9910"),
]
for vals in rows:
    cells = metrics.add_row().cells
    for i, value in enumerate(vals):
        set_cell_text(cells[i], value)

doc.add_picture(str(ROOT / "training_curves.png"), width=Cm(16.2))
add_caption(doc, "图 1 训练损失、验证损失及训练验证准确率变化曲线")

add_heading(doc, "六 测试结果与预测分析", 1)
add_body(doc, "用保存的模型测试 10000 张图片，其中 9914 张预测正确，86 张预测错误，测试准确率为 99.14%。这个结果和验证集准确率比较接近。")
add_body(doc, "错误样本中比较常见的是 3 和 5、0 和 6、7 和 2 之间的混淆。这些图片的写法比较接近，单看笔画时也容易判断错。报告中的样本图列出了正确和错误的预测结果。")
doc.add_picture(str(ROOT / "test_prediction_samples.png"), width=Cm(16.2))
add_caption(doc, "图 2 测试集预测示例；绿色为正确预测，红色为错误预测")

add_heading(doc, "七 源码与预训练模型", 1)
add_body(doc, "实验源码已经上传到 GitHub，训练好的 best_model.pth 放在 v1.0 Release 附件中。")
links = doc.add_table(rows=1, cols=2)
set_table_borders(links)
for i, h in enumerate(["资源", "地址"]):
    set_cell_text(links.cell(0, i), h, bold=True, color="FFFFFF")
    set_cell_shading(links.cell(0, i), "44546A")
for vals in [
    ("GitHub 源码", "https://github.com/xiaofeiyang666/CDUTexperiment-1"),
    ("预训练模型下载", "https://github.com/xiaofeiyang666/CDUTexperiment-1/releases/download/v1.0/best_model.pth"),
    ("本地模型文件", "ResNet_model/best_model.pth"),
]:
    cells = links.add_row().cells
    for i, value in enumerate(vals):
        set_cell_text(cells[i], value)
add_body(doc, "源码仓库中保留了模型定义、训练、测试和结果生成脚本。预训练模型可以从 v1.0 Release 下载。")

add_heading(doc, "八 实验总结", 1)
add_body(doc, "这次实验按照数据预处理、搭建网络、训练和测试的顺序完成。最终测试准确率为 99.14%，训练曲线中的损失逐步下降，准确率逐步提高。后面还可以尝试数据增强或缩小输入图片尺寸，以减少训练时间。")

doc.save(OUT)
print(OUT)
