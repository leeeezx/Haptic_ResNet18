import matplotlib.font_manager as fm

# 方式1：遍历打印所有Matplotlib已知字体的名称
for font in fm.fontManager.ttflist:
    print(font.name)