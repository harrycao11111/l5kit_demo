#!/bin/bash

# 设置文件大小阈值（例如 50MB）
SIZE_LIMIT=50

# 查找当前目录下大于指定大小的文件并跟踪
find . -type f -size +"${SIZE_LIMIT}M" | while read FILE; do
    echo "Tracking $FILE with Git LFS"
    git lfs track "$FILE"
done

# 提交 .gitattributes 文件
git add .gitattributes
git commit -m "Track large files with Git LFS"
