#!/bin/bash

# 定义要搜索的根目录，可以根据需要修改
ROOT_DIR="./"

# 使用find查找所有.pt文件，并循环处理每一个找到的文件
find "$ROOT_DIR" -type f -name "*.pt" | while read -r FILE; do
    # 获取文件的基本名称（不包括路径）
    BASENAME=$(basename "$FILE")
    
    # 如果文件名包含RUL，则重命名为test
    if [[ $BASENAME == *"RUL"* ]]; then
        NEW_NAME=${BASENAME//RUL/test}
    # 如果文件名包含test，则重命名为RUL
    elif [[ $BASENAME == *"test"* ]]; then
        NEW_NAME=${BASENAME//test/RUL}
    else
        # 如果既没有RUL也没有test，跳过该文件
        continue
    fi
    
    # 构造新的完整路径
    DIR_NAME=$(dirname "$FILE")
    NEW_FILE="$DIR_NAME/$NEW_NAME"
    
    # 重命名文件
    mv "$FILE" "$NEW_FILE"
    echo "Renamed '$FILE' to '$NEW_FILE'"
done